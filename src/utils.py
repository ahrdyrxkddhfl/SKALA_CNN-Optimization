"""공용 유틸: 시드 고정, 데이터 로드, 평가 지표, 학습 루프, 결과 기록"""
import os, csv, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (recall_score, f1_score, accuracy_score,
                             classification_report)

CLASS_NAMES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc',
               'Near-full', 'Random', 'Scratch', 'none']

# 과제 지정 소수 클래스 3종 (Near-full, Donut, Random)
MINOR_IDX = [CLASS_NAMES.index(c) for c in ['Near-full', 'Donut', 'Random']]


# ─────────────────────────────────────────────────────────
# 재현성
# ─────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────
class WaferDataset(Dataset):
    """웨이퍼맵 데이터셋.
    augment=True일 때 90도 단위 회전 + 좌우/상하 반전을 무작위 적용.
    웨이퍼맵은 원형이라 이들 변환에 클래스 라벨이 보존됨."""

    def __init__(self, images, labels, normalize=False, augment=False):
        x = images.astype(np.float32)
        if normalize:
            x = x / 2.0                      # 픽셀값 0/1/2 → 0/0.5/1.0
        self.images = torch.from_numpy(x)
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, lab = self.images[idx], self.labels[idx]
        if getattr(self, 'augment', False):
            k = torch.randint(0, 4, (1,)).item()
            if k:
                img = torch.rot90(img, k, dims=(1, 2))
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=(2,))     # 좌우
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=(1,))     # 상하
        return img, lab


def load_split(path='../data/cache/split.npz'):
    d = np.load(path)
    return {k: d[k] for k in
            ['X_train', 'y_train', 'X_valid', 'y_valid', 'X_test', 'y_test']}


def make_loaders(data, batch_size=64, normalize=False, seed=42, augment=False):
    """augment는 train에만 적용. valid/test는 항상 원본."""
    g = torch.Generator(); g.manual_seed(seed)
    tr = WaferDataset(data['X_train'], data['y_train'], normalize, augment=augment)
    va = WaferDataset(data['X_valid'], data['y_valid'], normalize)
    te = WaferDataset(data['X_test'],  data['y_test'],  normalize)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g),
            DataLoader(va, batch_size=batch_size, shuffle=False),
            DataLoader(te, batch_size=batch_size, shuffle=False))


def get_class_weights(y_train, device):
    """balanced 가중치: n / (n_classes * count)"""
    counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    w = len(y_train) / (len(CLASS_NAMES) * counts)
    return torch.FloatTensor(w).to(device)


# ─────────────────────────────────────────────────────────
# 평가
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, labels_all = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        total_loss += criterion(out, labels).item()
        _, p = torch.max(out.data, 1)
        correct += (p == labels).sum().item()
        total += labels.size(0)
        preds.extend(p.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    return (total_loss / len(loader), 100 * correct / total,
            np.array(labels_all), np.array(preds))


def recall_per_class(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None,
                        labels=list(range(len(CLASS_NAMES))), zero_division=0)


def minor_recall(y_true, y_pred):
    """소수 3종 Recall 평균과 전체 클래스별 Recall을 함께 반환"""
    r = recall_per_class(y_true, y_pred)
    return r[MINOR_IDX].mean(), r


def full_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
                                target_names=CLASS_NAMES, zero_division=0, digits=4))


# ─────────────────────────────────────────────────────────
# 학습
# ─────────────────────────────────────────────────────────
def train_model(model, train_loader, valid_loader, criterion, optimizer, device,
                epochs=30, patience=None, ckpt_path=None, verbose=True,
                scheduler=None):
    """patience=None이면 Early Stopping 없이 epochs 전부 학습.
    monitor 지표는 Valid-Macro-F1 (클수록 좋음).
    종료 시 best epoch의 가중치를 모델에 복원."""
    hist = {k: [] for k in
            ['tr_loss', 'tr_acc', 'va_loss', 'va_acc', 'va_f1', 'va_minor', 'lr']}
    best_f1, best_epoch, wait = -1.0, -1, 0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            _, p = torch.max(out.data, 1)
            correct += (p == labels).sum().item()
            total += labels.size(0)

        tr_loss, tr_acc = run_loss / len(train_loader), 100 * correct / total
        va_loss, va_acc, yt, yp = evaluate(model, valid_loader, criterion, device)
        va_f1 = f1_score(yt, yp, average='macro', zero_division=0)
        va_mn, _ = minor_recall(yt, yp)
        cur_lr = optimizer.param_groups[0]['lr']

        for k, v in zip(['tr_loss', 'tr_acc', 'va_loss', 'va_acc',
                         'va_f1', 'va_minor', 'lr'],
                        [tr_loss, tr_acc, va_loss, va_acc, va_f1, va_mn, cur_lr]):
            hist[k].append(v)

        # 스케줄러 (Valid-Macro-F1 기준)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_f1)
            else:
                scheduler.step()

        improved = va_f1 > best_f1
        if improved:
            best_f1, best_epoch, wait = va_f1, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            if ckpt_path:
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(best_state, ckpt_path)
        else:
            wait += 1

        if verbose:
            mark = ' *' if improved else ''
            print(f"Ep{ep:3d}  tr_loss {tr_loss:.4f} tr_acc {tr_acc:6.2f} | "
                  f"va_loss {va_loss:.4f} va_acc {va_acc:6.2f} "
                  f"va_F1 {va_f1:.4f} va_minor {va_mn:.4f}{mark}")

        if patience is not None and wait >= patience:
            print(f"\nEarly stopping at epoch {ep} "
                  f"(best epoch {best_epoch}, F1 {best_f1:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist['best_epoch'] = best_epoch
    print(f"[best] epoch {best_epoch}  train_acc {hist['tr_acc'][best_epoch-1]:.2f}  "
          f"valid_acc {hist['va_acc'][best_epoch-1]:.2f}  F1 {best_f1:.4f}  "
          f"gap {hist['tr_acc'][best_epoch-1] - hist['va_acc'][best_epoch-1]:.2f}%p")
    return hist


# ─────────────────────────────────────────────────────────
# 시각화 / 기록
# ─────────────────────────────────────────────────────────
def plot_curves(hist, title='', save_path=None):
    be = hist.get('best_epoch', -1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

    ax[0].plot(hist['tr_loss'], label='Train')
    ax[0].plot(hist['va_loss'], label='Valid')
    ax[0].set_title('Loss')

    ax[1].plot(hist['tr_acc'], label='Train')
    ax[1].plot(hist['va_acc'], label='Valid')
    ax[1].set_title('Accuracy (%)')

    ax[2].plot(hist['va_f1'], label='Macro-F1')
    ax[2].plot(hist['va_minor'], label='Minor-Recall')
    ax[2].set_title('Valid Metrics')

    for a in ax:
        if be > 0:
            a.axvline(be - 1, color='gray', ls='--', lw=1, alpha=.7)
        a.set_xlabel('Epoch'); a.legend(); a.grid(alpha=.3)

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


HEADER = (['exp', 'desc', 'hparams', 'best_epoch', 'train_acc', 'valid_acc',
           'gap', 'valid_macro_f1', 'valid_minor_recall']
          + [f'recall_{c}' for c in CLASS_NAMES])


def log_result(path, exp, desc, hparams, hist, y_true, y_pred):
    """모든 지표를 best-epoch 시점 기준으로 기록.
    valid 지표는 y_true/y_pred(=best 모델의 예측)에서 직접 계산."""
    r = recall_per_class(y_true, y_pred)
    be = hist['best_epoch']
    tr = hist['tr_acc'][be - 1] / 100
    va = accuracy_score(y_true, y_pred)
    row = [exp, desc, hparams, be,
           f"{tr:.4f}", f"{va:.4f}", f"{(tr - va) * 100:.2f}",
           f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
           f"{r[MINOR_IDX].mean():.4f}"] + [f'{v:.4f}' for v in r]

    need_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(HEADER)
        w.writerow(row)
    print(f"logged: exp{exp}")