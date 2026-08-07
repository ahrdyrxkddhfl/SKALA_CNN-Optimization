import torch.nn as nn


class BaselineCNN(nn.Module):
    """원본 90-wafer.ipynb 구조. 파라미터 822,281개 중 97.6%가 fc[1]에 집중."""
    def __init__(self, n_classes=9, dropout=0.0):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2, 2))   # 64→62→31
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2, 2))  # 31→29→14

        fc = [nn.Flatten(), nn.Linear(64 * 14 * 14, 64), nn.ReLU()]
        if dropout > 0:
            fc.append(nn.Dropout(dropout))
        fc.append(nn.Linear(64, n_classes))
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        return self.fc(self.layer2(self.layer1(x)))


class ImprovedCNN(nn.Module):
    """Conv 4블록 + BatchNorm + GAP.
    - padding=1로 크기 유지, pooling으로만 축소 (64→32→16→8→4)
    - Receptive Field 약 46px (Baseline 10px) → 전역 패턴 직접 인식
    - GAP로 FC 대체 → 위치 암기 구조 제거
    """
    def __init__(self, n_classes=9, channels=(32, 64, 128, 128),
                 dropout=0.0, use_bn=True, activation='relu'):
        super().__init__()
        act = {'relu':      lambda: nn.ReLU(),
               'leakyrelu': lambda: nn.LeakyReLU(0.01),
               'elu':       lambda: nn.ELU(),
               'gelu':      lambda: nn.GELU()}[activation]

        blocks, in_ch = [], 1
        for out_ch in channels:
            layer = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
            if use_bn:
                layer.append(nn.BatchNorm2d(out_ch))
            layer += [act(), nn.MaxPool2d(2, 2)]
            blocks.append(nn.Sequential(*layer))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

        head = [nn.Flatten()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(in_ch, n_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"{'layer':<24}{'params':>12}{'share':>9}")
    print('-' * 45)
    for name, m in model.named_modules():
        n = sum(p.numel() for p in m.parameters(recurse=False))
        if n:
            print(f"{name:<24}{n:>12,}{100*n/total:>8.1f}%")
    print('-' * 45)
    print(f"{'TOTAL':<24}{total:>12,}")
    return total