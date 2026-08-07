# 웨이퍼맵 불량 분류 CNN 최적화

WM-811K 반도체 웨이퍼맵 이미지 기반 9종 불량 유형 분류 CNN의 과적합 해소 및 성능 최적화

SKALA · 딥러닝 과제

---

## 최종 결과

| 지표 | Baseline | 최종 (#12) | 변화 |
| --- | --- | --- | --- |
| Test-Accuracy | 0.8270 | **0.9140** | +0.0870 |
| Test-Macro-F1 | 0.7552 | **0.8805** | +0.1253 |
| Test-Recall (mean) | 0.7461 | **0.8890** | +0.1429 |
| Scratch F1 | 0.2563 | **0.8441** | 3.3배 |
| 파라미터 | 822,281 | **242,121** | -71% |
| Train-Valid 격차 | 14.95%p | **1.29%p** | -91% |

**핵심 진단:** 과적합의 원인은 모델 용량 과다가 아니라 파라미터 배치의 구조적 결함이었다. 전체 파라미터의 97.6%가 단일 FC 층에 집중되고 Conv의 Receptive Field가 10px(입력 64px)에 불과하여, Conv가 전역 패턴을 인식하지 못하고 FC의 위치 암기가 이를 대체하고 있었다.

**해결:** Conv 4블록 + BatchNorm + GAP로 RF를 46px로 확대하고 파라미터를 71% 감축한 뒤, LR Scheduler로 학습을 안정화하고 웨이퍼맵의 회전 대칭성을 활용한 데이터 증강으로 암기 가능성을 차단했다.

---

## 환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 기준. PyTorch는 CUDA / Apple MPS / CPU를 자동 감지한다.

---

## 데이터 준비

1. Kaggle **WM-811K wafer map** 데이터셋에서 `LSWMD.pkl` 다운로드 (약 2GB)
2. `data/` 폴더에 배치

```
data/
└── LSWMD.pkl
```

> `LSWMD.pkl`은 용량 문제로 저장소에 포함되어 있지 않다 (`.gitignore` 처리).

---

## 실행 순서

### 1. 전처리 (1회만)

```
notebooks/01-prep.ipynb
```

원본 811,457장 → 30,519장 축소(불량 전량 25,519 + 정상 5,000 샘플링), 64×64 리사이즈, stratified 3분할 후 `data/cache/split.npz` 생성.

### 2. 실험 실행

```
notebooks/02-exp.ipynb
```

실험 #1~#12를 순차 실행하고 최종 Test 평가까지 수행. 결과는 `results/results.csv`와 `results/figures/`에 자동 저장된다.

전체 소요 시간: 약 1.5~2시간 (Apple M 시리즈 MPS 기준)

---

## 저장소 구조

```
├── notebooks/
│   ├── 00-baseline.ipynb      제공된 원본 (미수정, 참조용)
│   ├── 01-prep.ipynb          전처리 및 Train/Valid/Test 3분할
│   └── 02-exp.ipynb           실험 #1~#12 + Test 평가
├── src/
│   ├── models.py              BaselineCNN, ImprovedCNN, count_params
│   └── utils.py               학습 루프, 평가 지표, Early Stopping, 결과 기록
├── results/
│   ├── results.csv            실험별 성능 (최종)
│   ├── results-pilot.csv      사전 탐색 기록 (실험 설계 확정 이전)
│   ├── class_distribution.csv 분할별 클래스 분포
│   └── figures/               실험별 학습 곡선 (exp1~exp12)
├── data/                      (gitignore) LSWMD.pkl, split.npz
├── ckpt/                      (gitignore) 실험별 best 가중치
└── requirements.txt
```

---

## 실험 설계

모든 실험은 **직전 실험 대비 한 가지 요소만** 변경하는 누적 방식이다. 인접한 두 실험의 성능 차이가 해당 요소의 기여도에 해당한다.

### 트랙 A — 규제 중심 접근 (Baseline 구조 유지)

| # | 변경 요소 | Valid-Macro-F1 |
| --- | --- | --- |
| 1 | Baseline (ES 없이 30 epoch) | 0.7851 |
| 2 | + Early Stopping (patience=5) | 0.7914 |
| 3 | + Class Weight (balanced) | 0.7639 |
| 4 | + Dropout (p=0.4) | 0.7623 |
| 5 | + Weight Decay (1e-4) | 0.7529 |

규제를 누적할수록 성능이 저하되었고, 실험 #5는 Train Accuracy(76.54%)가 Valid(79.87%)보다 낮은 과소적합에 도달했다. 원인이 모델 용량이 아님을 반증한다.

### 트랙 B — 아키텍처 재설계 접근

| # | 변경 요소 | 평가 범주 | Valid-Macro-F1 |
| --- | --- | --- | --- |
| 6 | Baseline + ES (patience=10) | 기준점 | 0.7835 |
| 7 | + ImprovedCNN (Conv4+BN+GAP) | Layer | 0.7886 |
| 8 | + LR Scheduler | Optimizer | 0.8606 |
| 9 | + Dropout (p=0.3) | 규제 | 0.8526 |
| 10 | + Weight Decay (1e-5) | 규제 | 0.8550 |
| 11 | + LeakyReLU | Activation | 0.8527 |
| 12 | **+ Augmentation (rot90/flip)** | 데이터 | **0.9003** |

---

## 평가 지표

**Macro-F1을 주 지표로 사용한다.** 클래스 불균형이 최대 65:1(Edge-Ring 9,680 vs Near-full 149)이므로 Accuracy는 다수 클래스에 지배되는 허수 지표가 될 수 있다. 실제로 Baseline은 Accuracy 0.8270이었으나 Scratch Recall이 0.214에 불과했다.

**Test는 최종 조합 확정 후 1회만 평가했다.** Early Stopping 시점, 하이퍼파라미터 값, 최종 조합 선정을 모두 Valid만을 근거로 수행하여 test leakage를 차단했다.

**Valid-Recall(소수)** = Near-full, Donut, Random 3종의 Recall 평균 (과제 지정)

---

## 재현성

```python
set_seed(42)   # random / numpy / torch / cuda 전체 고정
```

데이터 분할은 `split.npz`로 고정 저장되어 모든 실험이 동일한 데이터를 사용한다. 모델 초기화와 배치 셔플 순서까지 통제되어, 실험 간 성능 차이는 변경된 요소에서만 기인한다.

---

## 알려진 이슈

**`LSWMD.pkl` 로딩 실패**

`LSWMD.pkl`은 Python 2 시절 pandas 0.x로 저장된 pickle이라 최신 환경에서 그대로 열리지 않는다. `01-prep.ipynb`의 다음 두 처리가 필요하다.

- **모듈 경로 shim** — `pandas.indexes.*`가 `pandas.core.indexes.*`로 이동했으므로 `sys.modules`에 가짜 모듈을 등록해 매핑
- **인코딩 지정** — `pickle.load(f, encoding='latin1')`. `pd.read_pickle`은 encoding 인자를 노출하지 않아 `pickle.load`를 직접 호출

발생하는 오류와 원인:

| 오류 | 원인 |
| --- | --- |
| `EOFError: Ran out of input` | 파일이 0바이트 — 다운로드 실패 |
| `ModuleNotFoundError: No module named 'pandas.indexes'` | shim 미적용 |
| `UnicodeDecodeError: 'ascii' codec can't decode` | `encoding='latin1'` 미지정 |

**메모리 사용량**

`LSWMD.pkl` 로딩 시 4~6GB를 사용한다. 전처리 완료 후 `del df; gc.collect()`로 해제하면 이후 작업은 `split.npz`(7.8MB)만 사용한다.

**`autoreload` 사용 시 주의**

`src/utils.py`의 클래스 정의(특히 `__init__`)를 수정한 경우 `autoreload`가 기존 인스턴스의 속성을 갱신하지 못한다. 커널 재시작을 권장한다.

---

## 데이터셋

**WM-811K** — 실제 반도체 팹에서 수집된 811,457장의 웨이퍼맵. 픽셀값은 0(웨이퍼 외부), 1(정상 다이), 2(불량 다이)의 이산값이다.

클래스: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none

| 분할 | 장수 |
| --- | --- |
| Train | 18,311 |
| Valid | 6,104 |
| Test | 6,104 |
| 합계 | 30,519 |

Baseline 코드의 방침에 따라 정상(none)은 5,000장만 샘플링하고 불량은 전량 사용했다. 그 결과 최다 클래스는 정상이 아닌 Edge-Ring(9,680장)이 되며, 불량 유형 간 불균형이 전면에 드러난다.