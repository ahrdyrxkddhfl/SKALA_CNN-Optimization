# 웨이퍼맵 불량 분류 CNN 최적화

WM-811K 반도체 웨이퍼맵 이미지 기반 9종 불량 유형 분류 CNN의 과적합 해소 및 성능 최적화

SKALA · 딥러닝 과제 · 2인 팀 프로젝트 (황도희, 이소정)

**역할 분담.** 두 사람이 각자 최적화를 진행한 뒤 Valid-Macro-F1로 비교해 한쪽 결과를 채택하기로 했고, 황도희의 결과가 채택되었다. 이 저장소의 전처리·모델 설계·실험 #1~#12 설계와 실행·Test 평가는 황도희가, 보고서 작성은 이소정이 담당했다.

---

## 최종 결과

| 지표 | Baseline | 최종 (#12) | 변화 |
| --- | --- | --- | --- |
| Test-Accuracy | 0.8270 | **0.9140** | +0.0870 |
| Test-Macro-F1 | 0.7552 | **0.8805** | +0.1253 |
| Test-Recall (mean) | 0.7461 | **0.8890** | +0.1429 |
| Scratch F1 | 0.2563 | **0.8441** | 3.3배 |
| 파라미터 | 822,281 | **242,121** | -71% |
| Train-Valid 격차* | 14.95%p | **1.29%p** | -91% |

\* Train Accuracy는 학습 중(train 모드) 측정값이라 Dropout·증강이 켜진 최종 모델에서는 격차가 작게 나타난다. eval 모드 재측정은 하지 않았다 ([재검토](#제출-후-재검토에서-확인한-한계) 참고).

**핵심 진단:** 과적합의 원인은 모델 용량 과다가 아니라 파라미터 배치의 구조적 결함이었다. 전체 파라미터의 97.6%가 단일 FC 층에 집중되고 Conv의 Receptive Field가 10px(입력 64px)에 불과하여, Conv가 전역 패턴을 인식하지 못하고 FC의 위치 암기가 이를 대체하고 있었다.

**해결:** Conv 4블록 + BatchNorm + GAP로 RF를 46px로 확대하고 파라미터를 71% 감축한 뒤, LR Scheduler로 학습을 안정화하고 웨이퍼맵의 회전 대칭성을 활용한 데이터 증강으로 암기 가능성을 차단했다.

---

## 보고서

- **[본문 (PDF)](report/report-main.pdf)** — 제출본
- [부록 (PDF)](report/report-appendix.pdf) — 실험 상세 데이터 및 보조 분석
- [상세 보고서 (Markdown)](report/full-report.md) — 진단 과정, 실험별 분석, 학습 곡선, 계산 근거

---

## 환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 기준. PyTorch는 CUDA / Apple MPS / CPU를 자동 감지한다.

노트북은 VS Code로 열거나, 브라우저에서 열려면 `pip install jupyterlab`을 추가로 설치한다. 노트북은 `notebooks/`를 작업 폴더로 두고 실행한다 (상대 경로 사용).

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

> `results.csv`는 이어 쓰기 방식이라 재실행 시 행이 중복된다. 다시 돌릴 때는 기존 파일을 지우거나 옮긴 뒤 실행한다.

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
├── report/
│   ├── report-main.pdf        보고서 본문 (제출본)
│   ├── report-appendix.pdf    부록
│   └── full-report.md         상세 보고서
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
| 1 | Baseline (ES 없이 30 epoch, best epoch 가중치로 평가) | 0.7851 |
| 2 | + Early Stopping (patience=5) | 0.7914 |
| 3 | + Class Weight (balanced) | 0.7639 |
| 4 | + Dropout (p=0.4) | 0.7623 |
| 5 | + Weight Decay (1e-4) | 0.7529 |

규제를 누적할수록 성능이 저하되었고, 실험 #5는 Train Accuracy(76.54%)가 Valid(79.87%)보다 낮게 측정되었다. 규제 추가만으로는 개선되지 않는다는 점이 구조 재설계(트랙 B)로 방향을 바꾼 근거다.

단, #3에서 추가한 Class Weight가 #4·#5에도 유지되므로 Dropout·Weight Decay 효과는 Class Weight가 적용된 상태에서 측정되었다. 또한 #5의 Train Accuracy는 Dropout(p=0.4)이 켜진 학습 모드 값이라 실제보다 낮게 나온다.

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

데이터 분할은 `split.npz`로 고정 저장되어 모든 실험이 동일한 데이터를 사용하고, 각 실험 시작 시 시드를 재설정해 모델 초기화를 고정한다.

다만 배치 셔플 순서는 실험마다 다르다. 트랙 A/B가 하나의 `train_loader`를 공유하여 셔플용 generator 상태가 실험을 거치며 이어지기 때문이다 (설정이 사실상 같은 #1과 #6의 1 epoch Train Loss가 0.9045 / 0.9313으로 다름). 또한 MPS는 결정론적 연산이 보장되지 않는다. 따라서 Macro-F1 차이가 약 0.01 이하인 실험 간 비교는 노이즈 범위로 해석해야 한다.

---

## 제출 후 재검토에서 확인한 한계

보고서 제출 이후 코드와 실행 로그를 다시 검토하며 확인한 사항이다. 보고서(PDF)는 제출본 그대로 두고 여기에 정리한다.

| 항목 | 내용 | 결과 해석에 미치는 영향 |
| --- | --- | --- |
| Train Accuracy 측정 방식 | 에폭 중 학습 모드(Dropout·증강 적용)의 누적 정확도를 기록 | #5의 음수 격차, #12의 격차 1.29%p가 실제보다 작게 측정됨. best 체크포인트로 Train 셋을 eval 모드에서 재측정해야 정확한 격차를 알 수 있음 |
| 셔플 순서 미통제 | 위 [재현성](#재현성) 참고 | 차이 0.01 이하 비교(#9·#10·#11, #1→#2 Early Stopping)는 노이즈 범위. Early Stopping은 best 가중치 복원과 함께 쓰이므로, 같은 경로라면 #1(30 epoch)의 best가 #2보다 낮을 수 없음 |
| 'none' 클래스 구성 | 전처리에서 라벨이 빈 웨이퍼도 `'none'`으로 변환됨 | 원본의 'none' 785,938장 중 다수가 무라벨이라, 정상 샘플 5,000장에 무라벨 웨이퍼가 섞였을 가능성이 큼. 라벨이 있는 정상만 샘플링하도록 수정 필요 |
| Valid 지표 진동 | 트랙 B(#7~#11)에서 epoch 간 Valid Macro-F1이 크게 흔들림 | 최고점 선택 방식이라 #8(LR Scheduler)의 개선폭은 낙관적으로 추정되었을 수 있음. 곡선이 안정적인 #12가 가장 신뢰할 만한 결과 |
| lot 단위 분할 미적용 | `lotName`을 무시한 무작위 층화 분할 | 같은 lot의 웨이퍼가 Train/Test에 함께 들어가 Test 성능이 낙관적일 수 있음. `GroupShuffleSplit(groups=lotName)`으로 검증 필요 |
| 단일 시드 | 모든 실험을 seed=42 한 번씩 실행 | 표준편차 없음. 주요 실험(#8, #12)은 복수 시드 반복 필요 |

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

