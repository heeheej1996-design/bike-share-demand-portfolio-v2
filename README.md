# 🚲 자전거 대여 수요 예측 포트폴리오

도시별 공공 자전거 대여 데이터를 **전처리 → 탐색적 분석 → 시각화 → 회귀 모델링 → 결과 해석 → 문서화** 순서로 분석한 기록입니다. 분석 대상마다 폴더를 나누고, 각 폴더는 번호가 붙은 스크립트로 전 과정을 재현할 수 있게 구성합니다.

## 분석 목록

| 폴더 | 대상 | 데이터 | 상태 | 핵심 결과 |
|---|---|---|---|---|
| [`01-washington/`](01-washington/) | Capital Bikeshare (워싱턴 D.C.) | UCI Bike Sharing Dataset, 2011–2012 일별 731일 | ✅ 완료 | 튜닝 XGBoost로 일일 대여량 예측, 테스트 **R² 0.904** |
| [`02-london/`](02-london/) | Santander Cycles (런던) | London Bike Sharing, 2015–2016 일별 727일 (시간별 17,414행에서 집계) | ✅ 완료 | 2015년 학습 → 2016년 예측, 테스트 **R² 0.763** (완전 시간 분리) |

각 폴더의 `README.md`에 그 분석의 요약과 재현 방법이, `REPORT.md`에 상세 서술이 있습니다.

## 폴더 구조

```
bike-share-demand-portfolio/
├── README.md              # 이 문서 (저장소 안내)
├── requirements.txt       # 모든 분석 공용 패키지 (버전 고정)
├── .gitignore
├── 01-washington/         # 워싱턴 D.C. 분석 (완료)
│   ├── README.md          # 요약 + 재현 방법
│   ├── REPORT.md          # 상세 분석 리포트
│   ├── data/              # 원본 데이터
│   ├── scripts/           # 01~12, 번호가 곧 실행 순서
│   ├── output/            # 차트 PNG, 학습된 모델
│   └── docs/              # 작업 과정 기록
└── 02-london/             # 런던 분석 (완료)
    ├── README.md          # 요약 + 재현 방법
    ├── REPORT.md          # 상세 분석 리포트
    ├── data/              # 원본 시간별 + 일별 + 연도별 분리
    ├── scripts/           # 01~12, 번호가 곧 실행 순서
    ├── output/            # 차트 PNG, 학습된 모델, 2016 예측 결과
    └── docs/              # 분석 계획 + 변환 계획 + 컬럼 대조
```

> 런던 원본은 **시간 단위**라 워싱턴에 없던 "시간별 → 일별 집계" 단계가 앞에 붙습니다(`01`~`02`).
> 집계 규칙은 [`01-washington/docs/HOURLY_TO_DAILY_AGGREGATION.md`](01-washington/docs/HOURLY_TO_DAILY_AGGREGATION.md)를 그대로 적용했습니다.
>
> **두 분석은 평가 방식이 달라 R² 수치를 직접 비교할 수 없습니다.** 워싱턴은 랜덤 8:2 분할(문서 스스로 "낙관적으로 부풀려진 수치"라고 기록),
> 런던은 2015년 학습 / 2016년 예측의 완전한 시간 분리입니다. 자세한 내용은 [`02-london/README.md`의 한계](02-london/README.md#한계) 참조.

## 시작하기

```bash
# 1. 패키지 설치 (저장소 루트에서 한 번만)
python3 -m pip install -r requirements.txt

# 2. 분석 폴더로 이동해서 실행
cd 01-washington
python3 scripts/01_explore_data.py
```

> ⚠️ **스크립트는 반드시 해당 분석 폴더 안에서 실행하세요.** 모든 스크립트가 `data/...`, `output/...`을 **상대경로**로 읽고 씁니다. 저장소 루트나 `scripts/` 안에서 실행하면 `FileNotFoundError`가 납니다.

## 공통 규칙

분석 폴더를 새로 추가할 때 지키는 약속입니다.

| 항목 | 규칙 |
|---|---|
| 폴더 이름 | `NN-도시명` (실행·작성 순서대로 번호) |
| 폴더 구성 | `data/`, `scripts/`, `output/`, `docs/` + `README.md`, `REPORT.md` |
| 스크립트 이름 | `NN_설명.py` — **번호가 곧 실행 순서** |
| 출력 파일 이름 | `NN_...` — 앞 번호 = 그 파일을 만든 스크립트 번호 |
| 경로 | 항상 분석 폴더 기준 상대경로 (`data/`, `output/`) |
| 재현성 | 난수는 `random_state=42`로 고정 |
| 패키지 | 루트 `requirements.txt` 하나로 공용 관리 |

## 환경

Python 3.9.6 · pandas 2.3.3 · numpy 2.0.2 · scikit-learn 1.6.1 · XGBoost 2.1.4 · matplotlib 3.9.4 · joblib 1.5.3

차트 스크립트가 한글 폰트를 `AppleGothic`으로 지정하므로 **macOS 기준**입니다. 다른 OS에서는 각 스크립트의 `plt.rcParams["font.family"]`를 해당 OS의 한글 폰트로 바꿔야 합니다.
