"""전처리 확인: train(2015)/test(2016) 구조·결측·분포 대조.

일별 변환 자체는 01에서 검증했으므로, 여기서는 학습에 쓸 두 파일이
서로 비교 가능한 상태인지, 그리고 어떤 컬럼을 피처에서 빼야 하는지를 확인한다.
"""

import pandas as pd

TRAIN = "data/london_daily_2015.csv"
TEST = "data/london_daily_2016.csv"

tr = pd.read_csv(TRAIN, parse_dates=["dteday"])
te = pd.read_csv(TEST, parse_dates=["dteday"])

print("=" * 60)
print("1. 규모")
print("=" * 60)
print(f"train(2015) : {tr.shape[0]}행 x {tr.shape[1]}열  "
      f"({tr.dteday.min():%Y-%m-%d} ~ {tr.dteday.max():%Y-%m-%d})")
print(f"test (2016) : {te.shape[0]}행 x {te.shape[1]}열  "
      f"({te.dteday.min():%Y-%m-%d} ~ {te.dteday.max():%Y-%m-%d})")
print()
print("2015년이 362일인 것은 원본 데이터가 1월 4일부터 시작하기 때문 (누락 아님).")

print()
print("=" * 60)
print("2. 컬럼·타입 일치")
print("=" * 60)
print(f"컬럼 동일        : {list(tr.columns) == list(te.columns)}")
print(f"타입 동일        : {tr.dtypes.equals(te.dtypes)}")

print()
print("=" * 60)
print("3. 결측치")
print("=" * 60)
print(f"train NaN : {int(tr.isna().sum().sum())}개")
print(f"test  NaN : {int(te.isna().sum().sum())}개")

print()
print("=" * 60)
print("4. ⚠ yr — 피처에서 제외해야 하는 이유")
print("=" * 60)
print(f"train(2015) yr 고유값 : {sorted(tr.yr.unique())}   <- 상수, 배울 정보 없음")
print(f"test (2016) yr 고유값 : {sorted(te.yr.unique())}   <- 학습에서 본 적 없는 값")
print()
print("연도로 파일을 나눴으므로 당연한 결과지만, yr 을 피처로 넣으면")
print("학습 시에는 무의미하고 예측 시에는 미지값이 되어 모델을 왜곡한다.")
print("워싱턴에서 중요도 1위(47%)였던 변수이므로 코드를 그대로 옮기면 안 된다.")
print()
print(f"참고: 일평균 cnt  2015 {tr.cnt.mean():,.0f} -> 2016 {te.cnt.mean():,.0f} "
      f"({(te.cnt.mean()/tr.cnt.mean()-1)*100:+.1f}%)")
print("     워싱턴은 +64% 였으나 런던은 성장이 미미해 yr 손실이 크지 않다.")

print()
print("=" * 60)
print("5. 분포 대조 (train vs test)")
print("=" * 60)
num_cols = ["cnt", "t1", "t2", "hum", "wind_speed", "weather_code"]
rows = []
for c in num_cols:
    rows.append({
        "컬럼": c,
        "train 평균": tr[c].mean(), "test 평균": te[c].mean(),
        "차이(%)": (te[c].mean() / tr[c].mean() - 1) * 100,
        "train std": tr[c].std(), "test std": te[c].std(),
    })
print(pd.DataFrame(rows).round(2).to_string(index=False))
print()
print("분포 이동이 크지 않아 별도 보정 없이 학습 가능.")

print()
print("=" * 60)
print("6. 범주형 값 집합 차이")
print("=" * 60)
for c in ["season", "weather_code", "weekday", "mnth"]:
    a, b = set(tr[c].unique()), set(te[c].unique())
    only_tr, only_te = sorted(a - b), sorted(b - a)
    mark = "" if not (only_tr or only_te) else "  <- 차이 있음"
    print(f"{c:<13} train만: {only_tr}  test만: {only_te}{mark}")
print()
print("weather_code 는 원본에 없는 값이 섞여 있고(01 단계 경고 참고) train/test 간")
print("값 집합도 다르다. 트리 모델은 수치 분기라 동작하지만, 원-핫 인코딩을 쓸 경우")
print("handle_unknown 설정이 필요하다.")

print()
print("=" * 60)
print("7. 기록 품질 (coverage)")
print("=" * 60)
for name, df in [("train(2015)", tr), ("test(2016)", te)]:
    full = int((df.n_hours == 24).sum())
    low = int((df.coverage < 0.90).sum())
    print(f"{name} : 24시간 온전 {full}일 / coverage<0.90 {low}일 / 전체 {len(df)}일")
print()
print("coverage<0.90 인 train 날짜:")
print(tr[tr.coverage < 0.90][["dteday", "n_hours", "coverage", "cnt"]]
      .to_string(index=False))
print()
print(f"-> 학습에서 제외하면 {int((tr.coverage >= 0.90).sum())}일 사용 (계획 6부 기본값)")

print()
print("=" * 60)
print("8. 피처 구성")
print("=" * 60)
DROP = ["instant", "dteday", "cnt", "yr", "n_hours", "coverage"]
FEATURES = [c for c in tr.columns if c not in DROP]
print(f"제외 : {DROP}")
print(f"  - instant/dteday : 식별자·날짜")
print(f"  - cnt            : 타겟")
print(f"  - yr             : 위 4번 참고")
print(f"  - n_hours/coverage : 예측 시점에 알 수 없는 사후 정보 (필터링에만 사용)")
print()
print(f"피처 {len(FEATURES)}개 : {FEATURES}")
print("타겟 : cnt")
print()
print("워싱턴과 달리 casual/registered 가 없어 타겟 누수 위험은 없다.")
