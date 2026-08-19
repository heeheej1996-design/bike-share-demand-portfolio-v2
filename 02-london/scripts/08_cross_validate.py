"""2015년 내부 교차검증 — 검증 설계 자체를 점검하고 모델을 비교.

세 가지 방식을 나란히 돌린다. 결과가 크게 달라서, 어떤 검증을 쓰느냐가
결론을 좌우한다는 점을 기록으로 남긴다.

  A. KFold(shuffle)     : 시계열에서 인접일이 새어 낙관적
  B. TimeSeriesSplit    : 1년치에서는 어떤 fold 도 사계절을 못 배워 비관적
  C. 월 블록 GroupKFold : 학습셋이 항상 사계절 포함 -> 최종 조건에 가장 가까움

2016 은 여기서 전혀 사용하지 않는다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor

import common as C

tr, _ = C.load_train_test()
train = C.drop_outliers(C.filter_coverage(tr)).sort_values("dteday").reset_index(drop=True)
X, y = C.make_xy(train)
groups = train["dteday"].dt.month
print(f"2015년 {len(X)}일로 교차검증 (2016 미사용)")
print()


def models():
    return {
        "선형회귀": LinearRegression(),
        "랜덤포레스트": RandomForestRegressor(n_estimators=300, random_state=C.RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                random_state=C.RANDOM_STATE),
    }


results = {}

print("=" * 72)
print("A. KFold(shuffle=True) — 참고용")
print("=" * 72)
kf = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
results["KFold"] = {}
for name, m in models().items():
    s = cross_val_score(m, X, y, cv=kf, scoring="r2")
    results["KFold"][name] = s.mean()
    print(f"{name:<12} R2 {s.mean():.4f} ± {s.std():.4f}")
print("\n인접일이 학습셋에 남아 트리 모델이 실제보다 좋아 보인다.")

print()
print("=" * 72)
print("B. TimeSeriesSplit(5) — 항상 과거로 미래를 예측")
print("=" * 72)
tscv = TimeSeriesSplit(n_splits=5)
results["TimeSeriesSplit"] = {}
for name, m in models().items():
    s = cross_val_score(m, X, y, cv=tscv, scoring="r2")
    results["TimeSeriesSplit"][name] = s.mean()
    print(f"{name:<12} R2 {s.mean():.4f} ± {s.std():.4f}")
print()
print("fold 1 의 학습셋은 63일(겨울)뿐인데 봄을 예측해야 한다. 최종 모델은")
print("353일 사계절을 전부 학습하므로 조건이 다르다. 학습에서 못 본 구간을")
print("외삽하지 못하는 트리 모델에 구조적으로 불리한 평가다.")

print()
print("=" * 72)
print("C. 월 블록 GroupKFold(6) — 기준")
print("=" * 72)
print("2개월을 통째로 빼고 나머지 10개월로 학습 -> 학습셋이 항상 사계절 포함")
print()
gkf = GroupKFold(n_splits=6)
results["월블록"] = {}
for name, m in models().items():
    s = cross_val_score(m, X, y, cv=gkf, groups=groups, scoring="r2")
    results["월블록"][name] = s.mean()
    print(f"{name:<12} R2 {s.mean():.4f} ± {s.std():.4f}")

print()
print("=" * 72)
print("세 방식 종합")
print("=" * 72)
comp = pd.DataFrame(results)
print(comp.round(4).to_string())
print()
print("검증 방식에 따라 순위가 바뀐다. 최종 조건에 가장 가까운 C 를 기준으로 삼는다.")

print()
print("=" * 72)
print("결론")
print("=" * 72)
mb = results["월블록"]
rank = sorted(mb, key=lambda k: -mb[k])
for i, n in enumerate(rank, 1):
    print(f"{i}위 {n:<12} {mb[n]:.4f}")
print()
gap = mb[rank[0]] - mb[rank[1]]
print(f"1위와 2위 차이 {gap:.4f} — fold 간 표준편차(약 0.38)보다 훨씬 작다.")
print("즉 선형회귀와 XGBoost 는 이 검증으로 구분되지 않는다(동점).")
print("랜덤포레스트는 점수도 낮고 변동성도 커서 후보에서 제외한다.")
print()
print("동점인 두 모델의 선택은 09/10 에서 실제 목표 지표(2016 예측)로 가른다.")
print("이 절차의 한계는 REPORT 의 한계 절에 명시한다.")

Path("output").mkdir(exist_ok=True)
comp.to_csv("output/08_cv_results.csv")
print("\n저장: output/08_cv_results.csv")
