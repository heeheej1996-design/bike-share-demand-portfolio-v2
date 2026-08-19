"""베이스라인: 나이브 기준선 3종 + 선형회귀.

워싱턴에 없던 단계다. 기준선 없이 R2 만 보면 모델이 잘한 건지 알 수 없다.
"요일x월 평균"을 못 이기면 날씨 변수가 값을 못 하고 있다는 뜻이다.

학습: 2015 (coverage>=0.90) / 평가: 2016 전체
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import common as C

tr, te = C.load_train_test()
print("학습셋 필터링:", end=" ")
tr = C.filter_coverage(tr)
print(f"평가셋: 2016년 {len(te)}일 전체 (실제 예측 상황이므로 필터링 없음)")
print()

y_te = te[C.TARGET]
results = {}

# ---------- 기준선 1: 전체 평균 ----------
results["① 전체 평균"] = C.evaluate(y_te, np.full(len(te), tr[C.TARGET].mean()))

# ---------- 기준선 2: 월별 평균 ----------
m_mean = tr.groupby("mnth")[C.TARGET].mean()
results["② 월별 평균"] = C.evaluate(y_te, te["mnth"].map(m_mean).values)

# ---------- 기준선 3: 요일 x 월 평균 ----------
wm_mean = tr.groupby(["mnth", "weekday"])[C.TARGET].mean()
pred_wm = pd.MultiIndex.from_arrays([te["mnth"], te["weekday"]]).map(wm_mean)
pred_wm = pd.Series(pred_wm, index=te.index).fillna(tr[C.TARGET].mean())
results["③ 요일x월 평균"] = C.evaluate(y_te, pred_wm.values)

# ---------- 선형회귀 ----------
X_tr, y_tr = C.make_xy(tr)
X_te, _ = C.make_xy(te)
lr = LinearRegression().fit(X_tr, y_tr)
results["④ 선형회귀"] = C.evaluate(y_te, lr.predict(X_te))

print("=" * 70)
print("2016년 예측 성능")
print("=" * 70)
for name, m in results.items():
    C.print_metrics(name, m, width=18)

print()
print("=" * 70)
print("해석")
print("=" * 70)
best_naive = max(["① 전체 평균", "② 월별 평균", "③ 요일x월 평균"],
                 key=lambda k: results[k]["R2"])
lr_r2 = results["④ 선형회귀"]["R2"]
nb_r2 = results[best_naive]["R2"]
print(f"가장 좋은 나이브 기준선 : {best_naive}  R2 {nb_r2:.4f}")
print(f"선형회귀                : R2 {lr_r2:.4f}")
print(f"차이                    : {lr_r2 - nb_r2:+.4f}")
print()
if lr_r2 > nb_r2:
    print("-> 날씨 변수가 달력 정보만으로는 못 잡는 신호를 추가로 설명하고 있다.")
else:
    print("-> 선형회귀가 달력 평균을 이기지 못한다. 날씨 변수 활용을 재검토해야 한다.")
print()
print("이후 트리 모델(07~10)은 최소한 이 기준선들을 넘어야 의미가 있다.")

print()
print("=" * 70)
print("선형회귀 계수")
print("=" * 70)
coef = pd.DataFrame({"feature": C.FEATURES, "coefficient": lr.coef_}) \
         .sort_values("coefficient", key=abs, ascending=False)
print(coef.to_string(index=False))
print(f"\n절편: {lr.intercept_:,.2f}")
print()
print("주의: is_weekend 와 workingday 는 서로 강하게 연관돼 있어")
print("      개별 계수를 단독으로 해석하면 안 된다.")
