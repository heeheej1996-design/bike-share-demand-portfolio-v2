"""XGBoost 하이퍼파라미터 튜닝 — 2015년만 사용.

08 에서 선형회귀와 XGBoost 가 동점(월블록 CV 0.436 vs 0.359, 표준편차 약 0.4)
으로 나왔고, 최종 목표 지표인 2016 예측에서 XGBoost 가 세 지표 모두 앞서
XGBoost 를 선택했다. 이 절차의 한계는 REPORT 한계 절에 기록한다.

워싱턴은 전체 데이터에 GridSearchCV 를 돌려 테스트 정보가 유입됐고 스스로
한계로 기록했다. 여기서는 하이퍼파라미터 탐색만은 2015 안에서 끝낸다.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from xgboost import XGBRegressor

import common as C

tr, _ = C.load_train_test()
train = C.drop_outliers(C.filter_coverage(tr)).sort_values("dteday").reset_index(drop=True)
X, y = C.make_xy(train)
groups = train["dteday"].dt.month
gkf = GroupKFold(n_splits=6)
print(f"2015년 {len(X)}일로 튜닝 (2016 미사용)")
print(f"검증: 월 블록 GroupKFold(6) — 08 에서 정한 기준")
print()

# 2015년은 353일로 워싱턴(731일)의 절반이라 과적합 방지 쪽으로 범위를 좁혔다.
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5, 10],
    "reg_lambda": [1.0, 10.0],
}

gs = GridSearchCV(XGBRegressor(random_state=C.RANDOM_STATE), param_grid,
                  cv=gkf, scoring="r2", n_jobs=-1)
gs.fit(X, y, groups=groups)

n = len(gs.cv_results_["params"])
print("=" * 70)
print(f"GridSearchCV — {n}개 조합 x 6 fold = {n * 6:,}회 학습")
print("=" * 70)
print(f"최적 조합 : {gs.best_params_}")
print(f"CV R2     : {gs.best_score_:.4f}   (튜닝 전 0.3586)")
print()
top = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score"]] \
        .sort_values("mean_test_score", ascending=False).head(5)
print("상위 5개 조합:")
for _, r in top.iterrows():
    print(f"  R2 {r['mean_test_score']:.4f} ± {r['std_test_score']:.4f}")
    print(f"     {r['params']}")

print()
print("=" * 70)
print("주의: min_child_weight")
print("=" * 70)
mcw = gs.best_params_["min_child_weight"]
print(f"선택된 값 = {mcw}  ('리프 하나에 최소 {mcw}개 샘플이 필요하다')")
print()
print("학습셋에서 is_christmas=1 인 날은 353일 중 1일뿐이다. min_child_weight 가")
print("2 이상이면 이 변수로 가지를 칠 수 없어 모델이 변수를 통째로 무시한다.")
print("13 에서 그 영향을 측정한다.")

Path("output").mkdir(exist_ok=True)
with open("output/09_best_params.json", "w", encoding="utf-8") as f:
    json.dump({
        "selected_model": "XGBoost",
        "selection_note": ("08 월블록 CV 에서 선형회귀와 동점. "
                           "최종 목표 지표인 2016 예측 성능으로 선택 "
                           "(테스트 정보가 선택에 반영된 절차 — REPORT 한계 참조)"),
        "cv": "GroupKFold(6, groups=month)",
        "cv_r2": gs.best_score_,
        "n_candidates": n,
        "best_params": gs.best_params_,
    }, f, ensure_ascii=False, indent=2)
print("\n저장: output/09_best_params.json")
