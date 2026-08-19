"""GridSearchCV로 XGBoost 하이퍼파라미터 튜닝 (5개 파라미터 x 144개 조합, 5-fold CV)."""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("data/day.csv")

drop_cols = ["instant", "dteday", "casual", "registered", "cnt"]
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df["cnt"]

# 07_cross_validate_models.py와 동일한 CV 구성
kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

base_model = XGBRegressor(random_state=42)

grid = GridSearchCV(
    base_model, param_grid, cv=kf, scoring="r2", n_jobs=-1, verbose=0
)
grid.fit(X, y)

print("=== 탐색한 조합 수 ===")
print(len(grid.cv_results_["params"]))
print()

print("=== 최적 하이퍼파라미터 ===")
print(grid.best_params_)
print()

print(f"튜닝 전 XGBoost 5-fold R2 평균 (05/07 기준): 0.8818")
print(f"튜닝 후 최적 조합 5-fold R2 평균           : {grid.best_score_:.4f}")
print()

# 04/05/06/07과 동일한 8:2 분할로 최종 성능 재확인
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_model = XGBRegressor(**grid.best_params_, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("=== 튜닝된 XGBoost 테스트셋 성능 (동일 8:2 분할) ===")
print(f"R2   : {r2:.4f}  (기존 XGBoost: 0.8928)")
print(f"RMSE : {rmse:.2f}  (기존 XGBoost: 655.49)")
print(f"MAE  : {mae:.2f}  (기존 XGBoost: 446.11)")

# 상위 5개 조합도 참고용으로 출력
results_df = pd.DataFrame(grid.cv_results_)[["params", "mean_test_score", "std_test_score"]]
results_df = results_df.sort_values("mean_test_score", ascending=False).head(5)
print()
print("=== 상위 5개 조합 ===")
for _, row in results_df.iterrows():
    print(f"R2 {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}  {row['params']}")
