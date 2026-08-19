"""LinearRegression vs RandomForest vs XGBoost 성능 및 변수 중요도 비교 (04와 동일 분할)."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("data/day.csv")

# instant(번호), dteday(날짜) 제외
# casual, registered는 합치면 cnt와 같아 타겟 누수(leakage)이므로 제외
drop_cols = ["instant", "dteday", "casual", "registered", "cnt"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df["cnt"]

# 04_linear_regression_cnt.py와 동일한 분할로 비교 가능하게 맞춤
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
    ),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results.append({"model": name, "R2": r2, "RMSE": rmse, "MAE": mae})

result_df = pd.DataFrame(results).set_index("model")
print("=== 모델별 성능 비교 (테스트셋) ===")
print(result_df.round(4))
print()

# 트리 모델 변수 중요도
rf = models["RandomForest"]
xgb = models["XGBoost"]

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "RandomForest_importance": rf.feature_importances_,
    "XGBoost_importance": xgb.feature_importances_,
}).sort_values("RandomForest_importance", ascending=False)

print("=== 변수 중요도 (RandomForest / XGBoost) ===")
print(importance_df.to_string(index=False))
