"""08에서 찾은 최적 하이퍼파라미터로 재학습한 최종 XGBoost 모델을 joblib으로 저장."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

df = pd.read_csv("data/day.csv")

drop_cols = ["instant", "dteday", "casual", "registered", "cnt"]
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df["cnt"]

# 04~08과 동일한 8:2 분할 (테스트셋 성능을 그대로 재현하기 위함)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 08_tune_xgboost.py의 GridSearchCV 최적 조합
BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

model = XGBRegressor(**BEST_PARAMS, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
metrics = {
    "r2": r2_score(y_test, y_pred),
    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    "mae": mean_absolute_error(y_test, y_pred),
}

joblib.dump(model, "output/09_final_xgboost_model.joblib")

info = {
    "model": "XGBRegressor",
    "params": {**BEST_PARAMS, "random_state": 42},
    "feature_cols": feature_cols,
    "target": "cnt",
    "split": {"test_size": 0.2, "random_state": 42},
    "test_metrics": metrics,
}
with open("output/09_final_model_info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print("저장 완료: output/09_final_xgboost_model.joblib")
print("저장 완료: output/09_final_model_info.json")
print()
print("=== 테스트셋 성능 ===")
print(f"R2   : {metrics['r2']:.4f}")
print(f"RMSE : {metrics['rmse']:.2f}")
print(f"MAE  : {metrics['mae']:.2f}")
