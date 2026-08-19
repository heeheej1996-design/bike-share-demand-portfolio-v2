"""yr(연도) 변수를 제외하고 재학습했을 때 성능이 얼마나 나빠지는지 확인 (yr 중요도 검증)."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

df = pd.read_csv("data/day.csv")

drop_cols = ["instant", "dteday", "casual", "registered", "cnt"]
feature_cols_full = [c for c in df.columns if c not in drop_cols]
feature_cols_no_yr = [c for c in feature_cols_full if c != "yr"]

y = df["cnt"]

# 08/09와 동일한 튜닝된 하이퍼파라미터, 동일 8:2 분할
BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def evaluate(feature_cols):
    X = df[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = XGBRegressor(**BEST_PARAMS, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
    }


with_yr = evaluate(feature_cols_full)
without_yr = evaluate(feature_cols_no_yr)

result_df = pd.DataFrame(
    {"yr 포함 (09)": with_yr, "yr 제외": without_yr}
).T
result_df["RMSE 변화(%)"] = (result_df["rmse"] / with_yr["rmse"] - 1) * 100

print("=== yr 포함 vs 제외 성능 비교 (동일 8:2 분할, 동일 하이퍼파라미터) ===")
print(result_df.round(4))
