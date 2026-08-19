"""5-fold 교차검증으로 04/05의 단일 분할 성능 순위가 우연이 아닌지 재확인."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv("data/day.csv")

drop_cols = ["instant", "dteday", "casual", "registered", "cnt"]
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df["cnt"]

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
    ),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("=== 5-Fold 교차검증 R2 (평균 ± 표준편차) ===")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    print(f"{name:17s}: {scores.mean():.4f} ± {scores.std():.4f}  (folds: {np.round(scores, 3)})")
