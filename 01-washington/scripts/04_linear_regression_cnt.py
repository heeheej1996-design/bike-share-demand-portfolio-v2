"""선형회귀로 cnt 예측 (casual/registered는 타겟 누수라 제외)."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("data/day.csv")

# instant(번호), dteday(날짜) 제외
# casual, registered는 합치면 cnt와 같아 타겟 누수(leakage)이므로 제외
drop_cols = ["instant", "dteday", "casual", "registered", "cnt"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df["cnt"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("=== 사용한 입력 변수 ===")
print(feature_cols)
print()
print(f"학습 데이터: {len(X_train)}건 / 테스트 데이터: {len(X_test)}건")
print()
print("=== 테스트셋 성능 지표 ===")
print(f"R2   : {r2:.4f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print()

print("=== 회귀 계수 (feature -> coefficient) ===")
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": model.coef_,
}).sort_values("coefficient", key=abs, ascending=False)
print(coef_df.to_string(index=False))
print()
print(f"절편(intercept): {model.intercept_:.2f}")
