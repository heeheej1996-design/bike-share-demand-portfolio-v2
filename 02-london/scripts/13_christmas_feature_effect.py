"""is_christmas 피처 추가 전후 비교.

12 에서 확인한 2016년 최대 오차(12/25, +22,038건)를 겨냥한 피처다.
is_holiday 는 '법정공휴일' 플래그라 12/25 가 주말이면 대체휴일로 옮겨간다.
2016-12-25(일)는 holiday=0 이었고 모델은 이 날을 평범한 일요일로 봤다.

⚠ 이 피처는 교차검증으로 검증할 수 없다. 학습 데이터에 크리스마스가
   2015-12-25 단 하루뿐이라, 어떤 fold 로 나눠도 표본이 1개다.
   즉 데이터가 아니라 '지하철 운휴' 라는 도메인 근거로 넣는 피처다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

import common as C

plt = C.setup_matplotlib()

tr, te = C.load_train_test()
train = C.drop_outliers(C.filter_coverage(tr, verbose=False))
y_tr, y_te = train[C.TARGET], te[C.TARGET]

print("=" * 66)
print("1. 왜 이 피처가 필요한가")
print("=" * 66)
both = pd.concat([train, te])
xmas = both[(both.dteday.dt.month == 12) & both.dteday.dt.day.between(23, 27)]
v = xmas[["dteday", C.TARGET, "holiday", "is_weekend", "is_christmas"]].copy()
v["dteday"] = v["dteday"].dt.strftime("%Y-%m-%d (%a)")
print(v.to_string(index=False))
print()
print("두 해 모두 12/25 에 수요가 주변일의 2~4배로 뛴다. 크리스마스 당일에는")
print("런던 지하철이 전면 운휴해 자전거로 수요가 몰리는 것으로 보인다.")
print()
print("holiday 플래그는 2015 년엔 12/25 에 붙었지만(금요일), 2016 년엔 일요일이라")
print("대체휴일 12/26·27 로 옮겨가 정작 12/25 는 0 이 되었다.")
print("is_christmas 는 요일과 무관하게 12/25 를 항상 표시한다.")

models = {
    "선형회귀": lambda: LinearRegression(),
    "랜덤포레스트": lambda: RandomForestRegressor(n_estimators=300, random_state=C.RANDOM_STATE),
    "XGBoost": lambda: XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                    min_child_weight=5, subsample=0.8, colsample_bytree=1.0,
                                    reg_lambda=1.0, random_state=C.RANDOM_STATE),
}

rows, xmas_rows = [], []
for name, mk in models.items():
    for label, feats in [("추가 전", C.FEATURES_BASE), ("추가 후", C.FEATURES)]:
        m = mk().fit(train[feats], y_tr)
        pred = m.predict(te[feats])
        rows.append({"모델": name, "구분": label, **C.evaluate(y_te, pred)})
        i = te.index[te["is_christmas"] == 1][0]
        xmas_rows.append({"모델": name, "구분": label,
                          "12/25 예측": pred[i], "실제": y_te.iloc[i],
                          "오차": y_te.iloc[i] - pred[i]})

res = pd.DataFrame(rows)
print()
print("=" * 66)
print("2. 전체 성능 변화 (2016년 365일)")
print("=" * 66)
pv = res.pivot(index="모델", columns="구분", values=["R2", "RMSE", "MAE"])
pv = pv.reindex(list(models))
for metric in ["R2", "RMSE", "MAE"]:
    pv[(metric, "변화")] = pv[(metric, "추가 후")] - pv[(metric, "추가 전")]
print(pv.round(4).to_string())

print()
print("=" * 66)
print("3. 12월 25일 하루의 변화  <- 이 피처의 실제 목적")
print("=" * 66)
xr = pd.DataFrame(xmas_rows)
print(xr.round(0).to_string(index=False))
print()
for name in models:
    b = xr[(xr["모델"] == name) & (xr["구분"] == "추가 전")]["오차"].values[0]
    a = xr[(xr["모델"] == name) & (xr["구분"] == "추가 후")]["오차"].values[0]
    print(f"{name:<8} 오차 {b:>+9,.0f} -> {a:>+9,.0f}   ({abs(b) - abs(a):>+9,.0f}건 개선)")

print()
print("=" * 66)
print("4. 판단")
print("=" * 66)
print("전체 R2 변화는 미미하다. 365일 중 하루짜리 피처이므로 당연한 결과다.")
print("이 피처의 가치는 평균 성능이 아니라 '최악의 실패를 없애는 것' 에 있다.")
print()
print("한계: 학습 표본이 2015-12-25 하루뿐이다. 그 하루의 특성(요일·날씨)이")
print("      함께 학습돼 왜곡될 수 있다. 여러 해 데이터가 쌓이면 안정된다.")

# ---------- 차트 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=C.SURFACE)

ax = axes[0]
C.style_axes(ax)
names = list(models)
x = np.arange(len(names))
before = [res[(res["모델"] == n) & (res["구분"] == "추가 전")]["MAE"].values[0] for n in names]
after = [res[(res["모델"] == n) & (res["구분"] == "추가 후")]["MAE"].values[0] for n in names]
ax.bar(x - 0.2, before, width=0.38, color=C.RECESS, label="추가 전")
ax.bar(x + 0.2, after, width=0.38, color=C.ACCENT, label="추가 후")
for xi, (b, a) in enumerate(zip(before, after)):
    ax.text(xi - 0.2, b, f"{b:,.0f}", ha="center", va="bottom", fontsize=9, color=C.INK)
    ax.text(xi + 0.2, a, f"{a:,.0f}", ha="center", va="bottom", fontsize=9, color=C.INK)
ax.set_xticks(x); ax.set_xticklabels(names, color=C.MUTED)
ax.set_ylabel("MAE (낮을수록 좋음)", color=C.SEC_INK)
ax.set_title("전체 365일 평균 오차", color=C.INK, fontsize=12, pad=10)
ax.legend(frameon=False, labelcolor=C.SEC_INK)
ax.set_ylim(0, max(before + after) * 1.2)

ax = axes[1]
C.style_axes(ax)
eb = [abs(xr[(xr["모델"] == n) & (xr["구분"] == "추가 전")]["오차"].values[0]) for n in names]
ea = [abs(xr[(xr["모델"] == n) & (xr["구분"] == "추가 후")]["오차"].values[0]) for n in names]
ax.bar(x - 0.2, eb, width=0.38, color=C.WARN, label="추가 전")
ax.bar(x + 0.2, ea, width=0.38, color=C.ACCENT, label="추가 후")
for xi, (b, a) in enumerate(zip(eb, ea)):
    ax.text(xi - 0.2, b, f"{b:,.0f}", ha="center", va="bottom", fontsize=9, color=C.INK)
    ax.text(xi + 0.2, a, f"{a:,.0f}", ha="center", va="bottom", fontsize=9, color=C.INK)
ax.set_xticks(x); ax.set_xticklabels(names, color=C.MUTED)
ax.set_ylabel("절대 오차 (건)", color=C.SEC_INK)
ax.set_title("2016년 12월 25일 하루", color=C.INK, fontsize=12, pad=10)
ax.legend(frameon=False, labelcolor=C.SEC_INK)
ax.set_ylim(0, max(eb + ea) * 1.2)

fig.suptitle("is_christmas 피처 추가 효과", color=C.INK, fontsize=15,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("output/13_christmas_feature_effect.png", dpi=150, facecolor=C.SURFACE,
            bbox_inches="tight")
plt.close(fig)
print()
print("저장: output/13_christmas_feature_effect.png")
