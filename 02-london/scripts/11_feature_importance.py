"""변수 중요도와 ablation.

세 가지를 함께 본다. 하나만 보면 반대 결론이 나올 수 있다.
  (1) XGBoost 내장 중요도 — 분기에 얼마나 쓰였나
  (2) 순열 중요도        — 값을 섞으면 성능이 얼마나 떨어지나
  (3) ablation          — 변수를 빼고 재학습하면 얼마나 떨어지나

워싱턴은 yr 을 대상으로 ablation 했지만 런던에는 yr 이 없다(03 참고).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

import common as C

plt = C.setup_matplotlib()

info = json.load(open("output/10_final_model_info.json", encoding="utf-8"))
model = joblib.load("output/10_final_model.joblib")
PARAMS = {k: v for k, v in info["params"].items() if k != "random_state"}
print(f"모델: {info['model']}")
print(f"학습: {info['train']['n_days']}일 (이상치 방침 {info['train']['outlier_strategy']})")
print()

tr_raw, te = C.load_train_test()
base = C.filter_coverage(tr_raw, verbose=False)
train, _ = C.clip_outliers(base, 0.99)
X_tr, y_tr = C.make_xy(train)
X_te, y_te = C.make_xy(te)

print("=" * 66)
print("1. XGBoost 내장 중요도 (분기에 쓰인 비중)")
print("=" * 66)
nat = pd.DataFrame({"변수": C.FEATURES, "중요도": model.feature_importances_}) \
        .sort_values("중요도", ascending=False)
print(nat.round(4).to_string(index=False))

print()
print("=" * 66)
print("2. 순열 중요도 (2016 예측 기준, 30회 반복)")
print("=" * 66)
perm = permutation_importance(model, X_te, y_te, n_repeats=30,
                              random_state=C.RANDOM_STATE, scoring="r2")
perm_df = pd.DataFrame({"변수": C.FEATURES, "중요도": perm.importances_mean,
                        "std": perm.importances_std}) \
            .sort_values("중요도", ascending=False)
print(perm_df.round(4).to_string(index=False))
print("\n값 = 그 변수를 무작위로 섞었을 때 R2 가 떨어지는 정도.")

print()
print("=" * 66)
print("3. Ablation — 변수를 빼고 동일 조건으로 재학습")
print("=" * 66)


def fit_eval(features):
    m = XGBRegressor(**PARAMS, random_state=C.RANDOM_STATE)
    m.fit(train[features], y_tr)
    return C.evaluate(y_te, m.predict(te[features]))


full = C.evaluate(y_te, model.predict(X_te))
rows = [{"제외 변수": "(없음 — 전체)", **full, "R2 변화": 0.0}]
for drop in ["t1", "t2", "weather_code", "hum", "wind_speed",
             "workingday", "season", "mnth", "is_christmas"]:
    feats = [f for f in C.FEATURES if f != drop]
    m = fit_eval(feats)
    rows.append({"제외 변수": drop, **m, "R2 변화": m["R2"] - full["R2"]})

weather = ["t1", "t2", "hum", "wind_speed", "weather_code"]
m = fit_eval([f for f in C.FEATURES if f not in weather])
rows.append({"제외 변수": "날씨 5개 전부", **m, "R2 변화": m["R2"] - full["R2"]})

abl = pd.DataFrame(rows)
print(abl.round(4).to_string(index=False))

singles = abl[~abl["제외 변수"].isin(["(없음 — 전체)", "날씨 5개 전부"])]
worst = singles.nsmallest(1, "R2 변화").iloc[0]
wrow = abl[abl["제외 변수"] == "날씨 5개 전부"].iloc[0]
print()
print(f"가장 중요한 단일 변수 : {worst['제외 변수']} (제외 시 R2 {worst['R2 변화']:+.4f})")
print(f"날씨 5개 전부 제외    : R2 {full['R2']:.4f} -> {wrow['R2']:.4f} ({wrow['R2 변화']:+.4f})")
print("-> 이 모델 예측력의 대부분이 날씨에서 나온다.")

print()
print("=" * 66)
print("4. 지표가 엇갈리는 경우 — t1 과 t2")
print("=" * 66)
t1 = abl[abl["제외 변수"] == "t1"].iloc[0]
t1_perm = perm_df[perm_df["변수"] == "t1"]["중요도"].values[0]
print(f"순열 중요도 t1 : {t1_perm:.3f}")
print(f"ablation  t1 : R2 {t1['R2 변화']:+.4f}")
print(f"t1-t2 상관   : {train['t1'].corr(train['t2']):.3f}")
print()
print("순열 중요도는 높은데 빼도 성능이 별로 안 떨어진다. t1 과 t2 가 거의 같은")
print("정보라 하나를 빼면 다른 하나가 대신하기 때문이다. 두 지표를 함께 봐야")
print("'기온 정보는 필수지만 t1/t2 중 하나면 충분하다'는 결론에 도달한다.")

pairs = perm_df.sort_values("중요도")
colors = [C.ACCENT if i == len(pairs) - 1 else C.RECESS for i in range(len(pairs))]
fig, ax = plt.subplots(figsize=(9, 7), facecolor=C.SURFACE)
ax.set_facecolor(C.SURFACE)
bars = ax.barh(pairs["변수"], pairs["중요도"], color=colors, height=0.6,
               xerr=pairs["std"], error_kw=dict(ecolor=C.MUTED, lw=1))
for bar, v in zip(bars, pairs["중요도"]):
    ax.text(bar.get_width() + pairs["중요도"].max() * 0.02,
            bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
            ha="left", va="center", color=C.INK, fontsize=10.5)
ax.set_xlim(min(0, pairs["중요도"].min() * 1.2), pairs["중요도"].max() * 1.22)
ax.grid(True, axis="x", color=C.GRID, linewidth=0.8)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(C.AXIS)
ax.tick_params(colors=C.MUTED, labelsize=11)
ax.tick_params(axis="y", length=0)
top = perm_df.iloc[0]
ax.set_title(f"{top['변수']}이(가) 가장 큰 기여 — 값은 R² 감소폭",
             color=C.MUTED, fontsize=10.5, pad=12, loc="left")
fig.suptitle("순열 중요도 — 2016년 예측 기준 (XGBoost)", color=C.INK, fontsize=15,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("output/11_feature_importance.png", dpi=150, facecolor=C.SURFACE,
            bbox_inches="tight")
plt.close(fig)
print()
print("저장: output/11_feature_importance.png")
