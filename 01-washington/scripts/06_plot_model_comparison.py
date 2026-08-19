"""모델별 성능 비교 차트: R2/RMSE/MAE, 기준(LinearRegression) 대비 델타, 지표별 최고 모델 표시."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/day.csv")

drop_cols = ["instant", "dteday", "casual", "registered", "cnt"]
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df["cnt"]

# 04, 05와 동일한 분할로 비교 가능하게 맞춤
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

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rows.append({
        "model": name,
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
    })
result_df = pd.DataFrame(rows).set_index("model")

# 모델별 고정 카테고리 색상 (전 차트에서 동일하게 재사용 = identity encoding)
MODEL_COLORS = {
    "LinearRegression": "#2a78d6",  # blue
    "RandomForest": "#eb6834",      # orange
    "XGBoost": "#1baf7a",           # aqua
}
INK = "#0b0b0b"
SEC_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

model_order = list(models.keys())
colors = [MODEL_COLORS[m] for m in model_order]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED)


GOOD = "#1b8f5a"  # delta text: improvement vs LinearRegression baseline


def bar_with_labels(ax, values, fmt, higher_is_better, delta_fmt):
    bars = ax.bar(model_order, values, color=colors, width=0.55)
    ymax = max(values)
    baseline = values[0]  # LinearRegression
    best_idx = int(np.argmax(values)) if higher_is_better else int(np.argmin(values))

    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.025,
            fmt.format(value),
            ha="center", va="bottom", color=INK, fontsize=13, fontweight="bold",
        )
        if i > 0:
            delta = value - baseline
            sign = "+" if delta > 0 else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.095,
                f"기준 대비 {sign}{delta_fmt.format(delta)}",
                ha="center", va="bottom", color=GOOD, fontsize=9.5,
            )
        if i == best_idx:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.16,
                "★ 최고",
                ha="center", va="bottom", color=MODEL_COLORS[model_order[i]],
                fontsize=10.5, fontweight="bold",
            )

    ax.set_ylim(0, ymax * 1.32)
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, fontsize=11)


fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), facecolor=SURFACE)

ax = axes[0]
style_axes(ax)
bar_with_labels(ax, result_df["R2"].values, "{:.3f}", higher_is_better=True, delta_fmt="{:.3f}")
ax.set_title("R² (높을수록 좋음)", color=INK, fontsize=15, pad=12, fontweight="bold")

ax = axes[1]
style_axes(ax)
bar_with_labels(ax, result_df["RMSE"].values, "{:.0f}", higher_is_better=False, delta_fmt="{:.0f}")
ax.set_title("RMSE (낮을수록 좋음)", color=INK, fontsize=15, pad=12, fontweight="bold")

ax = axes[2]
style_axes(ax)
bar_with_labels(ax, result_df["MAE"].values, "{:.0f}", higher_is_better=False, delta_fmt="{:.0f}")
ax.set_title("MAE (낮을수록 좋음)", color=INK, fontsize=15, pad=12, fontweight="bold")

fig.suptitle("모델별 성능 비교 (테스트셋)", color=INK, fontsize=18, fontweight="bold", y=1.03)

plt.tight_layout()
plt.savefig("output/06_model_comparison.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

print("저장 완료: output/06_model_comparison.png")
print(result_df.round(4))
