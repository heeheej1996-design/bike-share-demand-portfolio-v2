"""저장된 최종 모델(09)의 변수 중요도 순위를 확인하고 가로 막대 차트로 시각화."""

import json

import joblib
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
ACCENT = "#1baf7a"   # emphasis: 최상위 변수(yr)
RECESS = "#c9c7bd"   # 나머지 변수(뚜렷한 순위 차이를 강조하기 위해 회색 처리)

model = joblib.load("output/09_final_xgboost_model.joblib")
info = json.load(open("output/09_final_model_info.json", encoding="utf-8"))
feature_cols = info["feature_cols"]

pairs = sorted(zip(feature_cols, model.feature_importances_), key=lambda p: p[1])
names = [p[0] for p in pairs]
values = [p[1] for p in pairs]
colors = [ACCENT if i == len(pairs) - 1 else RECESS for i in range(len(pairs))]

fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

bars = ax.barh(names, values, color=colors, height=0.6)
for bar, value in zip(bars, values):
    ax.text(
        bar.get_width() + max(values) * 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        ha="left", va="center", color=INK, fontsize=10.5,
    )

ax.set_xlim(0, max(values) * 1.15)
ax.grid(True, axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=11)
ax.tick_params(axis="y", length=0)

ax.set_title(
    "yr(연도)가 전체 중요도의 약 47%로 압도적 1위",
    color=MUTED, fontsize=10.5, pad=12, loc="left",
)
fig.suptitle("튜닝된 XGBoost 변수 중요도", color=INK, fontsize=15, fontweight="bold", y=1.04)

plt.tight_layout()
plt.savefig("output/11_feature_importance.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

print("저장 완료: output/11_feature_importance.png")
for name, value in reversed(pairs):
    print(f"{name:15s} {value:.4f}")
