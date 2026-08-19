"""XGBoost 튜닝 전(05/07)과 후(08/09)의 성능·하이퍼파라미터를 표 이미지로 비교."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BAND = "#f2f1ec"
HEADER_BG = "#3a3936"
GOOD = "#1b8f5a"
NEUTRAL = "#898781"

# 05/07(튜닝 전) vs 08/09(튜닝 후) 결과
metric_rows = [
    ("5-fold CV R²", "0.882", "0.897", "+0.015", GOOD),
    ("테스트셋 R²", "0.8928", "0.9040", "+0.0112", GOOD),
    ("테스트셋 RMSE", "655.49", "620.45", "-35.04 (-5.3%)", GOOD),
    ("테스트셋 MAE", "446.11", "410.13", "-35.98 (-8.1%)", GOOD),
]

param_rows = [
    ("n_estimators", "300", "300", "동일", NEUTRAL),
    ("max_depth", "4", "4", "동일", NEUTRAL),
    ("learning_rate", "0.05", "0.05", "동일", NEUTRAL),
    ("subsample", "1.0", "0.8", "변경", GOOD),
    ("colsample_bytree", "1.0", "0.8", "변경", GOOD),
]


def draw_table(ax, title, rows, col_labels):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_rows = len(rows) + 1  # + header
    row_h = 1 / (n_rows + 0.6)
    top = 1 - row_h * 0.6

    ax.text(0, top + row_h * 0.55, title, ha="left", va="bottom",
             color=INK, fontsize=15, fontweight="bold")

    col_x = [0.0, 0.40, 0.60, 0.80]
    col_align = ["left", "center", "center", "center"]
    col_w = 0.20  # gap between successive col_x entries (last column ends at 1.0)

    header_y0 = top - row_h
    ax.add_patch(FancyBboxPatch(
        (0, header_y0), 1, row_h, boxstyle="round,pad=0,rounding_size=0.012",
        linewidth=0, facecolor=HEADER_BG, transform=ax.transData,
    ))
    for x, label, align in zip(col_x, col_labels, col_align):
        tx = x if align == "left" else x + col_w / 2
        ax.text(tx, header_y0 + row_h / 2, label, ha=align, va="center",
                 color="#ffffff", fontsize=11.5, fontweight="bold")

    for i, (name, before, after, delta, delta_color) in enumerate(rows):
        y0 = header_y0 - row_h * (i + 1)
        if i % 2 == 1:
            ax.add_patch(plt.Rectangle((0, y0), 1, row_h, linewidth=0, facecolor=BAND))
        ax.text(col_x[0], y0 + row_h / 2, name, ha="left", va="center",
                 color=INK, fontsize=11.5)
        ax.text(col_x[1] + col_w / 2, y0 + row_h / 2, before, ha="center", va="center",
                 color=MUTED, fontsize=11.5)
        ax.text(col_x[2] + col_w / 2, y0 + row_h / 2, after, ha="center", va="center",
                 color=INK, fontsize=11.5, fontweight="bold")
        ax.text(col_x[3] + col_w / 2, y0 + row_h / 2, delta, ha="center", va="center",
                 color=delta_color, fontsize=11, fontweight="bold")

    bottom = header_y0 - row_h * len(rows)
    ax.plot([0, 1], [bottom, bottom], color=GRID, linewidth=1)


fig, axes = plt.subplots(2, 1, figsize=(9, 7.4), facecolor=SURFACE,
                          gridspec_kw={"height_ratios": [4.6, 5.6]})

col_labels_metric = ["지표", "튜닝 전", "튜닝 후", "변화"]
col_labels_param = ["하이퍼파라미터", "튜닝 전", "튜닝 후", "변화"]

draw_table(axes[0], "성능 지표", metric_rows, col_labels_metric)
draw_table(axes[1], "하이퍼파라미터", param_rows, col_labels_param)

fig.suptitle("XGBoost 튜닝 전후 비교", color=INK, fontsize=17, fontweight="bold", y=1.01)
fig.text(0.5, -0.01,
          "튜닝 전: scripts/05, 07 · 튜닝 후: scripts/08, 09 (동일 8:2 분할 / 5-fold CV, random_state=42)",
          ha="center", color=MUTED, fontsize=9)

plt.tight_layout()
plt.savefig("output/10_tuning_before_after.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

print("저장 완료: output/10_tuning_before_after.png")
