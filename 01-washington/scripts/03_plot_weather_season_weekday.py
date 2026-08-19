"""날씨 등급별 대여 건수 분포(boxplot)와 계절 x 요일 평균 대여 건수 히트맵."""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/day.csv")

BLUE = "#2a78d6"
INK = "#0b0b0b"
SEC_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED)


# ---------- 1. 날씨 등급별 대여 수 비교 ----------
weathersit_map = {1: "맑음", 2: "흐림/안개", 3: "악천후(비/눈)"}
df["weathersit_label"] = df["weathersit"].map(weathersit_map)

order = ["맑음", "흐림/안개", "악천후(비/눈)"]
groups = [df.loc[df["weathersit_label"] == label, "cnt"] for label in order]

fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
style_axes(ax)
box = ax.boxplot(
    groups, tick_labels=order, patch_artist=True, widths=0.5,
    medianprops=dict(color=INK, linewidth=1.5),
    whiskerprops=dict(color=AXIS),
    capprops=dict(color=AXIS),
    flierprops=dict(markeredgecolor=MUTED, markersize=4),
)
for patch in box["boxes"]:
    patch.set_facecolor(BLUE)
    patch.set_alpha(0.55)
    patch.set_edgecolor(BLUE)

ax.set_title("날씨 등급별 일일 대여 건수 분포", color=INK, fontsize=14, pad=12)
ax.set_ylabel("일일 대여 건수 (cnt)", color=SEC_INK)
plt.tight_layout()
plt.savefig("output/03_weathersit_vs_cnt.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

weather_summary = df.groupby("weathersit_label")["cnt"].agg(["count", "mean"]).reindex(order)
print("=== 날씨 등급별 평균 대여 건수 ===")
print(weather_summary)
print()

# ---------- 2. 계절 x 요일 평균 대여 건수 히트맵 ----------
season_map = {1: "봄", 2: "여름", 3: "가을", 4: "겨울"}
df["season_label"] = df["season"].map(season_map)
season_order = ["봄", "여름", "가을", "겨울"]

# day.csv 문서 기준 weekday: 0=일, 1=월, 2=화, 3=수, 4=목, 5=금, 6=토
weekday_map = {0: "일", 1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토"}
df["weekday_label"] = df["weekday"].map(weekday_map)
weekday_order = ["일", "월", "화", "수", "목", "금", "토"]

pivot = df.pivot_table(
    index="season_label", columns="weekday_label", values="cnt", aggfunc="mean"
).reindex(index=season_order, columns=weekday_order)

# 순차(sequential) 블루 램프: 값이 작을수록 밝고, 클수록 진하게
blue_ramp = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
seq_cmap = LinearSegmentedColormap.from_list("seq_blue", blue_ramp)

fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

im = ax.imshow(pivot.values, cmap=seq_cmap, aspect="auto")

ax.set_xticks(range(len(weekday_order)))
ax.set_xticklabels(weekday_order, color=MUTED)
ax.set_yticks(range(len(season_order)))
ax.set_yticklabels(season_order, color=MUTED)
ax.set_title("계절 x 요일 평균 대여 건수", color=INK, fontsize=14, pad=12)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

# 각 셀에 값 직접 라벨링 (배경 밝기에 따라 글자색 자동 대비)
vmin, vmax = pivot.values.min(), pivot.values.max()
for i in range(len(season_order)):
    for j in range(len(weekday_order)):
        value = pivot.values[i, j]
        norm = (value - vmin) / (vmax - vmin)
        text_color = "#ffffff" if norm > 0.55 else "#0b0b0b"
        ax.text(
            j, i, f"{value:,.0f}",
            ha="center", va="center", color=text_color, fontsize=9,
        )

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.outline.set_visible(False)
cbar.ax.tick_params(colors=MUTED, length=0)
cbar.set_label("평균 대여 건수 (cnt)", color=SEC_INK)

plt.tight_layout()
plt.savefig("output/03_season_weekday_heatmap.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

print("=== 계절 x 요일 평균 대여 건수 ===")
print(pivot.round(0))
