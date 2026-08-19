"""EDA 시각화 4종: 기온-대여량, 날씨별 분포, 계절x요일 히트맵, 월별 추이(2015 vs 2016)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import common as C

plt = C.setup_matplotlib()

tr, te = C.load_train_test()

# ---------- 1. 기온 vs 대여량 ----------
# 런던 t1 은 이미 섭씨 원값이라 워싱턴과 달리 역변환이 필요 없다.
fig, ax = plt.subplots(figsize=(8, 6), facecolor=C.SURFACE)
C.style_axes(ax, axis="both")
ax.scatter(tr["t1"], tr[C.TARGET], s=20, color=C.BLUE, alpha=0.55, linewidths=0)
corr = tr["t1"].corr(tr[C.TARGET])
ax.set_title("기온에 따른 일일 대여 건수 (2015년)", color=C.INK, fontsize=14, pad=12)
ax.set_xlabel("기온 (°C)", color=C.SEC_INK)
ax.set_ylabel("일일 대여 건수 (cnt)", color=C.SEC_INK)
ax.text(0.02, 0.96, f"상관계수 (Pearson r) = {corr:.2f}", transform=ax.transAxes,
        color=C.SEC_INK, fontsize=10, va="top")
plt.tight_layout()
plt.savefig("output/04_temp_vs_cnt.png", dpi=150, facecolor=C.SURFACE)
plt.close(fig)
print(f"저장: output/04_temp_vs_cnt.png   (기온-대여량 상관계수 {corr:.4f})")

# ---------- 2. 날씨 코드별 분포 ----------
# weather_code 는 일별 집계 과정에서 원본에 없는 값(5,6,8,9)이 섞였다(01 단계 참고).
# 등급이 아니라 코드이므로 라벨에 원본 코드 여부를 함께 표시한다.
SRC_CODES = {1, 2, 3, 4, 7, 10, 26}
codes = sorted(tr["weather_code"].unique())
groups = [tr.loc[tr["weather_code"] == c, C.TARGET] for c in codes]
labels = [f"{c}" if c in SRC_CODES else f"{c}*" for c in codes]

fig, ax = plt.subplots(figsize=(9, 6), facecolor=C.SURFACE)
C.style_axes(ax)
box = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.5,
                 medianprops=dict(color=C.INK, linewidth=1.5),
                 whiskerprops=dict(color=C.AXIS), capprops=dict(color=C.AXIS),
                 flierprops=dict(markeredgecolor=C.MUTED, markersize=4))
for patch, c in zip(box["boxes"], codes):
    patch.set_facecolor(C.BLUE if c in SRC_CODES else C.RECESS)
    patch.set_alpha(0.55)
    patch.set_edgecolor(C.BLUE if c in SRC_CODES else C.MUTED)

ax.set_title("날씨 코드별 일일 대여 건수 분포 (2015년)", color=C.INK, fontsize=14, pad=12)
ax.set_xlabel("weather_code   (* = 집계 평균으로 생긴 값, 원본 코드 아님)", color=C.SEC_INK)
ax.set_ylabel("일일 대여 건수 (cnt)", color=C.SEC_INK)
plt.tight_layout()
plt.savefig("output/04_weather_vs_cnt.png", dpi=150, facecolor=C.SURFACE)
plt.close(fig)
print("저장: output/04_weather_vs_cnt.png")

summary = tr.groupby("weather_code")[C.TARGET].agg(["count", "mean"]).round(0)
summary.index = [f"{int(c)}{'' if c in SRC_CODES else ' *'}" for c in summary.index]
print("\n=== 날씨 코드별 평균 대여 건수 (2015) ===")
print(summary.to_string())

# ---------- 3. 계절 x 요일 히트맵 ----------
d = tr.copy()
d["season_label"] = d["season"].map(C.SEASON_LABEL)
d["weekday_label"] = d["weekday"].map(C.WEEKDAY_LABEL)
pivot = d.pivot_table(index="season_label", columns="weekday_label",
                      values=C.TARGET, aggfunc="mean") \
         .reindex(index=C.SEASON_ORDER, columns=C.WEEKDAY_ORDER)

blue_ramp = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("seq_blue", blue_ramp)

fig, ax = plt.subplots(figsize=(9, 6), facecolor=C.SURFACE)
ax.set_facecolor(C.SURFACE)
im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
ax.set_xticks(range(len(C.WEEKDAY_ORDER)))
ax.set_xticklabels(C.WEEKDAY_ORDER, color=C.MUTED)
ax.set_yticks(range(len(C.SEASON_ORDER)))
ax.set_yticklabels(C.SEASON_ORDER, color=C.MUTED)
ax.set_title("계절 x 요일 평균 대여 건수 (2015년)", color=C.INK, fontsize=14, pad=12)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

vmin, vmax = np.nanmin(pivot.values), np.nanmax(pivot.values)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        norm = (v - vmin) / (vmax - vmin)
        ax.text(j, i, f"{v:,.0f}", ha="center", va="center",
                color="#ffffff" if norm > 0.55 else C.INK, fontsize=9)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.outline.set_visible(False)
cbar.ax.tick_params(colors=C.MUTED, length=0)
cbar.set_label("평균 대여 건수 (cnt)", color=C.SEC_INK)
plt.tight_layout()
plt.savefig("output/04_season_weekday_heatmap.png", dpi=150, facecolor=C.SURFACE)
plt.close(fig)
print("\n저장: output/04_season_weekday_heatmap.png")
print("\n=== 계절 x 요일 평균 대여 건수 (2015) ===")
print(pivot.round(0).to_string())

# ---------- 4. 월별 추이 (2015 vs 2016) ----------
# 학습 연도와 예측 연도의 계절 패턴이 실제로 닮았는지 눈으로 확인한다.
m_tr = tr.groupby("mnth")[C.TARGET].mean()
m_te = te.groupby("mnth")[C.TARGET].mean()

fig, ax = plt.subplots(figsize=(10, 6), facecolor=C.SURFACE)
C.style_axes(ax)
ax.plot(m_tr.index, m_tr.values, marker="o", color=C.BLUE, linewidth=2,
        label="2015 (train)")
ax.plot(m_te.index, m_te.values, marker="s", color=C.ACCENT, linewidth=2,
        label="2016 (test)")
ax.set_xticks(range(1, 13))
ax.set_xlabel("월", color=C.SEC_INK)
ax.set_ylabel("일평균 대여 건수 (cnt)", color=C.SEC_INK)
ax.set_title("월별 일평균 대여 건수 — 2015 vs 2016", color=C.INK, fontsize=14, pad=12)
leg = ax.legend(frameon=False, labelcolor=C.SEC_INK)
plt.tight_layout()
plt.savefig("output/04_monthly_trend.png", dpi=150, facecolor=C.SURFACE)
plt.close(fig)
print("\n저장: output/04_monthly_trend.png")

comp = pd.DataFrame({"2015": m_tr, "2016": m_te})
comp["차이(%)"] = (comp["2016"] / comp["2015"] - 1) * 100
print("\n=== 월별 일평균 대여 건수 ===")
print(comp.round(1).to_string())
print(f"\n월별 패턴 상관계수 (2015 vs 2016): {m_tr.corr(m_te):.4f}")
