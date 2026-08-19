"""이상치 탐지: IQR / z-score / 모델 잔차 3종을 대조.

핵심은 세 번째다. 날씨가 좋은 날의 높은 수요는 IQR 로도 잡히지만,
"이 날씨면 3만 건이어야 하는데 7만 건"이라는 사실은 잔차로만 드러난다.

판단은 2015년(train)에서만 한다. 2016년을 보고 결정하면 테스트 정보 유입이다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import common as C

plt = C.setup_matplotlib()

tr, _ = C.load_train_test()
y = tr[C.TARGET]

print("=" * 70)
print("1. IQR 기준")
print("=" * 70)
q1, q3 = y.quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
iqr_mask = (y < lo) | (y > hi)
print(f"Q1 {q1:,.0f} / Q3 {q3:,.0f} / IQR {iqr:,.0f}")
print(f"정상 범위: {lo:,.0f} ~ {hi:,.0f}")
print(f"이상치 {int(iqr_mask.sum())}일")
print(tr.loc[iqr_mask, ["dteday", C.TARGET, "t1", "weather_code", "workingday", "n_hours"]]
      .to_string(index=False))

print()
print("=" * 70)
print("2. z-score 기준 (|z| > 3)")
print("=" * 70)
z = (y - y.mean()) / y.std()
z_mask = z.abs() > 3
print(f"평균 {y.mean():,.0f} / 표준편차 {y.std():,.0f}")
print(f"이상치 {int(z_mask.sum())}일")
print(tr.loc[z_mask, ["dteday", C.TARGET]].assign(z=z[z_mask].round(2)).to_string(index=False))

print()
print("=" * 70)
print("3. 모델 잔차 기준 (선형회귀 잔차 |z| > 3)  <- 핵심")
print("=" * 70)
X = tr[C.FEATURES]
lr = LinearRegression().fit(X, y)
pred = lr.predict(X)
resid = y - pred
rz = (resid - resid.mean()) / resid.std()
r_mask = rz.abs() > 3
print("날씨·달력 변수로 설명한 뒤 남는 오차가 큰 날 = 변수로 설명되지 않는 날")
print(f"이상치 {int(r_mask.sum())}일")
det = tr.loc[r_mask, ["dteday", C.TARGET, "t1", "weather_code", "workingday"]].copy()
det["예측"] = pred[r_mask].round(0)
det["잔차"] = resid[r_mask].round(0)
det["잔차z"] = rz[r_mask].round(2)
print(det.to_string(index=False))

print()
print("=" * 70)
print("4. 세 방법 종합")
print("=" * 70)
flags = pd.DataFrame({
    "dteday": tr["dteday"].dt.strftime("%Y-%m-%d"),
    C.TARGET: y, "IQR": iqr_mask, "z-score": z_mask, "잔차": r_mask,
})
flagged = flags[flags[["IQR", "z-score", "잔차"]].any(axis=1)].copy()
flagged["탐지수"] = flagged[["IQR", "z-score", "잔차"]].sum(axis=1)
print(flagged.sort_values("탐지수", ascending=False).to_string(index=False))
print()
three = flagged[flagged["탐지수"] == 3]["dteday"].tolist()
print(f"세 방법 모두가 지목한 날: {three}")
print()
print("두 날 모두 날씨 좋음·평일·24시간 온전 기록이라 데이터 오류가 아니다.")
print("유력한 가설은 런던 지하철 파업이나, 외부 자료로 확인하지 않았으므로")
print("미검증 가설로 둔다. 원인 변수가 데이터에 없다는 점이 중요하다.")

print()
print("=" * 70)
print("5. 기록 부실일 (coverage < 0.90)")
print("=" * 70)
low = tr[tr["coverage"] < C.COVERAGE_MIN]
print(f"{len(low)}일 — cnt 가 실제보다 낮게 집계된 날 (측정 누락이지 수요 감소가 아님)")
print(low[["dteday", "n_hours", "coverage", C.TARGET]].to_string(index=False))

print()
print("=" * 70)
print("6. 처리 방침 3안 — 학습셋 크기 비교")
print("=" * 70)
base = C.filter_coverage(tr, verbose=False)
a = base
b = C.drop_outliers(base)
c, cap = C.clip_outliers(base, 0.99)
print(f"(A) 유지      : {len(a)}일  cnt 최대 {a[C.TARGET].max():,.0f}")
print(f"(B) 제거      : {len(b)}일  cnt 최대 {b[C.TARGET].max():,.0f}")
print(f"(C) 클리핑    : {len(c)}일  cnt 최대 {c[C.TARGET].max():,.0f}  (99분위 {cap:,.0f})")
print()
print("어느 쪽이 좋은지는 2016년 성능으로만 판정할 수 있다 -> 12에서 비교한다.")

# ---------- 시각화 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=C.SURFACE)

ax = axes[0]
C.style_axes(ax)
bp = ax.boxplot([y], tick_labels=["2015 cnt"], patch_artist=True, widths=0.4,
                medianprops=dict(color=C.INK, linewidth=1.5),
                whiskerprops=dict(color=C.AXIS), capprops=dict(color=C.AXIS),
                flierprops=dict(markeredgecolor=C.WARN, markersize=6))
bp["boxes"][0].set_facecolor(C.BLUE)
bp["boxes"][0].set_alpha(0.55)
bp["boxes"][0].set_edgecolor(C.BLUE)
ax.axhline(hi, color=C.WARN, linewidth=1, linestyle="--")
ax.text(1.28, hi, f" IQR 상한 {hi:,.0f}", color=C.WARN, fontsize=9, va="center")
ax.set_ylabel("일일 대여 건수 (cnt)", color=C.SEC_INK)
ax.set_title("분포와 IQR 상한", color=C.INK, fontsize=13, pad=10)

ax = axes[1]
C.style_axes(ax, axis="both")
normal = ~r_mask
ax.scatter(pred[normal], resid[normal], s=20, color=C.BLUE, alpha=0.5, linewidths=0)
ax.scatter(pred[r_mask], resid[r_mask], s=70, color=C.WARN, linewidths=0, zorder=3)
for _, row in tr.loc[r_mask].iterrows():
    i = row.name
    ax.annotate(row["dteday"].strftime("%m-%d"), (pred[i], resid[i]),
                textcoords="offset points", xytext=(8, 0),
                color=C.WARN, fontsize=9, va="center")
ax.axhline(0, color=C.AXIS, linewidth=1)
ax.set_xlabel("선형회귀 예측값", color=C.SEC_INK)
ax.set_ylabel("잔차 (실제 - 예측)", color=C.SEC_INK)
ax.set_title("잔차 기준 이상치 — 변수로 설명되지 않는 날", color=C.INK, fontsize=13, pad=10)

fig.suptitle("2015년 이상치 탐지", color=C.INK, fontsize=15, fontweight="bold", y=1.0)
plt.tight_layout()
plt.savefig("output/05_outlier_detection.png", dpi=150, facecolor=C.SURFACE,
            bbox_inches="tight")
plt.close(fig)
print()
print("저장: output/05_outlier_detection.png")
