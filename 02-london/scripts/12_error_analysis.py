"""2016년 예측 오차 분석 — 어디서 왜 틀렸는지.

성능 수치 하나로 끝내지 않고, 잔차를 구간별로 쪼개 체계적 편향이 있는지 본다.
이상치 처리 방침(A/B/C)의 2016 결과 비교도 여기서 사후 분석으로 다룬다.
(선택은 10 에서 2015 CV 로 이미 끝났다. 여기 결과는 판단 근거가 아니라 기록이다.)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import common as C

plt = C.setup_matplotlib()

info = json.load(open("output/10_final_model_info.json", encoding="utf-8"))
pred = pd.read_csv("output/10_predictions_2016.csv", parse_dates=["dteday"])
_, te = C.load_train_test()

d = te.merge(pred[["dteday", "predicted", "residual"]], on="dteday")
d["abs_err"] = d["residual"].abs()
d["ape"] = (d["abs_err"] / d[C.TARGET]) * 100

print("=" * 74)
print("1. 전체 오차 요약")
print("=" * 74)
print(f"R2   {info['test_metrics']['R2']:.4f}   "
      f"RMSE {info['test_metrics']['RMSE']:,.0f}   "
      f"MAE {info['test_metrics']['MAE']:,.0f}")
print(f"MAPE {d['ape'].mean():.1f}%   중앙 절대오차 {d['abs_err'].median():,.0f}")
print()
over = int((d["residual"] < 0).sum())
print(f"과대예측(예측>실제) {over}일 / 과소예측 {len(d) - over}일")
print(f"잔차 평균 {d['residual'].mean():+,.0f}  (0에 가까울수록 편향 없음)")

print()
print("=" * 74)
print("2. 가장 크게 틀린 20일")
print("=" * 74)
top = d.nlargest(20, "abs_err")[["dteday", C.TARGET, "predicted", "residual",
                                 "ape", "t1", "weather_code", "workingday",
                                 "holiday", "n_hours"]].copy()
top["dteday"] = top["dteday"].dt.strftime("%Y-%m-%d (%a)")
print(top.round(1).to_string(index=False))

print()
print("=" * 74)
print("3. 원인 분류")
print("=" * 74)
worst = d.nlargest(20, "abs_err")
cats = {
    "기록 부실 (coverage<0.90)": (worst["coverage"] < 0.90).sum(),
    "공휴일": (worst["holiday"] == 1).sum(),
    "연말연시 (12/24~1/3)": worst["dteday"].apply(
        lambda x: (x.month == 12 and x.day >= 24) or (x.month == 1 and x.day <= 3)).sum(),
    "과대예측": (worst["residual"] < 0).sum(),
    "과소예측": (worst["residual"] > 0).sum(),
}
for k, v in cats.items():
    print(f"{k:<26} {int(v):>3}일 / 20일")

print()
print("=" * 74)
print("4. 구간별 평균 잔차 — 체계적 편향 확인")
print("=" * 74)
for col, label, mapper in [
    ("mnth", "월", None),
    ("season", "계절", C.SEASON_LABEL),
    ("weekday", "요일", C.WEEKDAY_LABEL),
    ("weather_code", "날씨코드", None),
]:
    g = d.groupby(col).agg(일수=("residual", "size"),
                           평균잔차=("residual", "mean"),
                           평균절대오차=("abs_err", "mean"))
    if mapper:
        g.index = [mapper.get(i, i) for i in g.index]
    g.index.name = label
    print(f"\n[{label}별]")
    print(g.round(0).to_string())

print()
print("=" * 74)
print("5. 최대 오차 사례 — 12월 25일과 holiday 플래그의 함정")
print("=" * 74)
tr_x, _ = C.load_train_test()
xmas = []
for lbl, df in [("2015(train)", tr_x), ("2016(test)", te)]:
    w = df[(df.dteday.dt.month == 12) & (df.dteday.dt.day.between(23, 27))].copy()
    w.insert(0, "연도", lbl)
    xmas.append(w[["연도", "dteday", C.TARGET, "holiday", "is_weekend", "workingday"]])
xm = pd.concat(xmas)
xm["dteday"] = xm["dteday"].dt.strftime("%m-%d (%a)")
print(xm.to_string(index=False))
print()
print("두 해 모두 12/25 에 대여가 급증한다(주변일의 2~4배). 크리스마스 당일은")
print("런던 지하철이 전면 운휴해 자전거 수요가 몰리는 것으로 보인다.")
print()
print("문제는 holiday 플래그다:")
print("  2015-12-25 (금)  holiday=1  <- 법정공휴일로 표시됨")
print("  2016-12-25 (일)  holiday=0  <- 일요일이라 대체휴일(12/26,27)에 플래그가 붙음")
print()
print("즉 is_holiday 는 '법정공휴일'이지 '크리스마스 당일'이 아니다. 2016 년에는")
print("모델이 12/25 를 평범한 일요일로 보고 14,615 건을 예측했으나 실제는 36,653 건")
print("이었고, 이것이 그 해 최대 오차가 되었다.")
print()
print("-> 개선안: '12월 25일' 자체를 나타내는 달력 피처를 추가하면 해결된다.")
print("   원본 데이터 문제가 아니라 피처 설계 문제다.")

print()
print("=" * 74)
print("6. 이상치 처리 방침별 2016 결과 (사후 기록 — 선택 근거 아님)")
print("=" * 74)
tr_raw, _ = C.load_train_test()
base = C.filter_coverage(tr_raw, verbose=False).sort_values("dteday").reset_index(drop=True)
clipped, cap = C.clip_outliers(base, 0.99)
strategies = {"A 유지": base, "B 제거": C.drop_outliers(base), "C 클리핑": clipped}

PARAMS = {k: v for k, v in info["params"].items() if k != "random_state"}
X_te, y_te = C.make_xy(te)
rows = []
for name, df in strategies.items():
    m = XGBRegressor(**PARAMS, random_state=C.RANDOM_STATE)
    X, y = C.make_xy(df)
    m.fit(X, y)
    rows.append({"방침": name, "학습일수": len(df), **C.evaluate(y_te, m.predict(X_te))})
sel = info["train"]["outlier_strategy"]
res = pd.DataFrame(rows)
res["선택"] = np.where(res["방침"] == sel, "<- 10에서 선택", "")
print(res.round(4).to_string(index=False))
print()
print("2015 CV 로 고른 방침이 2016 에서도 최선인지 확인하는 용도다.")
best_2016 = res.loc[res["R2"].idxmax(), "방침"]
if best_2016 == sel:
    print(f"-> 일치. 2015 CV 기반 선택({sel})이 2016 에서도 최선이었다.")
else:
    print(f"-> 불일치. 2016 기준 최선은 {best_2016} 였다. "
          f"다만 이를 보고 선택을 바꾸면 테스트 정보 유입이므로 {sel} 를 유지한다.")

# ---------- 차트 ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=C.SURFACE)

ax = axes[0]
C.style_axes(ax, axis="both")
ax.scatter(d["predicted"], d[C.TARGET], s=22, color=C.BLUE, alpha=0.55, linewidths=0)
lim = [min(d["predicted"].min(), d[C.TARGET].min()) * 0.95,
       max(d["predicted"].max(), d[C.TARGET].max()) * 1.05]
ax.plot(lim, lim, color=C.WARN, linewidth=1, linestyle="--")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("예측", color=C.SEC_INK); ax.set_ylabel("실제", color=C.SEC_INK)
ax.set_title("실제 vs 예측", color=C.INK, fontsize=12, pad=10)

ax = axes[1]
C.style_axes(ax)
mr = d.groupby("mnth")["residual"].mean()
ax.bar(mr.index, mr.values, color=[C.WARN if v < 0 else C.ACCENT for v in mr.values],
       width=0.6)
ax.axhline(0, color=C.AXIS, linewidth=1)
ax.set_xticks(range(1, 13))
ax.set_xlabel("월", color=C.SEC_INK); ax.set_ylabel("평균 잔차", color=C.SEC_INK)
ax.set_title("월별 평균 잔차 — 체계적 편향", color=C.INK, fontsize=12, pad=10)

ax = axes[2]
C.style_axes(ax)
ax.hist(d["residual"], bins=35, color=C.BLUE, alpha=0.75)
ax.axvline(0, color=C.WARN, linewidth=1.2, linestyle="--")
ax.set_xlabel("잔차 (실제 - 예측)", color=C.SEC_INK)
ax.set_ylabel("일수", color=C.SEC_INK)
ax.set_title("잔차 분포", color=C.INK, fontsize=12, pad=10)

fig.suptitle("2016년 예측 오차 분석", color=C.INK, fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("output/12_error_analysis.png", dpi=150, facecolor=C.SURFACE,
            bbox_inches="tight")
plt.close(fig)
print("\n저장: output/12_error_analysis.png")
