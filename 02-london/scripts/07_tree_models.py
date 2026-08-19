"""세 모델 비교: 선형회귀 / 랜덤포레스트 / XGBoost.

학습: 2015년 (coverage>=0.90, 이상치 2일 제외 = 353일)
평가: 2016년 365일 전체
지표: R2, RMSE, MAE

워싱턴은 전체를 8:2 랜덤 분할했지만 여기서는 연도로 완전히 분리한다.
테스트 날짜의 전날·다음날이 학습셋에 들어가지 않아 평가가 정직하다.
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
train = C.drop_outliers(C.filter_coverage(tr))
X_tr, y_tr = C.make_xy(train)
X_te, y_te = C.make_xy(te)

print(f"학습 : 2015년 {len(X_tr)}일")
print(f"평가 : 2016년 {len(X_te)}일")
print(f"피처 : {len(C.FEATURES)}개 — {C.FEATURES}")
print(f"타겟 : {C.TARGET}")
print()

models = {
    "선형회귀": LinearRegression(),
    "랜덤포레스트": RandomForestRegressor(n_estimators=300, random_state=C.RANDOM_STATE),
    # 09 튜닝 결과 반영
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                            min_child_weight=5, subsample=0.8, colsample_bytree=1.0,
                            reg_lambda=1.0, random_state=C.RANDOM_STATE),
}

rows, importances = [], {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    rows.append({"모델": name, **C.evaluate(y_te, model.predict(X_te))})
    if hasattr(model, "feature_importances_"):
        importances[name] = model.feature_importances_

res = pd.DataFrame(rows).set_index("모델")

print("=" * 62)
print("2016년 예측 성능 비교")
print("=" * 62)
print(res.round(2).to_string())
print()
print(f"{'':12}{'R2':>8}{'RMSE':>10}{'MAE':>10}   (낮을수록 좋음: RMSE/MAE)")
print("-" * 62)
print(f"{'기준선*':<12}{0.5168:>8.4f}{6059.37:>10.0f}{4691.64:>10.0f}")
for name, r in res.iterrows():
    print(f"{name:<12}{r['R2']:>8.4f}{r['RMSE']:>10.0f}{r['MAE']:>10.0f}")
print("* 기준선 = 2015년 '요일x월 평균'을 그대로 쓴 나이브 예측 (06 참고)")
print()
best = res["R2"].idxmax()
print(f"가장 좋은 모델: {best}  (R2 {res.loc[best, 'R2']:.4f}, "
      f"MAE {res.loc[best, 'MAE']:,.0f}건)")
print(f"기준선 대비 MAE {4691.64 - res.loc[best, 'MAE']:,.0f}건 감소")

print()
print("=" * 62)
print("변수 중요도 (트리 모델)")
print("=" * 62)
imp = pd.DataFrame({"변수": C.FEATURES, **importances})
print(imp.sort_values("랜덤포레스트", ascending=False).round(4).to_string(index=False))

# ---------- 차트 ----------
metrics = ["R2", "RMSE", "MAE"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=C.SURFACE)
names = list(res.index)
x = np.arange(len(names))

for ax, metric in zip(axes, metrics):
    C.style_axes(ax)
    vals = res[metric].values
    best_i = int(np.argmax(vals)) if metric == "R2" else int(np.argmin(vals))
    colors = [C.ACCENT if i == best_i else C.RECESS for i in range(len(vals))]
    bars = ax.bar(x, vals, color=colors, width=0.55)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        label = f"{v:.4f}" if metric == "R2" else f"{v:,.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label + (" ★" if i == best_i else ""),
                ha="center", va="bottom", color=C.INK, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names, color=C.MUTED, fontsize=10)
    ax.set_title(metric + ("  (높을수록 좋음)" if metric == "R2" else "  (낮을수록 좋음)"),
                 color=C.SEC_INK, fontsize=11, pad=10)
    ax.set_ylim(0, max(vals) * 1.18)

fig.suptitle("2016년 예측 성능 — 2015년 353일 학습",
             color=C.INK, fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("output/07_model_comparison.png", dpi=150, facecolor=C.SURFACE,
            bbox_inches="tight")
plt.close(fig)
print()
print("저장: output/07_model_comparison.png")
