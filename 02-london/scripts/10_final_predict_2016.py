"""최종 모델(XGBoost) 학습 -> 2016년 예측 -> 성능 확정 및 저장.

이상치 처리 방침(A 유지 / B 제거 / C 클리핑)은 2015년 CV 로 고른다.
2016 성능을 보고 고르면 테스트 정보 유입이다. 방침별 2016 결과 비교는
12 에서 사후 기록으로만 다룬다.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from xgboost import XGBRegressor

import common as C

plt = C.setup_matplotlib()

BEST = json.load(open("output/09_best_params.json", encoding="utf-8"))
PARAMS = BEST["best_params"]
print(f"09 선택 모델: {BEST['selected_model']}   CV R2 {BEST['cv_r2']:.4f}")
print(f"파라미터: {PARAMS}")
print()


def build_model():
    return XGBRegressor(**PARAMS, random_state=C.RANDOM_STATE)


tr_raw, te = C.load_train_test()
base = C.filter_coverage(tr_raw).sort_values("dteday").reset_index(drop=True)

clipped, cap = C.clip_outliers(base, 0.99)
strategies = {"A 유지": base, "B 제거": C.drop_outliers(base), "C 클리핑": clipped}

print()
print("=" * 70)
print("이상치 처리 방침 선택 (2015년 월블록 CV — 2016 미사용)")
print("=" * 70)
rows = []
for name, df in strategies.items():
    X, y = C.make_xy(df)
    g = df["dteday"].dt.month
    s = cross_val_score(build_model(), X, y, cv=GroupKFold(n_splits=6),
                        groups=g, scoring="r2")
    rows.append({"방침": name, "학습일수": len(df), "CV R2": s.mean(), "std": s.std()})
    print(f"{name:<9} {len(df):>4}일   CV R2 {s.mean():.4f} ± {s.std():.4f}")

cv_df = pd.DataFrame(rows)
best_strategy = cv_df.loc[cv_df["CV R2"].idxmax(), "방침"]
train = strategies[best_strategy]
print()
print(f"선택: {best_strategy}  ({len(train)}일로 최종 학습)")

X_tr, y_tr = C.make_xy(train)
X_te, y_te = C.make_xy(te)
model = build_model().fit(X_tr, y_tr)
y_pred = model.predict(X_te)
metrics = C.evaluate(y_te, y_pred)

print()
print("=" * 70)
print("2016년 예측 성능 (최종)")
print("=" * 70)
print(f"학습 : 2015년 {len(X_tr)}일 x 피처 {len(C.FEATURES)}개")
print(f"예측 : 2016년 {len(X_te)}일")
print()
print(f"R2   : {metrics['R2']:.4f}")
print(f"RMSE : {metrics['RMSE']:,.2f}")
print(f"MAE  : {metrics['MAE']:,.2f}")
print(f"MAPE : {(np.abs(y_te - y_pred) / y_te).mean() * 100:.1f}%")
print()
print("비교:")
print("  전체 평균 (기준선)   R2 -0.0086 / MAE 7,123")
print("  요일x월 평균 (기준선) R2  0.5168 / MAE 4,692")
print(f"  최종 모델            R2 {metrics['R2']:>7.4f} / MAE {metrics['MAE']:,.0f}")

Path("output").mkdir(exist_ok=True)
joblib.dump(model, "output/10_final_model.joblib")
info = {
    "model": "XGBRegressor",
    "params": {**PARAMS, "random_state": C.RANDOM_STATE},
    "feature_cols": C.FEATURES,
    "target": C.TARGET,
    "train": {"file": C.TRAIN_PATH, "n_days": len(X_tr),
              "coverage_min": C.COVERAGE_MIN, "outlier_strategy": best_strategy},
    "test": {"file": C.TEST_PATH, "n_days": len(X_te)},
    "cv": {"method": "GroupKFold(6, groups=month)", "r2": BEST["cv_r2"]},
    "test_metrics": metrics,
}
with open("output/10_final_model_info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

pred_df = te[["dteday", C.TARGET]].copy()
pred_df["predicted"] = y_pred.round(1)
pred_df["residual"] = (pred_df[C.TARGET] - pred_df["predicted"]).round(1)
pred_df.to_csv("output/10_predictions_2016.csv", index=False)

print()
print("저장: output/10_final_model.joblib")
print("저장: output/10_final_model_info.json")
print("저장: output/10_predictions_2016.csv")

fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor=C.SURFACE,
                         gridspec_kw={"height_ratios": [2, 1]})
ax = axes[0]
C.style_axes(ax)
ax.plot(pred_df["dteday"], pred_df[C.TARGET], color=C.BLUE, linewidth=1.4, label="실제")
ax.plot(pred_df["dteday"], pred_df["predicted"], color=C.WARN, linewidth=1.4,
        alpha=0.85, label="예측")
ax.set_ylabel("일일 대여 건수 (cnt)", color=C.SEC_INK)
ax.set_title(f"2016년 실제 vs 예측 (XGBoost)    R² {metrics['R2']:.3f} · "
             f"RMSE {metrics['RMSE']:,.0f} · MAE {metrics['MAE']:,.0f}",
             color=C.SEC_INK, fontsize=11, pad=10, loc="left")
ax.legend(frameon=False, labelcolor=C.SEC_INK)

ax = axes[1]
C.style_axes(ax)
colors = [C.WARN if r < 0 else C.ACCENT for r in pred_df["residual"]]
ax.bar(pred_df["dteday"], pred_df["residual"], color=colors, width=1.0)
ax.axhline(0, color=C.AXIS, linewidth=1)
ax.set_ylabel("잔차 (실제 - 예측)", color=C.SEC_INK)
ax.set_xlabel("날짜", color=C.SEC_INK)

fig.suptitle("2015년 학습 모델의 2016년 예측 결과", color=C.INK, fontsize=15,
             fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig("output/10_actual_vs_predicted.png", dpi=150, facecolor=C.SURFACE,
            bbox_inches="tight")
plt.close(fig)
print("저장: output/10_actual_vs_predicted.png")
