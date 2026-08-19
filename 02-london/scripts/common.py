"""03~12 스크립트가 공유하는 설정·헬퍼.

피처 목록과 분할 조건을 한 곳에 두어 스크립트마다 어긋나지 않게 한다.
(워싱턴에서는 스크립트마다 drop_cols 를 반복 정의했는데, 하나만 고치면
나머지가 조용히 달라지는 위험이 있어 런던에서는 모듈로 분리했다.)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRAIN_PATH = "data/london_daily_2015.csv"
TEST_PATH = "data/london_daily_2016.csv"

TARGET = "cnt"
DROP_COLS = ["instant", "dteday", "cnt", "yr", "n_hours", "coverage"]
# is_christmas 는 원본에 없는 파생 컬럼이다. load() 가 dteday 에서 자동 생성한다.
# 배경: is_holiday 는 '법정공휴일' 플래그라 12/25 가 주말이면 대체휴일로 옮겨간다.
# 실제로 2016-12-25(일)는 holiday=0 이었고, 그 해 최대 오차(+22,038)가 났다.
FEATURES_BASE = ["season", "mnth", "holiday", "weekday", "workingday", "is_weekend",
                 "weather_code", "t1", "t2", "hum", "wind_speed"]
FEATURES = FEATURES_BASE + ["is_christmas"]

RANDOM_STATE = 42
COVERAGE_MIN = 0.90        # 학습에서 제외할 기록 부실일 기준 (계획 6부)

# 계획 2부에서 확인한 2015년 극단값 (지하철 파업 가설, 미검증)
OUTLIER_DATES = ["2015-07-09", "2015-08-06"]

# ── 차트 스타일 (워싱턴 02/03 과 동일) ─────────────────────────────────
BLUE = "#2a78d6"
ACCENT = "#1baf7a"
WARN = "#d6552a"
INK = "#0b0b0b"
SEC_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
RECESS = "#c9c7bd"

SEASON_LABEL = {0: "봄", 1: "여름", 2: "가을", 3: "겨울"}
SEASON_ORDER = ["봄", "여름", "가을", "겨울"]
WEEKDAY_LABEL = {0: "일", 1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토"}
WEEKDAY_ORDER = ["일", "월", "화", "수", "목", "금", "토"]


def setup_matplotlib():
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def style_axes(ax, axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED)


def add_calendar_features(df):
    """원본에 없는 달력 파생 컬럼을 추가. 날짜만 있으면 계산되므로 외부 데이터가 필요 없다."""
    out = df.copy()
    d = out["dteday"]
    out["is_christmas"] = ((d.dt.month == 12) & (d.dt.day == 25)).astype(int)
    return out


def load(path):
    return add_calendar_features(pd.read_csv(path, parse_dates=["dteday"]))


def load_train_test():
    return load(TRAIN_PATH), load(TEST_PATH)


def make_xy(df, features=None):
    features = FEATURES if features is None else features
    return df[features], df[TARGET]


def filter_coverage(df, min_coverage=COVERAGE_MIN, verbose=True):
    """기록이 부실한 날을 학습셋에서 제외."""
    keep = df[df["coverage"] >= min_coverage].reset_index(drop=True)
    if verbose:
        print(f"coverage >= {min_coverage}: {len(df)}일 -> {len(keep)}일 "
              f"({len(df) - len(keep)}일 제외)")
    return keep


def drop_outliers(df, dates=None):
    dates = OUTLIER_DATES if dates is None else dates
    mask = ~df["dteday"].dt.strftime("%Y-%m-%d").isin(dates)
    return df[mask].reset_index(drop=True)


def clip_outliers(df, quantile=0.99):
    """상위 분위수로 타겟을 winsorize."""
    out = df.copy()
    cap = out[TARGET].quantile(quantile)
    out[TARGET] = out[TARGET].clip(upper=cap)
    return out, cap


def evaluate(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def print_metrics(name, m, width=26):
    print(f"{name:<{width}} R2 {m['R2']:>7.4f} | RMSE {m['RMSE']:>9.2f} | MAE {m['MAE']:>9.2f}")
