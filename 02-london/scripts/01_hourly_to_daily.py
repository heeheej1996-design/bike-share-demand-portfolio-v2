"""런던 시간별 데이터를 일별로 집계.

집계 규칙은 워싱턴 D.C. 데이터에서 역추출한 규칙서
(../01-washington/docs/HOURLY_TO_DAILY_AGGREGATION.md)를 그대로 적용한다.

    누적량(flow)      cnt                            -> sum
    상태량(level)     t1, t2, hum, wind_speed        -> mean
    순서형(ordinal)   weather_code                   -> mean 후 half-up 반올림
    날짜 속성         season, is_holiday, is_weekend -> first (상수 검사 후)
    행 식별자         instant                        -> 1부터 날짜순 재생성

런던 원본에 없어 새로 만드는 컬럼: dteday, instant, yr, mnth, weekday, workingday,
n_hours, coverage.
워싱턴에 있으나 런던에 만들 수 없는 컬럼: casual, registered (원본에 이용자 구분 없음).

다른 기간의 파일을 넣어도 코드 수정 없이 동작한다. 경로와 기준연도는 인자로 준다.

    python3 scripts/01_hourly_to_daily.py
    python3 scripts/01_hourly_to_daily.py --input data/london_2018.csv --year-base 2015
"""

import argparse

import numpy as np
import pandas as pd

# ── 입력 스키마 (원본 컬럼명이 바뀌면 여기만 고친다) ──────────────────────
TS_COL = "timestamp"
SUM_COLS = ["cnt"]
MEAN_COLS = ["t1", "t2", "hum", "wind_speed"]
ROUND_COLS = ["weather_code"]
CONST_COLS = ["season", "is_holiday", "is_weekend"]

DATE_COL = "dteday"
HOURS_PER_DAY = 24

OUT_COLS = [
    "instant", DATE_COL,
    "season", "yr", "mnth", "holiday", "weekday", "workingday", "is_weekend",
    "weather_code",
    "t1", "t2", "hum", "wind_speed",
    "cnt",
    "n_hours", "coverage",
]


def half_up(s):
    """사사오입. np.round / .round() 는 half-even 이라 쓰면 안 됨.

    weather_code 는 항상 양수라 np.floor(x + 0.5) 로 충분하다.
    음수가 들어올 수 있는 컬럼에 재사용한다면 부호를 분리해야 한다.
    """
    return np.floor(s + 0.5).astype(int)


def check_constant(hourly, cols, date_col=DATE_COL):
    """날짜 속성 컬럼이 하루 안에서 상수인지 검사. 빈 결과여야 정상."""
    n = hourly.groupby(date_col)[cols].nunique()
    return n[(n > 1).any(axis=1)]


def hourly_to_daily(hourly, year_base=None, fill_missing_dates=False):
    hourly = hourly.copy()
    hourly[TS_COL] = pd.to_datetime(hourly[TS_COL])
    hourly[DATE_COL] = hourly[TS_COL].dt.normalize()

    bad = check_constant(hourly, CONST_COLS)
    if len(bad):
        raise ValueError(f"날짜 속성이 하루 안에서 상수가 아닌 날 {len(bad)}건:\n{bad}")

    g = hourly.groupby(DATE_COL)
    daily = pd.concat([
        g[SUM_COLS].sum(),
        g[MEAN_COLS].mean(),
        g[ROUND_COLS].mean().apply(half_up),
        g[CONST_COLS].first(),
    ], axis=1).sort_index()

    daily["n_hours"] = g.size()

    if fill_missing_dates:
        full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full)
        daily["n_hours"] = daily["n_hours"].fillna(0).astype(int)

    daily.index.name = DATE_COL

    # ── 날짜에서 파생하는 달력 컬럼 (런던 원본에 없음) ────────────────────
    idx = pd.DatetimeIndex(daily.index)
    base = year_base if year_base is not None else int(idx.year.min())
    daily["yr"] = idx.year - base           # 워싱턴과 동일한 0-based 오프셋
    daily["mnth"] = idx.month
    daily["weekday"] = (idx.dayofweek + 1) % 7   # 워싱턴 기준 0=일요일 (pandas 0=월요일)

    daily["holiday"] = daily["is_holiday"]
    daily["workingday"] = ((1 - daily["is_holiday"]) * (1 - daily["is_weekend"]))

    daily["coverage"] = daily["n_hours"] / HOURS_PER_DAY

    for c in ["season", "holiday", "workingday", "is_weekend"]:
        daily[c] = daily[c].astype("Int64")

    daily = daily.reset_index()
    daily[DATE_COL] = daily[DATE_COL].dt.strftime("%Y-%m-%d")
    daily.insert(0, "instant", np.arange(1, len(daily) + 1))
    return daily[OUT_COLS]


def report(hourly, daily):
    """집계 결과에서 바로 확인해야 할 것들을 출력."""
    print("=" * 60)
    print("입력")
    print("=" * 60)
    print(f"행 수      : {len(hourly):,}")
    print(f"기간       : {hourly[TS_COL].min()} ~ {hourly[TS_COL].max()}")

    print()
    print("=" * 60)
    print("출력")
    print("=" * 60)
    print(f"행 수      : {len(daily):,}일")
    print(f"컬럼       : {len(daily.columns)}개")

    span = pd.date_range(pd.to_datetime(daily[DATE_COL]).min(),
                         pd.to_datetime(daily[DATE_COL]).max(), freq="D")
    missing = span.difference(pd.to_datetime(daily[DATE_COL]))
    print(f"달력상 일수: {len(span):,}일")
    if len(missing):
        print(f"⚠ 24시간 전부 결측이라 행이 없는 날짜 {len(missing)}건: "
              f"{[str(d.date()) for d in missing]}")

    print()
    print("=" * 60)
    print("기록된 시간 수 (coverage = n_hours / 24)")
    print("=" * 60)
    full = (daily["n_hours"] == HOURS_PER_DAY).sum()
    short = (daily["n_hours"] < HOURS_PER_DAY).sum()
    lost = int((HOURS_PER_DAY - daily["n_hours"]).clip(lower=0).sum())
    print(f"24시간 온전   : {full:>4}일")
    print(f"시간이 빠진 날: {short:>4}일 (누락 {lost}시간)")
    print()
    for th in (0.90, 0.75, 0.50):
        n = (daily["coverage"] < th).sum()
        print(f"coverage < {th:.2f} : {n:>4}일")
    low = daily.nsmallest(5, "coverage")[[DATE_COL, "n_hours", "coverage"]]
    print("\n가장 낮은 5일:")
    print(low.to_string(index=False))

    print()
    print("=" * 60)
    print("weather_code — 원본 코드 vs 집계 결과")
    print("=" * 60)
    src = sorted(int(v) for v in hourly["weather_code"].dropna().unique())
    out = sorted(int(v) for v in daily["weather_code"].dropna().unique())
    print(f"시간별 원본에 등장한 코드 : {src}")
    print(f"일별 집계 후 나온 값      : {out}")
    invalid = sorted(set(out) - set(src))
    if invalid:
        n = daily["weather_code"].isin(invalid).sum()
        print(f"⚠ 원본 코드 체계에 없는 값 {invalid} 가 {n}일에 발생했습니다.")
        print("  weather_code 는 1,2,3,4,7,10,26 처럼 간격이 일정하지 않은 코드라")
        print("  평균이 코드 사이의 빈 구간에 떨어집니다. 워싱턴 weathersit(1~4 연속)과")
        print("  다른 점이므로, 이 컬럼을 등급으로 해석할 때 주의가 필요합니다.")


def main():
    p = argparse.ArgumentParser(description="런던 시간별 -> 일별 집계")
    p.add_argument("--input", default="data/london_merged.csv")
    p.add_argument("--output", default="data/london_daily.csv")
    p.add_argument("--year-base", type=int, default=None,
                   help="yr=0 으로 삼을 연도. 기본값은 입력 파일의 첫 연도. "
                        "여러 기간의 파일을 각각 변환할 때는 같은 값을 지정해야 "
                        "yr 이 파일 간에 호환된다.")
    p.add_argument("--fill-missing-dates", action="store_true",
                   help="24시간 전부 결측인 날짜도 행으로 남긴다 (집계값은 NaN).")
    args = p.parse_args()

    hourly = pd.read_csv(args.input, parse_dates=[TS_COL])
    daily = hourly_to_daily(hourly, args.year_base, args.fill_missing_dates)

    report(hourly, daily)

    daily.to_csv(args.output, index=False)
    print()
    print(f"저장 완료: {args.output}")


if __name__ == "__main__":
    main()
