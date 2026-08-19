"""일별 데이터를 연도별 파일로 분리.

01_hourly_to_daily.py 가 만든 london_daily.csv 를 연도별로 나눠 저장한다.
일수가 너무 적은 연도(기본 30일 미만)는 표본이 되지 못하므로 제외한다.
런던 데이터의 2017년은 1/1~1/3 3일뿐이라 이 기준에서 자동으로 빠진다.

    python3 scripts/02_split_by_year.py
    python3 scripts/02_split_by_year.py --years 2015 2016
    python3 scripts/02_split_by_year.py --min-days 100

instant 는 각 파일 안에서 1부터 다시 매긴다. 원본 규칙서 기준으로 instant 는
'그 파일의 행 번호'이지 날짜의 고유 ID가 아니다. 파일을 다시 합칠 일이 있으면
instant 가 아니라 dteday 를 키로 써야 한다.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATE_COL = "dteday"


def split_by_year(daily, years=None, min_days=30):
    """연도별 DataFrame 딕셔너리와 제외된 연도 목록을 돌려준다."""
    daily = daily.copy()
    daily[DATE_COL] = pd.to_datetime(daily[DATE_COL])
    daily["_year"] = daily[DATE_COL].dt.year

    counts = daily["_year"].value_counts().sort_index()
    if years is None:
        keep = [int(y) for y in counts.index if counts[y] >= min_days]
    else:
        keep = [int(y) for y in years]
    dropped = [(int(y), int(counts[y])) for y in counts.index if int(y) not in keep]

    out = {}
    for y in keep:
        part = daily[daily["_year"] == y].drop(columns="_year").sort_values(DATE_COL)
        part = part.reset_index(drop=True)
        part["instant"] = np.arange(1, len(part) + 1)   # 파일 안에서 1부터 재부여
        part[DATE_COL] = part[DATE_COL].dt.strftime("%Y-%m-%d")
        out[y] = part
    return out, dropped


def main():
    p = argparse.ArgumentParser(description="일별 데이터를 연도별로 분리")
    p.add_argument("--input", default="data/london_daily.csv")
    p.add_argument("--outdir", default="data")
    p.add_argument("--prefix", default="london_daily")
    p.add_argument("--years", type=int, nargs="+", default=None,
                   help="남길 연도를 직접 지정 (예: --years 2015 2016). "
                        "지정하지 않으면 --min-days 기준으로 자동 선택.")
    p.add_argument("--min-days", type=int, default=30,
                   help="이 일수 미만인 연도는 제외 (기본 30)")
    args = p.parse_args()

    daily = pd.read_csv(args.input)
    parts, dropped = split_by_year(daily, args.years, args.min_days)

    print("=" * 60)
    print("입력")
    print("=" * 60)
    print(f"파일 : {args.input}")
    print(f"규모 : {len(daily):,}행 x {daily.shape[1]}열")

    if dropped:
        print()
        print("=" * 60)
        print("제외한 연도")
        print("=" * 60)
        for y, n in dropped:
            print(f"{y}년 : {n}일  (기준 {args.min_days}일 미만)")

    print()
    print("=" * 60)
    print("저장 결과")
    print("=" * 60)
    outdir = Path(args.outdir)
    total = 0
    for y, part in parts.items():
        path = outdir / f"{args.prefix}_{y}.csv"
        part.to_csv(path, index=False)
        total += len(part)
        print(f"{y}년 : {len(part):>3}일  "
              f"({part[DATE_COL].iloc[0]} ~ {part[DATE_COL].iloc[-1]})  "
              f"-> {path}")

    print()
    print(f"합계 {total}일 저장  (입력 {len(daily)}일 중 "
          f"{len(daily) - total}일 제외)")


if __name__ == "__main__":
    main()
