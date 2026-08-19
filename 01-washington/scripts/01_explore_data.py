"""데이터 구조 확인: 행/열 개수, 컬럼 타입, 결측치, 상위 데이터 미리보기."""

import pandas as pd

df = pd.read_csv("data/day.csv")

print("=" * 50)
print("1. 행, 열 개수")
print("=" * 50)
print(f"행(row): {df.shape[0]}, 열(column): {df.shape[1]}")

print()
print("=" * 50)
print("2. 컬럼명 전체")
print("=" * 50)
print(df.columns.tolist())

print()
print("=" * 50)
print("3. 컬럼별 데이터 타입")
print("=" * 50)
print(df.dtypes)

print()
print("=" * 50)
print("4. 결측치 개수")
print("=" * 50)
print(df.isnull().sum())

print()
print("=" * 50)
print("5. 상위 데이터 미리보기 (head 5)")
print("=" * 50)
print(df.head())
