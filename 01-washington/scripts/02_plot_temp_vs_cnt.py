"""기온(temp) vs 일일 대여 건수(cnt) 산점도와 Pearson 상관계수."""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/day.csv")

# day.csv 문서 기준: temp는 (t - t_min) / (t_max - t_min)로 정규화된 값
# t_min=-8, t_max=+39 (섭씨) -> 실제 온도로 역변환해서 보기 쉽게 표시
df["temp_celsius"] = df["temp"] * (39 - (-8)) + (-8)

fig, ax = plt.subplots(figsize=(8, 6), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")

ax.scatter(
    df["temp_celsius"],
    df["cnt"],
    s=18,
    color="#2a78d6",
    alpha=0.55,
    linewidths=0,
)

ax.set_title("기온에 따른 자전거 대여 건수", color="#0b0b0b", fontsize=14, pad=12)
ax.set_xlabel("기온 (°C)", color="#52514e")
ax.set_ylabel("일일 대여 건수 (cnt)", color="#52514e")

ax.grid(True, color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")

ax.tick_params(colors="#898781")

corr = df["temp_celsius"].corr(df["cnt"])
ax.text(
    0.02, 0.96, f"상관계수 (Pearson r) = {corr:.2f}",
    transform=ax.transAxes, color="#52514e", fontsize=10,
    verticalalignment="top",
)

plt.tight_layout()
plt.savefig("output/02_temp_vs_cnt.png", dpi=150, facecolor="#fcfcfb")
print("저장 완료: output/02_temp_vs_cnt.png")
print(f"상관계수: {corr:.4f}")
