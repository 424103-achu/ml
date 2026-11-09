import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task1"))  # data files live here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("thyroid_cancer_risk_data.csv")


def describe(name, x):
    print(f"\n--- {name} ---")
    kind = "discrete" if np.all(x == np.round(x)) else "continuous"
    print("Type:", kind, "| min:", x.min(), "| max:", x.max(), "| skew:", round(stats.skew(x), 3))


fig, ax = plt.subplots(2, 2, figsize=(10, 8))

x = df["T3_Level"].values
describe("A: T3_Level", x)
lo, hi = x.min(), x.max()
z = (x - lo) / (hi - lo)
m, v = z.mean(), z.var()
a = m * (m * (1 - m) / v - 1)
b = (1 - m) * (m * (1 - m) / v - 1)
print(f"Guess: Beta  a={a:.2f}  b={b:.2f}")
ax[0, 0].hist(x, bins=40, density=True, color="lightgray", edgecolor="black")
t = np.linspace(lo, hi, 200)
ax[0, 0].plot(t, stats.beta.pdf((t - lo) / (hi - lo), a, b) / (hi - lo), "r-", lw=2)
ax[0, 0].set_title("T3_Level ~ Beta")

x = df["Nodule_Size"].values
describe("B: Nodule_Size", x)
lo, hi = x.min(), x.max()
z = (x - lo) / (hi - lo)
m, v = z.mean(), z.var()
a = m * (m * (1 - m) / v - 1)
b = (1 - m) * (m * (1 - m) / v - 1)
print(f"Guess: Beta  a={a:.2f}  b={b:.2f}")
ax[0, 1].hist(x, bins=40, density=True, color="lightgray", edgecolor="black")
t = np.linspace(lo, hi, 200)
ax[0, 1].plot(t, stats.beta.pdf((t - lo) / (hi - lo), a, b) / (hi - lo), "r-", lw=2)
ax[0, 1].set_title("Nodule_Size ~ Beta")

flags = ["Radiation_Exposure", "Iodine_Deficiency", "Smoking", "Obesity"]
x = (df[flags] == "Yes").sum(axis=1).values
describe("C: Risk_Factor_Count", x)
p = x.mean() / 4
print(f"Guess: Binomial  n=4  p={p:.3f}")
k = np.arange(0, 5)
ax[1, 0].hist(x, bins=np.arange(-0.5, 5.5), density=True, color="lightgray", edgecolor="black")
ax[1, 0].plot(k, stats.binom.pmf(k, 4, p), "ro-", lw=2)
ax[1, 0].set_title("Risk_Factor_Count ~ Binomial(4, p)")

diag = (df["Diagnosis"] == "Malignant").astype(int).values[:5000 * 30]
x = diag.reshape(5000, 30).sum(axis=1)
describe("D: Malignant per 30", x)
p = x.mean() / 30
print(f"Guess: Binomial  n=30  p={p:.3f}")
k = np.arange(x.min(), x.max() + 1)
ax[1, 1].hist(x, bins=np.arange(x.min() - 0.5, x.max() + 1.5), density=True,
              color="lightgray", edgecolor="black")
ax[1, 1].plot(k, stats.binom.pmf(k, 30, p), "ro-", lw=2)
ax[1, 1].set_title("Malignant in 30 ~ Binomial(30, p)")

plt.tight_layout()
plt.show()
