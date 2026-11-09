import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task1"))  # data files live here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("student_placement_synthetic.csv")


def describe(name, x):
    print(f"\n--- {name} ---")
    kind = "discrete" if np.all(x == np.round(x)) else "continuous"
    print("Type:", kind, "| min:", x.min(), "| max:", x.max(), "| skew:", round(stats.skew(x), 3))


def plot_gaussian(a, x, title):
    mu, sd = x.mean(), x.std()
    print(f"Guess: Gaussian  mu={mu:.2f}  sigma={sd:.2f}")
    a.hist(x, bins=40, density=True, color="lightgray", edgecolor="black")
    t = np.linspace(x.min(), x.max(), 200)
    a.plot(t, stats.norm.pdf(t, mu, sd), "r-", lw=2)
    a.set_title(title)


def plot_binomial(a, x, n, title):
    p = x.mean() / n
    print(f"Guess: Binomial  n={n}  p={p:.3f}")
    k = np.arange(0, n + 1)
    a.hist(x, bins=np.arange(-0.5, n + 1.5), density=True, color="lightgray", edgecolor="black")
    a.plot(k, stats.binom.pmf(k, n, p), "ro-", lw=2)
    a.set_title(title)


fig, ax = plt.subplots(2, 3, figsize=(14, 8))

x = df["aptitude_score"].values
describe("A: aptitude_score", x)
plot_gaussian(ax[0, 0], x, "Aptitude ~ Gaussian")

x = df["salary_package_lpa"].dropna().values
describe("B: salary_package_lpa", x)
plot_gaussian(ax[0, 1], x, "Salary ~ Gaussian")

x = df["projects_count"].values
describe("C1: projects_count", x)
plot_binomial(ax[0, 2], x, int(x.max()), "Projects ~ Binomial")

x = df["certifications"].values
describe("C2: certifications", x)
plot_binomial(ax[1, 0], x, int(x.max()), "Certifications ~ Binomial")

status = df["placement_status"].values
chunks = len(status) // 20
x = status[:chunks * 20].reshape(chunks, 20).sum(axis=1)
describe("D: placed per 20", x)
plot_binomial(ax[1, 1], x, 20, "Placed in 20 ~ Binomial(20, p)")

ax[1, 2].axis("off")
plt.tight_layout()
plt.show()
