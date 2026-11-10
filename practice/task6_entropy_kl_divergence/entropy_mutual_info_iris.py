import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task6"))  # data files live here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
names = iris.feature_names
data = iris.data

binned = np.column_stack([
    pd.cut(data[:, j], bins=10, labels=False)
    for j in range(4)
])


def pmf(x):
    return np.bincount(x, minlength=10) / len(x)


p0 = pmf(binned[:, 0])
print("PMF of", names[0], ":", p0.round(3))
print("Sum =", p0.sum())


def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def joint_entropy(pxy):
    p = pxy[pxy > 0]
    return -np.sum(p * np.log2(p))


def mutual_information(x, y):
    pxy = np.zeros((10, 10))
    for a, b in zip(x, y):
        pxy[a, b] += 1
    pxy /= len(x)
    return entropy(pmf(x)) + entropy(pmf(y)) - joint_entropy(pxy)


MI = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        MI[i, j] = mutual_information(binned[:, i], binned[:, j])

short = ["sepal len", "sepal wid", "petal len", "petal wid"]
print("\nMutual Information matrix:\n", pd.DataFrame(MI, index=short, columns=short).round(3))

off = MI.copy()
np.fill_diagonal(off, -1)
i, j = np.unravel_index(off.argmax(), off.shape)
print(f"\nHighest MI pair: {names[i]} & {names[j]}  (MI = {MI[i, j]:.3f})")

sns.heatmap(MI, annot=True, fmt=".3f", xticklabels=short, yticklabels=short, cmap="viridis")
plt.title("Mutual Information (bits)")
plt.show()
