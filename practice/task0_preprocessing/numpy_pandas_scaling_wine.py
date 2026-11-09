import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task0"))  # data files live here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

m = np.random.rand(5, 5)
m[m > 0.5] = 1.5
print("Matrix:\n", m.round(2))

df = pd.read_csv("data.csv")
print("\nSummary:\n", df.describe(include="all"))
print("\nMissing % per column:\n", df.isnull().mean() * 100)
df = df.loc[:, df.isnull().mean() <= 0.5]
print("\nColumns kept:", list(df.columns))

wine = pd.read_csv("winequality-red.csv", sep=";").dropna()
features = ["alcohol", "pH", "density", "sulphates"]
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
for a, f in zip(ax.flat, features):
    a.hist(wine[f], bins=30, color="skyblue", edgecolor="black")
    a.set_title(f)
plt.tight_layout()

X = wine.drop(columns=["quality"])
y = wine["quality"]
X_scaled = StandardScaler().fit_transform(X)
print("\nScaled X (first 3 rows):\n", X_scaled[:3].round(2))
print("Mean ~0:", X_scaled.mean(axis=0).round(2))
print("Std  ~1:", X_scaled.std(axis=0).round(2))

plt.show()
