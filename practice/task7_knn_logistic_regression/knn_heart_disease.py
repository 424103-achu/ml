import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task7"))  # data files live here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# KNN classifier (from ML-Lab-Practice-master/KNN/classifier.py)
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.train_x = X
        self.train_y = y

    def predict(self, X):
        pred = []
        for x in X:
            dist = np.sum((self.train_x - x) ** 2, axis=1)
            nearest = self.train_y[np.argsort(dist)[:self.k]]
            pred.append(np.argmax(np.bincount(nearest)))
        return pred


# Load data
df = pd.read_csv("heart-disease.csv", encoding="utf-8-sig").dropna()
X = df.drop("target", axis=1).values.astype(float)
y = df["target"].values

# Train / test split (80 / 20)
idx = np.random.permutation(len(X))
n_test = int(0.2 * len(X))
X_test, y_test = X[idx[:n_test]], y[idx[:n_test]]
X_train, y_train = X[idx[n_test:]], y[idx[n_test:]]

# Scale features
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Try K = 1..20
ks = range(1, 21)
accs = []
for k in ks:
    model = KNN(k)
    model.fit(X_train, y_train)
    accs.append(np.mean(np.array(model.predict(X_test)) == y_test))
best_k = ks[np.argmax(accs)]
print("Best K =", best_k, " accuracy =", max(accs))

# Final model with best K
model = KNN(best_k)
model.fit(X_train, y_train)
y_pred = np.array(model.predict(X_test))
print(pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]))

# Plot
plt.plot(list(ks), accs, "o-")
plt.xlabel("K"); plt.ylabel("Test accuracy")
plt.title("Accuracy vs K")
plt.show()
