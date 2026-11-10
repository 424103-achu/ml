import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),
         "../../ML/task7"))  # data files live here


# Logistic Regression (from ML-Lab-Practice-master/Logistics Regression/normal.py + newton.py)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, method="gd", lr=0.1, iters=1000):
        self.method, self.lr, self.iters = method, lr, iters

    def loss(self, X, y):
        p = np.clip(sigmoid(X @ self.w), 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X, y):
        X = np.c_[np.ones(len(X)), X]                    # intercept column
        self.w = np.zeros(X.shape[1])
        self.history = []
        for _ in range(self.iters):
            p = sigmoid(X @ self.w)
            if self.method == "gd":                      # logreg_gd step
                grad = X.T @ (p - y) / len(y)
                self.w = self.w - self.lr * grad
            else:                                        # logreg_newton step
                grad = X.T @ (p - y)
                S = np.diag(p * (1 - p))
                H = X.T @ S @ X
                self.w = self.w - np.linalg.solve(H, grad)
            self.history.append(self.loss(X, y))


    def predict(self, X):
        X = np.hstack((np.ones((len(X), 1)), X))
        return (sigmoid(X @ self.w) >= 0.5).astype(int)


train = pd.read_csv("ds1_train.csv")
valid = pd.read_csv("ds1_valid.csv")
X_train, y_train = train[["x_1", "x_2"]].values, train["y"].values
X_valid, y_valid = valid[["x_1", "x_2"]].values, valid["y"].values

gd = LogisticRegression(method="gd", lr=0.1, iters=5000)
gd.fit(X_train, y_train)
newton = LogisticRegression(method="newton", iters=10)
newton.fit(X_train, y_train)

print("Gradient Descent: w =", gd.w.round(4),
      " iterations =", len(gd.history), " valid acc =", np.mean(gd.predict(X_valid) == y_valid))
print("Newton's Method : w =", newton.w.round(4),
      " iterations =", len(newton.history), " valid acc =", np.mean(newton.predict(X_valid) == y_valid))
print("Newton reaches the minimum in a few steps; GD needs thousands of small steps.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(gd.history, label="Gradient Descent")
ax[0].plot(newton.history, "o-", label="Newton")
ax[0].set_xscale("log")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Loss")
ax[0].legend()
ax[0].set_title("Loss curves")

ax[1].scatter(valid.x_1, valid.x_2, c=valid.y, cmap="bwr", s=15)
x1 = np.linspace(valid.x_1.min(), valid.x_1.max(), 50)
for w, name in [(gd.w, "GD"), (newton.w, "Newton")]:
    ax[1].plot(x1, -(w[0] + w[1] * x1) / w[2], label=name)
ax[1].set_ylim(valid.x_2.min() - 1, valid.x_2.max() + 1)
ax[1].legend()
ax[1].set_title("Decision boundary (validation set)")
plt.tight_layout()
plt.show()
