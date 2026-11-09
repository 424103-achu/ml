import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task5"))  # data files live here
import pandas as pd
import numpy as np


# LDA classifier (from ML-Lab-Practice-master/LDA/classifier.py)
class LDA:
    def __init__(self):
        self.mu = []
        self.priors = []
        self.sigma_inv = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        n, d = X.shape
        self._classes = y.unique()
        sigma = np.zeros((d, d))
        for k in self._classes:
            X_k = X[y == k]
            mu = X_k.mean()
            self.mu.append(mu)
            self.priors.append(len(X_k) / n)
            sigma += (X_k - mu).T @ (X_k - mu)

        sigma /= (n - len(self._classes))
        self.sigma_inv = np.linalg.inv(sigma)

    def predict(self, X):
        scores = []
        for mu, p in zip(self.mu, self.priors):
            delta = X @ self.sigma_inv @ mu - 0.5 * mu.T @ self.sigma_inv @ mu + np.log(p)
            scores.append(delta)
        return self._classes[np.argmax(np.array(scores).T, axis=1)]


# QDA classifier (from ML-Lab-Practice-master/QDA/classifier.py)
class QDA:
    def __init__(self):
        self.mu = []
        self.prior = []
        self.sigma = []

    def fit(self, X, y):
        n, d = X.shape
        self._classes = np.unique(y)
        for k in self._classes:
            X_k = X[y == k]
            n_k = len(X_k)

            mu = np.mean(X_k, axis=0)
            self.mu.append(mu)
            self.sigma.append((X_k - mu).T @ (X_k - mu) / (n_k - 1))
            self.prior.append(n_k / n)

    def predict(self, X):
        scores = []
        for mu, s, p in zip(self.mu, self.sigma, self.prior):
            diff = X - mu
            mahalanobis = np.diag(diff @ np.linalg.inv(s) @ diff.T)
            delta = -0.5 * np.log(np.linalg.det(s)) - 0.5 * mahalanobis + np.log(p)
            scores.append(delta)
        return self._classes[np.argmax(np.array(scores).T, axis=1)]


# Load data
df = pd.read_csv("SAheart.csv")
if df["famhist"].dtype == object:
    df["famhist"] = df["famhist"].map({"Present": 1, "Absent": 0})

X = df.drop("chd", axis=1).values.astype(float)
y = df["chd"].values

# Train / test split (70 / 30)
idx = np.random.RandomState(42).permutation(len(X))
n_test = int(0.3 * len(X))
X_test, y_test = X[idx[:n_test]], y[idx[:n_test]]
X_train, y_train = X[idx[n_test:]], y[idx[n_test:]]

# Train and evaluate both models
lda = LDA()
lda.fit(pd.DataFrame(X_train), pd.Series(y_train))   # LDA.fit expects DataFrame / Series
qda = QDA()
qda.fit(X_train, y_train)

print("| Model | Accuracy | Precision | Recall |")
print("|---|---|---|---|")
for name, model in [("LDA", lda), ("QDA", qda)]:
    y_pred = model.predict(X_test)
    tp = np.sum((y_pred == 1) & (y_test == 1))
    acc = np.mean(y_pred == y_test)
    precision = tp / np.sum(y_pred == 1)
    recall = tp / np.sum(y_test == 1)
    print(f"| {name} | {acc:.3f} | {precision:.3f} | {recall:.3f} |")
    print(pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=[name]))
