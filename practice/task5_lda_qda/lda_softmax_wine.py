import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),
         "../../ML/task5"))  # data files live here


# LDA dimension reduction (from ML-Lab-Practice-master/LDA/dimension_reduction.py)
class LDA:
    def __init__(self):
        self.s_w = []
        self.s_b = []
        self.eigen_vec = []
        self.eigen_val = []
        self.eigen_order = []
        self.mu = []

    def fit(self, X, y):
        n, d = X.shape
        self._classes = np.unique(y)

        self.mu = np.mean(X, axis=0)
        self.s_w = np.zeros((d, d))
        self.s_b = np.zeros((d, d))
        for k in self._classes:
            X_c = X[y == k]
            mu_k = np.mean(X_c, axis=0)
            X_cen = X_c - mu_k
            self.s_w += X_cen.T @ X_cen
            diff = (mu_k - self.mu).reshape(-1, 1)
            self.s_b += len(X_c) * (diff @ diff.T)

        self.eigen_val, self.eigen_vec = np.linalg.eig(
            np.linalg.pinv(self.s_w) @ self.s_b)
        self.eigen_order = np.argsort(self.eigen_val)[::-1]

    def transform(self, X, k):
        return (X - self.mu) @ self.eigen_vec[:, self.eigen_order[:k]]


# Multiclass logistic regression (softmax) to classify in LDA space
class SoftmaxRegression:
    def __init__(self, lr=0.1, iters=2000):
        self.lr, self.iters = lr, iters

    def fit(self, X, y):
        self.classes = np.unique(y)
        Xb = np.column_stack([np.ones(len(X)), X])
        Y = (y[:, None] == self.classes).astype(float)    # one-hot labels
        self.weights = np.zeros((Xb.shape[1], len(self.classes)))
        for _ in range(self.iters):                         # gradient descent
            scores = Xb @ self.weights
            P = np.exp(scores - scores.max(axis=1, keepdims=True))
            P /= P.sum(axis=1, keepdims=True)
            self.weights -= self.lr * Xb.T @ (P - Y) / len(Y)

    def predict(self, X):
        Xb = np.column_stack([np.ones(len(X)), X])
        return self.classes[np.argmax(Xb @ self.weights, axis=1)]


# Load data
df = pd.read_csv("Wine.csv")
X = df.drop("Customer_Segment", axis=1).values.astype(float)
y = df["Customer_Segment"].values

# Train / test split (80 / 20)
idx = np.random.permutation(len(X))
n_test = int(0.2 * len(X))
X_test, y_test = X[idx[:n_test]], y[idx[:n_test]]
X_train, y_train = X[idx[n_test:]], y[idx[n_test:]]

# Scale features
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# LDA: reduce 13 features to 2 (eig can return tiny imaginary parts -> keep real part)
lda = LDA()
lda.fit(X_train, y_train)
Z_train = np.real(lda.transform(X_train, k=2))
Z_test = np.real(lda.transform(X_test, k=2))

# Logistic regression (softmax) on the 2 LDA features
clf = SoftmaxRegression(lr=0.1, iters=2000)
clf.fit(Z_train, y_train)
y_pred = clf.predict(Z_test)

# Accuracy and confusion matrix
print("Accuracy:", np.mean(y_pred == y_test))
print(pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]))

# Plot test data in LDA space
for c, col in zip(clf.classes, ["red", "green", "blue"]):
    plt.scatter(Z_test[y_test == c, 0], Z_test[y_test ==
                c, 1], color=col, label=f"Class {c}")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("Wine test data in LDA space")
plt.legend()
plt.show()
