import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logreg_gd(X, y, lr=0.1, iters=1000):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y)
        w = w - lr * grad
    return w
