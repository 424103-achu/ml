import numpy as np

def linreg_gd(X, y, lr=0.1, iters=1000):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        grad = 2 / len(y) * X.T @ (X @ w - y)
        w = w - lr * grad
    return w

def rss(X, y, w):
    r = y - X @ w
    return r @ r

def mse(X, y, w):
    return rss(X, y, w) / len(y)
