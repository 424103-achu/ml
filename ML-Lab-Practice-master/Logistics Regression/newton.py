import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logreg_newton(X, y, iters=10):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y)
        S = np.diag(p * (1 - p))
        H = X.T @ S @ X
        w = w - np.linalg.solve(H, grad)
    return w
