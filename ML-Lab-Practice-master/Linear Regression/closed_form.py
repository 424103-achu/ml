import numpy as np

def normal(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)
