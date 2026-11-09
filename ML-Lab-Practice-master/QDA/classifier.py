import numpy as np

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
