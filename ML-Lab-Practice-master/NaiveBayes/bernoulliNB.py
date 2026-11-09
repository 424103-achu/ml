import numpy as np

class BernoulliNB:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def fit(self, X, y):
        X = (X > 0).astype(float)
        self._classes = np.unique(y)
        self.log_priors_ = np.log([(y == c).mean() for c in self._classes])
        n_c = np.array([(y == c).sum() for c in self._classes])[:, None]
        n_c_j = np.array([X[y == c].sum(axis=0) for c in self._classes])
        self.theta_ = (n_c_j + self.alpha) / (n_c + 2 * self.alpha)
        return self
    
    def predict(self, X):
        X = (X > 0).astype(float)
        theta_log_ = np.log(self.theta_)
        theta_minus_log_ = np.log(1 - self.theta_)
        return self._classes[np.argmax(X @ theta_log_.T + (1 - X) @ theta_minus_log_.T + self.log_priors_, axis=1)]
