import numpy as np

class MultinomialNB:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def fit(self, X, y):
        self._classes = np.unique(y)
        self.log_prior_ = np.log([np.mean(y == c) for c in self._classes])
        
        counts = np.array([X[y == c].sum(axis=0) for c in self._classes]) + self.alpha
        self.log_theta_ = np.log(counts / np.sum(counts, axis=1, keepdims=True))
        return self
    
    def predict(self, X):
        return self._classes[np.argmax(X @ self.log_theta_.T + self.log_prior_, axis=1)]
