import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.train_x = X
        self.train_y = y
    
    def predict(self, X):
        pred = []
        for x in X:
            dist = np.sum((self.train_x - x) ** 2, axis=1)
            nearest = self.train_y[np.argsort(dist)[:self.k]]
            pred.append(np.argmax(np.bincount(nearest)))
        return pred
