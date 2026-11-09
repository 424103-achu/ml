import numpy as np

class DecisionTree:
    def __init__(self, max_depth = 15):
        self.max_depth = max_depth
        self.tree = {
            "feature": None,
            "threshold": None,
            "left": None,
            "right": None
        }
    
    def gini(self, y):
        _, count = np.unique(y, return_counts=True)
        p = count / len(y)
        return 1 - np.sum(p ** 2)
    
    def entropy(self, y):
        _, count = np.unique(y, return_counts=True)
        p = count / len(y)
        return -np.sum(p * np.log2(p))
    
    def _split(self, X, y):
        n, d = X.shape
        best_gain, best_fea, best_thres = 0, None, None
        parent = self.gini(y)
        for j in range(d):
            for t in np.unique(X[:, j]):
                left = X[:, j] <= t
                right = ~left
                gain = parent - (
                    (np.sum(left) * self.gini(y[left]) +
                    np.sum(right) * self.gini(y[right])) / n
                )
                if gain > best_gain:
                    best_gain = gain
                    best_fea = j
                    best_thres = t
        return best_fea, best_thres
    
    def _build_tree(self, X, y, depth = 0):
        if len(np.unique(y)) == 1 or depth > self.max_depth:
            vals, counts = np.unique(y, return_counts=True)
            return vals[np.argmax(counts)]
        
        fea, thres = self._split(X, y)
        if fea is None:
            vals, counts = np.unique(y, return_counts=True)
            return vals[np.argmax(counts)]
        
        mask = X[:, fea] <= thres
        return {
            "feature": fea,
            "threshold": thres,
            "left": self._build_tree(X[mask], y[mask], depth + 1),
            "right": self._build_tree(X[~mask], y[~mask], depth + 1)
        }
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def _pred(self, tree, X):
        while isinstance(tree, dict):
            tree = tree["left"] if X[tree["feature"]] <= tree["threshold"] else tree["right"]
        return tree
    
    def predict(self, X):
        return np.array([self._pred(self.tree, x) for x in X])
