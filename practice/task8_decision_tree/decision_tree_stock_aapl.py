import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task8"))  # data files live here
import numpy as np
import pandas as pd


# DecisionTree (from ML-Lab-Practice-master/Decision Tree/classifier.py)
# only change: criterion="gini" or "entropy" picks the impurity used in _split
class DecisionTree:
    def __init__(self, max_depth=15, criterion="gini"):
        self.max_depth = max_depth
        self.criterion = criterion
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
        impurity = self.gini if self.criterion == "gini" else self.entropy
        n, d = X.shape
        best_gain, best_fea, best_thres = 0, None, None
        parent = impurity(y)
        for j in range(d):
            for t in np.unique(X[:, j]):
                left = X[:, j] <= t
                right = ~left
                if right.sum() == 0:
                    continue
                gain = parent - (
                    (np.sum(left) * impurity(y[left]) +
                     np.sum(right) * impurity(y[right])) / n
                )
                if gain > best_gain:
                    best_gain = gain
                    best_fea = j
                    best_thres = t
        return best_fea, best_thres

    def _build_tree(self, X, y, depth=0):
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


df = pd.read_csv("AAPL.csv")
close, vol = df["Adj_Close"], df["Adj_Volume"]
feat = pd.DataFrame({
    "Return_1d": close.pct_change(),
    "Return_5d": close.pct_change(5),
    "MA_ratio": close / close.rolling(10).mean() - 1,
    "Volume_chg": vol.pct_change(),
})
feat["Target"] = (close.shift(-1) > close).astype(int)
feat = feat.iloc[:-1].replace([np.inf, -np.inf], np.nan).dropna()

names = ["Return_1d", "Return_5d", "MA_ratio", "Volume_chg"]
X = feat[names].round(4).values
y = feat["Target"].values

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = DecisionTree(max_depth=3, criterion="entropy")   # entropy = information gain
model.fit(X_train, y_train)
print("Root split:", names[model.tree["feature"]], "<=", model.tree["threshold"])

preds = model.predict(X_test)
print(f"Directional accuracy: {np.mean(preds == y_test) * 100:.2f}%")
print(f"Baseline (always 'up'): {np.mean(y_test == 1) * 100:.2f}%")
