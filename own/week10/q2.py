import pandas as pd
import numpy as np
# ---------- Load data and build features ----------
df = pd.read_csv("AAPL.csv")                 # change to "AMD.csv" or "LNG.csv"

df["Return"]      = df["Close"].pct_change()                     # today's % change
df["Return_Prev"] = df["Return"].shift(1)                        # yesterday's % change
df["HL_Range"]    = (df["High"] - df["Low"]) / df["Close"]       # how much it swung today
df["OC_Change"]   = (df["Close"] - df["Open"]) / df["Open"]      # open-to-close move today
df["Vol_Change"]  = df["Volume"].pct_change()                    # volume change vs yesterday

# Target: 1 if TOMORROW's close is higher than today's, else 0
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

df = df.iloc[:-1]        # last day has no "tomorrow", so drop it
df = df.dropna()         # first rows have no "yesterday", so drop them

features = ["Return", "Return_Prev", "HL_Range", "OC_Change", "Vol_Change"]
X = df[features].values  # 2D numpy array: rows = days, columns = features
y = df["Target"].values  # 1D numpy array of 0s and 1s


# ---------- Task 1: Entropy and Information Gain ----------
def entropy(labels):
    p = np.bincount(labels) / len(labels)    # fraction of 0s and 1s
    p = p[p > 0]                             # log2(0) is undefined, so drop zeros
    return -np.sum(p * np.log2(p))


def info_gain(parent, left, right):
    n = len(parent)
    return entropy(parent) - (len(left) / n) * entropy(left) - (len(right) / n) * entropy(right)


# ---------- Task 2: Recursive tree builder ----------
def build_tree(X, y, depth, max_depth=4, min_samples=2):
    majority = np.bincount(y).argmax()       # 0 or 1, whichever is more common

    # Stopping rules: pure node, max depth reached, too few rows
    if len(np.unique(y)) == 1 or depth == max_depth or len(y) < min_samples:
        return majority

    # Try every distinct value of every feature as a threshold
    best_gain, best_col, best_value = 0, None, None
    for col in range(X.shape[1]):                    # col = 0, 1, 2, 3, 4
        for value in np.unique(X[:, col]):           # every distinct value in this column
            mask = X[:, col] <= value                # Left: <= value, Right: > value
            left, right = y[mask], y[~mask]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = info_gain(y, left, right)
            if gain > best_gain:
                best_gain, best_col, best_value = gain, col, value

    if best_col is None:                     # no split helps -> make a leaf
        return majority

    print("  " * depth + f"Depth {depth}: {features[best_col]} <= {best_value:.4f}  (IG={best_gain:.4f})")

    mask = X[:, best_col] <= best_value
    return {"col": best_col,
            "value": best_value,
            "left": build_tree(X[mask], y[mask], depth + 1, max_depth, min_samples),
            "right": build_tree(X[~mask], y[~mask], depth + 1, max_depth, min_samples)}


# ---------- Task 3: Predict one day ----------
def predict(node, row):
    while isinstance(node, dict):            # a leaf is just 0 or 1
        if row[node["col"]] <= node["value"]:
            node = node["left"]
        else:
            node = node["right"]
    return node


# ---------- Chronological train/test split and accuracy ----------
cut = int(len(X) * 0.67)                     # first 67% = past, last 33% = future
X_train, y_train = X[:cut], y[:cut]
X_test,  y_test  = X[cut:], y[cut:]
print("Train rows:", len(X_train), " Test rows:", len(X_test))

tree = build_tree(X_train, y_train, depth=0, max_depth=4)

predictions = np.array([predict(tree, row) for row in X_test])
correct = np.sum(predictions == y_test)
print(f"\nAccuracy: {correct}/{len(y_test)} = {correct / len(y_test) * 100:.2f}%")

# For comparison: accuracy if we just always guessed the most common direction
baseline = max(np.mean(y_test), 1 - np.mean(y_test)) * 100
print(f"Baseline (always guess the most common direction): {baseline:.2f}%")