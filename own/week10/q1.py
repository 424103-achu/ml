import pandas as pd
import numpy as np

# ---------- Load data ----------
df = pd.read_csv("drug200.csv")
features = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]


# ---------- Task 1: Entropy and Information Gain ----------
def entropy(labels):
    p = labels.value_counts() / len(labels)
    return -np.sum(p * np.log2(p))


def info_gain(parent, left, right):
    n = len(parent)
    return entropy(parent) - (len(left) / n) * entropy(left) - (len(right) / n) * entropy(right)


# ---------- Ask a question to every row at once ----------
def ask(rows, col, value):
    if isinstance(value, str):            # text column -> "== category?"
        return rows[col] == value
    else:                                 # numeric column -> "<= threshold?"
        return rows[col] <= value


# ---------- Task 2: Recursive tree builder ----------
def build_tree(rows, depth, max_depth=4, min_samples=2):
    labels = rows["Drug"]
    majority = labels.mode()[0]

    # Stopping rules: pure node, max depth reached, too few rows
    if labels.nunique() == 1 or depth == max_depth or len(rows) < min_samples:
        return majority

    # Try every question on every column, keep the one with the highest gain
    best_gain, best_col, best_value = 0, None, None
    for col in features:
        values = np.unique(rows[col])                 # sorted, no duplicates
        if pd.api.types.is_numeric_dtype(rows[col]):  # numeric column -> midpoints
            candidates = (values[:-1] + values[1:]) / 2
        else:                                         # text column -> categories
            candidates = values

        for value in candidates:
            mask = ask(rows, col, value)
            left, right = labels[mask], labels[~mask]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = info_gain(labels, left, right)
            if gain > best_gain:
                best_gain, best_col, best_value = gain, col, value

    if best_col is None:                  # no question helps -> make a leaf
        return majority

    print("  " * depth + f"Depth {depth}: split on {best_col} at {best_value}  (IG={best_gain:.3f})")

    mask = ask(rows, best_col, best_value)
    return {"col": best_col,
            "value": best_value,
            "left": build_tree(rows[mask], depth + 1, max_depth, min_samples),
            "right": build_tree(rows[~mask], depth + 1, max_depth, min_samples)}


# ---------- Task 3: Predict one patient ----------
def predict(node, patient):
    while isinstance(node, dict):         # a leaf is just a drug name (string)
        if isinstance(node["value"], str):
            go_left = patient[node["col"]] == node["value"]
        else:
            go_left = patient[node["col"]] <= node["value"]
        node = node["left"] if go_left else node["right"]
    return node


# ---------- Train / test split and accuracy ----------
cut = int(len(df) * 0.67)
train = df.iloc[:cut]
test = df.iloc[cut:]
print("Train rows:", len(train), " Test rows:", len(test))

tree = build_tree(train, depth=0, max_depth=4)

predictions = np.array([predict(tree, row) for _, row in test.iterrows()])
correct = np.sum(predictions == test["Drug"].values)
print(f"Accuracy: {correct}/{len(test)} = {correct / len(test) * 100:.2f}%")