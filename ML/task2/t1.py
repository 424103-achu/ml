import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import math

# Load data
df_train = pd.read_csv("SalaryData_Train.csv")
df_test = pd.read_csv("SalaryData_Test.csv")

# Encode categorical columns consistently
for col in df_train.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])

    if col in df_test.columns:
        df_test[col] = df_test[col].map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

# Split features and target
X = df_train.drop(columns=["Salary"]).values
Y = df_train["Salary"].values

X_test = df_test.drop(columns=["Salary"]).values
y_test = df_test["Salary"].values


# Gaussian Naive Bayes training
def calculate_stats(X, y):
    classes = np.unique(y)
    stats = {}

    for c in classes:
        X_c = X[y == c]
        prior = len(X_c) / len(y)

        means = np.mean(X_c, axis=0)
        variances = np.var(X_c, axis=0) + 1e-9

        stats[c] = {
            "prior": prior,
            "features": list(zip(means, variances))
        }

    return stats


# Predict one sample
def predict_single(sample, stats):
    best_class = None
    max_log_posterior = -float("inf")

    for c, c_info in stats.items():
        log_posterior = math.log(c_info["prior"])

        for col, x in enumerate(sample):
            mean, var = c_info["features"][col]

            log_pdf = -0.5 * math.log(2 * math.pi * var) \
                      - ((x - mean) ** 2 / (2 * var))

            log_posterior += log_pdf

        if log_posterior > max_log_posterior:
            max_log_posterior = log_posterior
            best_class = c

    return best_class


# Train and predict
model_stats = calculate_stats(X, Y)
y_pred = [predict_single(sample, model_stats) for sample in X_test]

# Confusion matrix values
tp = sum(1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 1)
tn = sum(1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 0)
fp = sum(1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 1)
fn = sum(1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 0)

# Metrics
accuracy = (tp + tn) / len(y_test)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0 else 0
)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"True Negatives (TN): {tn:<6} | False Negatives (FN): {fn}")
print(f"False Positives (FP): {fp:<5} | True Positives (TP): {tp}")
