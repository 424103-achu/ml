import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task2"))  # data files live here
import pandas as pd
import numpy as np


# GaussianNB (from ML-Lab-Practice-master/NaiveBayes/gaussianNB.py)
class GaussianNB:
    def __init__(self):
        self.classes = []
        self.priors = {}
        self.means = {}
        self.variances = {}

    def fit(self, X, y):
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + 1e-9

    def gaussian_pdf(self, x, mean, variance):
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return exponent / denominator

    def predict(self, X):
        predictions = []
        for x in X.values:
            posteriors = []

            for c in self.classes:
                prior = np.log(self.priors[c])
                posterior = np.sum(np.log(self.gaussian_pdf(x, self.means[c], self.variances[c]) + 1e-9))
                posterior = prior + posterior
                posteriors.append(posterior)

            predictions.append(self.classes[np.argmax(posteriors)])

        return predictions


# Load data
df = pd.read_csv("Social_Network_Ads.csv")
print(df.head())
print("Class distribution:\n", df["Purchased"].value_counts())

df = df.drop(columns=["UserID"])
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

X = df.drop("Purchased", axis=1).astype(float)
y = df["Purchased"].values

# Train / test split (80 / 20)
idx = np.random.RandomState(0).permutation(len(X))
n_test = int(0.2 * len(X))
X_test, y_test = X.iloc[idx[:n_test]], y[idx[:n_test]]
X_train, y_train = X.iloc[idx[n_test:]], y[idx[n_test:]]

# Scale features
mean, std = X_train.mean(), X_train.std(ddof=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Train and predict
model = GaussianNB()
model.fit(X_train.values, y_train)
y_pred = np.array(model.predict(X_test))      # DataFrame: predict() uses X.values

# Accuracy and confusion matrix
print("Accuracy:", np.mean(y_pred == y_test))
print(pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]))
