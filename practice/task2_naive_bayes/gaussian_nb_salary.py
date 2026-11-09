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
train = pd.read_csv("SalaryData_Train.csv")
test = pd.read_csv("SalaryData_Test.csv")

# Encode categorical columns to numbers (same codes for train and test)
for col in train.columns:
    if col == "Salary" or pd.api.types.is_numeric_dtype(train[col]):
        continue
    cats = sorted(train[col].unique())
    train[col] = pd.Categorical(train[col], categories=cats).codes
    test[col] = pd.Categorical(test[col], categories=cats).codes

# Split features and target
X_train = train.drop("Salary", axis=1).values
y_train = train["Salary"].values
X_test = test.drop("Salary", axis=1)          # DataFrame: predict() uses X.values
y_test = test["Salary"].values

# Train and predict
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = np.array(model.predict(X_test))

# Accuracy and confusion matrix
print("Accuracy:", np.mean(y_pred == y_test))
print(pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]))
