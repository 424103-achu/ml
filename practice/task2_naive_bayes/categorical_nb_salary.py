import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task2"))  # data files live here
import pandas as pd
import numpy as np


# CategoricalNB (from ML-Lab-Practice-master/NaiveBayes/categoricalNB.py)
class CategoricalNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = []
        self.priors = {}
        self.likelihoods = {}
        self.categories = {}

    def fit(self, X, y):
        self.classes = np.unique(y)

        for col in X.columns:
            self.categories[col] = np.unique(X[col])

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)

            self.likelihoods[c] = {}
            for col in X.columns:
                self.likelihoods[c][col] = {}
                V_col = len(self.categories[col])
                N_c_col = X_c[col].value_counts()
                total_count = len(X_c)

                for category in self.categories[col]:
                    numerator = N_c_col.get(category, 0) + self.alpha
                    denominator = total_count + (self.alpha * V_col)
                    self.likelihoods[c][col][category] = numerator / denominator

    def predict(self, X):
        predictions = []

        for _, x in X.iterrows():
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = 0
                for col in X.columns:
                    category = x[col]
                    if category in self.likelihoods[c][col]:
                        likelihood += np.log(self.likelihoods[c][col][category])
                    else:
                        likelihood += np.log(self.alpha / (len(self.categories[col]) + self.alpha))
                posterior = prior + likelihood
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return predictions


# Load data
train = pd.read_csv("SalaryData_Train.csv").dropna()
test = pd.read_csv("SalaryData_Test.csv").dropna()
print("Train shape:", train.shape, " Test shape:", test.shape)
print("Class distribution:\n", train["Salary"].value_counts())

# Numeric columns -> 10 quantile bins (so every column is categorical)
for col in train.select_dtypes("number").columns:
    edges = np.unique(np.quantile(train[col], np.linspace(0, 1, 11))[1:-1])
    train[col] = np.digitize(train[col], edges)
    test[col] = np.digitize(test[col], edges)

# Split features and target (CategoricalNB works on DataFrames)
X_train = train.drop("Salary", axis=1)
y_train = train["Salary"].values
X_test = test.drop("Salary", axis=1)
y_test = test["Salary"].values

# Train and predict
model = CategoricalNB(alpha=1.0)
model.fit(X_train, y_train)
y_pred = np.array(model.predict(X_test))

# Accuracy and confusion matrix
print("Accuracy:", np.mean(y_pred == y_test))
print(pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]))
