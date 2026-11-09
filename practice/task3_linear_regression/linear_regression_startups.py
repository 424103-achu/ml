import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task3"))  # data files live here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Linear Regression (from ML-Lab-Practice-master/Linear Regression/closed_form.py + gradient.py)
def normal(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)


def rss(X, y, w):
    r = y - X @ w
    return r @ r


def mse(X, y, w):
    return rss(X, y, w) / len(y)


class LinearRegression:
    def fit(self, X, y):
        X = np.column_stack([np.ones(len(X)), X])     # column of 1s for the intercept
        self.w = normal(X, y)
        self.intercept, self.coef = self.w[0], self.w[1:]

    def predict(self, X):
        X = np.column_stack([np.ones(len(X)), X])
        return X @ self.w


# Load data
df = pd.read_csv("50_Startups.csv")
print(df.head())

# One-hot encode State
X = pd.get_dummies(df.drop("Profit", axis=1), columns=["State"], drop_first=True, dtype=int)
features = list(X.columns)
X = X.values.astype(float)
y = df["Profit"].values

# Train / test split (80 / 20)
idx = np.random.RandomState(0).permutation(len(X))
n_test = int(0.2 * len(X))
X_test, y_test = X[idx[:n_test]], y[idx[:n_test]]
X_train, y_train = X[idx[n_test:]], y[idx[n_test:]]

# Train
model = LinearRegression()
model.fit(X_train, y_train)
print("Intercept:", round(model.intercept, 2))
for f, c in zip(features, model.coef):
    print(f"  {f:18s}: {c:.4f}")

# Predict
y_pred = model.predict(X_test)
print(pd.DataFrame({"Actual": y_test, "Predicted": y_pred.round(2)}))

# Errors
Xb_test = np.column_stack([np.ones(len(X_test)), X_test])
n, k = len(y_test), len(features)
mae = np.mean(np.abs(y_test - y_pred))
test_mse = mse(Xb_test, y_test, model.w)
r2 = 1 - rss(Xb_test, y_test, model.w) / np.sum((y_test - y_test.mean()) ** 2)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
print("MAE :", round(mae, 2))
print("MSE :", round(test_mse, 2))
print("RMSE:", round(np.sqrt(test_mse), 2))
print("R2  :", round(r2, 4), "  Adjusted R2:", round(adj_r2, 4))

# Plot
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r-")
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("50 Startups: Actual vs Predicted")
plt.show()
