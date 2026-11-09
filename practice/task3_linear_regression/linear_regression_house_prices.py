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


# Load data and split 80 / 20
data = pd.read_csv("kc_house_data.csv")
train = data.sample(frac=0.8, random_state=0)
test = data.drop(train.index)
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

features = ["sqft_living", "bedrooms", "bathrooms", "grade", "view", "waterfront", "yr_built"]
X_train = train[features].values.astype(float)
y_train = train["price"].values
X_test = test[features].values.astype(float)
y_test = test["price"].values

# Train
model = LinearRegression()
model.fit(X_train, y_train)
print("Intercept:", round(model.intercept, 2))
for f, c in zip(features, model.coef):
    print(f"  {f:12s}: {c:,.2f}")

# Predict
y_pred = model.predict(X_test)

# Errors
Xb_test = np.column_stack([np.ones(len(X_test)), X_test])
test_mse = mse(Xb_test, y_test, model.w)
r2 = 1 - rss(Xb_test, y_test, model.w) / np.sum((y_test - y_test.mean()) ** 2)
print("RSS :", round(rss(Xb_test, y_test, model.w), 2))
print("MSE :", round(test_mse, 2))
print("RMSE:", round(np.sqrt(test_mse), 2))
print("R2  :", round(r2, 4))

# Plot
plt.scatter(y_test, y_pred, s=5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r-")
plt.xlabel("Actual price"); plt.ylabel("Predicted price")
plt.title("Actual vs Predicted")
plt.show()
