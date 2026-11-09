# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv('50_Startups.csv')
df = pd.get_dummies(df, columns=['State'])
df = df.astype(
    {'State_California': int, 'State_Florida': int, 'State_New York': int})
df.info()

# %%
X = df[['R&D Spend', 'Administration', 'Marketing Spend',
        'State_California', 'State_Florida', 'State_New York']]
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# %%
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
X_tr = np.array(X_train_scaled)
y_tr = np.array(y_train)
X_te = np.array(X_test_scaled)
y_te = np.array(y_test)

num_samples, num_features = X_tr.shape

weights = np.zeros(num_features)
learning_rate = 0.05
bias = 0.0
iterations = 1000
mse_history = []

for epoch in range(iterations):

    y_pred = (X_tr @ weights) + bias
    error = y_pred - y_tr
    mse = np.mean(error ** 2)
    mse_history.append(mse)

    dw = (2 / num_samples) * (X_tr.T @ error)
    db = (2 / num_samples) * np.sum(error)

    weights -= learning_rate * dw
    bias -= learning_rate * db

y_pred_test = (X_te @ weights) + bias
test_mse = np.mean((y_pred_test - y_te) ** 2)
test_rmse = np.sqrt(test_mse)

print(f"Weights:{np.round(weights, 2)}")
print(f"Bias:{bias:,.2f}")
print(f"MSE:{test_mse:,.2f}")
print(f"RMSE:{test_rmse:,.2f}")


# %%
plt.plot(mse_history, color='RED')
plt.title("MSE Loss")
plt.xlabel("iters")
plt.ylabel("MSE")
plt.show()
