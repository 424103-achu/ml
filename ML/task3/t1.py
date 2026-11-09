# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np

# %%
df = pd.read_csv("kc_house_data.csv")
df.head()

# %%
df = df.drop(columns=['id', 'date', 'lat', 'long', 'zipcode', 'sqft_above'])
df = df.astype({'floors': int, 'bathrooms': int})
df.info()

# %%
train = df.sample(frac=0.8, random_state=222)
test = df.drop(train.index)
test.head()

# %%
X_train = train.drop(columns=['price'])
Y_train = train['price']

X_test = test.drop(columns=['price'])
Y_test = test['price']

Y_train.info()

# %%
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
X_tr = np.array(X_train_scaled)
y_tr = np.array(Y_train)
X_te = np.array(X_test_scaled)
y_te = np.array(Y_test)

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
plt.plot(mse_history, color='red')
plt.title("MSE Loss")
plt.xlabel("iters")
plt.ylabel("MSE")
plt.show()
