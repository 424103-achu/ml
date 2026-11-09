# %%
import numpy as np
import pandas as pd

df_train = pd.read_csv("ds1_train.csv")
df_valid = pd.read_csv("ds1_valid.csv")
df_test = pd.read_csv("ds1_test.csv")

X_tr = df_train[["x_1", "x_2"]].values
y_tr = df_train["y"].values.reshape(-1, 1)
X_va = df_valid[["x_1", "x_2"]].values
y_va = df_valid["y"].values.reshape(-1, 1)
X_te = df_test[["x_1", "x_2"]].values
y_te = df_test["y"].values.reshape(-1, 1)

X_tr_inter = np.hstack((np.ones((X_tr.shape[0], 1)), X_tr))
X_va_inter = np.hstack((np.ones((X_va.shape[0], 1)), X_va))
X_te_inter = np.hstack((np.ones((X_te.shape[0], 1)), X_te))

m, n = X_tr_inter.shape


# %%
learn = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
best_lr = learn[0]
bestval = -1.0
theta_gd = np.zeros((n, 1))

for lr in learn:
    temp_theta = np.zeros((n, 1))
    for itr in range(10000):
        z = np.dot(X_tr_inter, temp_theta)
        h = 1 / (1 + np.exp(-z))
        gradient = np.dot(X_tr_inter.T, (h - y_tr)) / m
        temp_theta -= lr * gradient

    z_val = np.dot(X_va_inter, temp_theta)
    h_val = 1 / (1 + np.exp(-z_val))
    p_val = (h_val >= 0.5).astype(int)
    val_acc = np.mean(p_val == y_va)

    if val_acc > bestval:
        bestval = val_acc
        best_lr = lr
        theta_gd = temp_theta.copy()

# %%
theta_nm = np.zeros((n, 1))
epochs_nm = 20

for itr in range(epochs_nm):
    z = np.dot(X_tr_inter, theta_nm)
    h = 1 / (1 + np.exp(-z))
    gradient = np.dot(X_tr_inter.T, (h - y_tr)) / m
    diag = (h * (1 - h)).flatten()
    R = np.diag(diag)
    Hessian = np.dot(X_tr_inter.T, np.dot(R, X_tr_inter)) / m
    theta_nm -= np.dot(np.linalg.inv(Hessian), gradient)

# %%
z_gd_test = np.dot(X_te_inter, theta_gd)
h_gd_test = 1 / (1 + np.exp(-z_gd_test))
p_gd_test = (h_gd_test >= 0.5).astype(int)
acc_gd = np.mean(p_gd_test == y_te)

z_nm_test = np.dot(X_te_inter, theta_nm)
h_nm_test = 1 / (1 + np.exp(-z_nm_test))
p_nm_test = (h_nm_test >= 0.5).astype(int)
acc_nm = np.mean(p_nm_test == y_te)

print("Best lr:", best_lr)
print("GD Validation Accuracy:", bestval)
print("GD Test Accuracy:", acc_gd)
print("GD Weights:", theta_gd.flatten())

print("Newton Accuracy:", acc_nm)
print("Newton Weights:", theta_nm.flatten())
