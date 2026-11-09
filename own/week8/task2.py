
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict_probabilities(X, theta):
    return sigmoid(np.dot(X, theta))

def predict_classes(X, theta, threshold=0.5):
    probs = predict_probabilities(X, theta)
    return (probs >= threshold).astype(int)

def fit_gradient_descent(X, y, lr=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    
    for iteration in range(iterations):
        h = predict_probabilities(X, theta)
        gradient = np.dot(X.T, (h - y)) / m
        theta -= lr * gradient
        cost = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
        cost_history.append(cost)
        
    return theta, cost_history

def fit_newtons_method(X, y, iterations=10):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    
    for iteration in range(iterations):
        h = predict_probabilities(X, theta)
        gradient = np.dot(X.T, (h - y)) / m
        R = np.diag(h * (1 - h))
        Hessian = np.dot(X.T, np.dot(R, X)) / m

        Hessian += np.eye(n) * 1e-6
        
        theta -= np.dot(np.linalg.inv(Hessian), gradient)
        
        cost = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
        cost_history.append(cost)
        
    return theta, cost_history

train_path = "ds1_train.csv"
valid_path = "ds1_valid.csv"
test_path  = "ds1_test.csv"

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)
test_df  = pd.read_csv(test_path)

print("--- Objective 2: Separate Dataset Subsets Loaded ---")
print(f"Train Shape: {train_df.shape} | Valid Shape: {valid_df.shape} | Test Shape: {test_df.shape}\n")

X_train_raw = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_valid_raw = valid_df.iloc[:, :-1].values
y_valid = valid_df.iloc[:, -1].values

X_test_raw = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_valid_scaled = scaler.transform(X_valid_raw)
X_test_scaled  = scaler.transform(X_test_raw)

X_train = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
X_valid = np.hstack((np.ones((X_valid_scaled.shape[0], 1)), X_valid_scaled))
X_test  = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))

print("Scanning learning rates using the Validation Dataset...")

candidate_lrs = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

perfect_lr = None
best_val_accuracy = -1

for lr in candidate_lrs:
    temp_theta, _ = fit_gradient_descent(X_train, y_train, lr=lr, iterations=500)
    y_val_pred = predict_classes(X_valid, temp_theta)
    val_acc = accuracy_score(y_valid, y_val_pred)
    
    print(f" Evaluated lr = {lr:<6} | Validation Accuracy: {val_acc * 100:.2f}%")
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        perfect_lr = lr

print(f"\n Perfect Learning Rate found: lr = {perfect_lr} (Validation Accuracy: {best_val_accuracy * 100:.2f}%)")

print(f"Training final Gradient Descent model with perfect lr = {perfect_lr}...")
theta_gd, cost_gd = fit_gradient_descent(X_train, y_train, lr=perfect_lr, iterations=1500)

print("Training Newton's Method model...")
theta_nm, cost_nm = fit_newtons_method(X_train, y_train, iterations=15)
y_pred_gd = predict_classes(X_test, theta_gd)
y_pred_nm = predict_classes(X_test, theta_nm)


print(f"\n--- [METHOD 1] GRADIENT DESCENT ACCURACY (lr={perfect_lr}): {accuracy_score(y_test, y_pred_gd)*100:.2f}% ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gd))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gd))

print(f"\n--- [METHOD 2] NEWTON'S METHOD ACCURACY: {accuracy_score(y_test, y_pred_nm)*100:.2f}% ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nm))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nm))





