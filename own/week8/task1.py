
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def predict_single_point(x_test_point, X_train, y_train, k):
    distances = []
    
    for i, x_train_point in enumerate(X_train):
        dist = euclidean_distance(x_test_point, x_train_point)
        distances.append((dist, y_train[i]))
        
    distances.sort(key=lambda item: item)
    
    k_nearest_labels = [label for (_, label) in distances[:k]]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

def knn_predict(X_test, X_train, y_train, k):
    return [predict_single_point(x, X_train, y_train, k) for x in X_test]



df = pd.read_csv("heart-disease.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



baseline_k = 5

y_pred = knn_predict(X_test_scaled, X_train_scaled, y_train, k=baseline_k)

print("\nClassification Report Summary:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Overall Model Accuracy on Test Data: {accuracy:.2f}%")


error_rates = []
k_range = range(1, 22,2) 

for current_k in k_range:
    scan_preds = knn_predict(X_test_scaled, X_train_scaled, y_train, k=current_k)
    
    mean_error = np.mean(np.array(scan_preds) != np.array(y_test))
    error_rates.append(mean_error)
    print(f"Evaluated K = {current_k:2d} | Test Error Rate: {mean_error:.4f}")

plt.figure(figsize=(9, 5))
plt.plot(k_range, error_rates, color='crimson', linestyle='--', marker='s', 
         markerfacecolor='black', markersize=8)
plt.title('Error Rate vs. K (Evaluated Using Separate Test File)')
plt.xlabel('Value of K (Neighbors Count)')
plt.ylabel('Mean Test Error Rate')
plt.grid(True, alpha=0.3)
plt.show()

optimal_k = list(k_range)[np.argmin(error_rates)]
print(f"\nOptimal value determined for classification: K = {optimal_k}")



# optimised
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import Counter
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score

# # Highly optimized, fully vectorized prediction function (No row loops)
# def knn_predict(X_test, X_train, y_train, k):
#     predictions = []
#     for x_test_point in X_test:
#         # Subtracts x_test_point from ALL rows of X_train simultaneously 
#         distances = np.sqrt(np.sum((X_train - x_test_point) ** 2, axis=1))
#         # Pair with labels, sort, slice top K, and extract the majority vote
#         nearest_labels = [lbl for _, lbl in sorted(zip(distances, y_train))[:k]]
#         predictions.append(Counter(nearest_labels).most_common(1)[0][0])
#     return np.array(predictions)

# # Load data, separate features/labels, and split
# df = pd.read_csv("heart-disease.csv")
# X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# # Scale parameters
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Baseline Evaluation (K=5)
# y_pred = knn_predict(X_test_scaled, X_train_scaled, y_train, k=5)
# print(f"\nClassification Report Summary:\n{classification_report(y_test, y_pred)}")
# print(f"Overall Model Accuracy on Test Data: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# # Scan K values to find optimal error rate
# k_range = range(1, 22, 2)
# error_rates = [np.mean(knn_predict(X_test_scaled, X_train_scaled, y_train, k=k) != y_test) for k in k_range]

# for k, err in zip(k_range, error_rates):
#     print(f"Evaluated K = {k:2d} | Test Error Rate: {err:.4f}")

# # Plot Error Curve
# plt.figure(figsize=(9, 5))
# plt.plot(k_range, error_rates, color='crimson', linestyle='--', marker='s', markerfacecolor='black', markersize=8)
# plt.title('Error Rate vs. K')
# plt.xlabel('Value of K')
# plt.ylabel('Mean Test Error Rate')
# plt.grid(True, alpha=0.3)
# plt.show()

# print(f"\nOptimal value determined for classification: K = {list(k_range)[np.argmin(error_rates)]}")
