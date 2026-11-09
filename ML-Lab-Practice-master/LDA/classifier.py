import numpy as np
import pandas as pd

class LDA:
    def __init__(self):
        self.mu = []
        self.priors = []
        self.sigma_inv = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        n, d = X.shape
        self._classes = y.unique()
        sigma = np.zeros((d, d))
        for k in self._classes:
            X_k = X[y == k]
            mu = X_k.mean()
            self.mu.append(mu)
            self.priors.append(len(X_k) / n)
            sigma += (X_k - mu).T @ (X_k - mu)
        
        sigma /= (n - len(self._classes))
        self.sigma_inv = np.linalg.inv(sigma)
    
    def predict(self, X):
        scores = []
        for mu, p in zip(self.mu, self.priors):
            delta = X @ self.sigma_inv @ mu - 0.5 * mu.T @ self.sigma_inv @ mu + np.log(p)
            scores.append(delta)
        return self._classes[np.argmax(np.array(scores).T, axis=1)]


# import numpy as np
# import pandas as pd

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # ==================================
# # 1. LOAD DATASET
# # ==================================

# iris = load_iris(as_frame=True)

# df = iris.frame

# X = df.drop(columns="target")
# y = df["target"]

# print("Dataset:")
# print(df.head())

# print("\nX shape:", X.shape)
# print("y shape:", y.shape)


# # ==================================
# # 2. TRAIN TEST SPLIT
# # ==================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )


# # ==================================
# # 3. LDA IMPLEMENTATION
# # ==================================

# class LDA:

#     def __init__(self):
#         self.mu = []
#         self.priors = []
#         self.sigma_inv = []

#     def fit(self, X, y):

#         n, d = X.shape

#         self._classes = y.unique()

#         self.mu = []
#         self.priors = []

#         sigma = np.zeros((d, d))

#         for k in self._classes:

#             # Extract samples belonging to class k
#             X_k = X[y == k]

#             # Calculate class mean
#             mu = X_k.mean().values

#             self.mu.append(mu)

#             # Calculate prior probability
#             self.priors.append(len(X_k) / n)

#             # Calculate within-class scatter
#             sigma += (X_k.values - mu).T @ (X_k.values - mu)

#         # Calculate pooled covariance
#         sigma /= (n - len(self._classes))

#         # Calculate inverse covariance
#         self.sigma_inv = np.linalg.inv(sigma)


#     def predict(self, X):

#         X = np.asarray(X)

#         scores = []

#         for mu, p in zip(self.mu, self.priors):

#             delta = (
#                 X @ self.sigma_inv @ mu
#                 - 0.5 * mu.T @ self.sigma_inv @ mu
#                 + np.log(p)
#             )

#             scores.append(delta)

#         scores = np.array(scores).T

#         return self._classes[np.argmax(scores, axis=1)]


# # ==================================
# # 4. INITIALIZE MODEL
# # ==================================

# model = LDA()


# # ==================================
# # 5. TRAIN MODEL
# # ==================================

# model.fit(X_train, y_train)


# # ==================================
# # 6. PREDICT
# # ==================================

# y_pred = model.predict(X_test)


# # ==================================
# # 7. EVALUATION
# # ==================================

# print("\nActual labels:")
# print(y_test.values)

# print("\nPredicted labels:")
# print(y_pred)

# print("\nAccuracy:")
# print(accuracy_score(y_test, y_pred))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# print("\nClassification Report:")
# print(classification_report(
#     y_test,
#     y_pred,
#     target_names=iris.target_names
# ))