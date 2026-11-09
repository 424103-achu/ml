import numpy as np
import pandas as pd


class LDA:
    def __init__(self):
        self.s_w = []
        self.s_b = []
        self.eigen_vec = []
        self.eigen_val = []
        self.eigen_order = []
        self.mu = []

    def fit(self, X, y):
        n, d = X.shape
        self._classes = np.unique(y)

        self.mu = np.mean(X, axis=0)
        self.s_w = np.zeros((d, d))
        self.s_b = np.zeros((d, d))
        for k in self._classes:
            X_c = X[y == k]
            mu_k = np.mean(X_c, axis=0)
            X_cen = X_c - mu_k
            self.s_w += X_cen.T @ X_cen
            diff = (mu_k - self.mu).reshape(-1, 1)
            self.s_b += len(X_c) * (diff @ diff.T)

        self.eigen_val, self.eigen_vec = np.linalg.eig(
            np.linalg.pinv(self.s_w) @ self.s_b)
        self.eigen_order = np.argsort(self.eigen_val)[::-1]

    def transform(self, X, k):
        return (X - self.mu) @ self.eigen_vec[:, self.eigen_order[:k]]


# import numpy as np
# import pandas as pd

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix


# # ==================================
# # 1. LOAD DATASET
# # ==================================

# iris = load_iris(as_frame=True)

# X = iris.data
# y = iris.target

# print("Original X shape:", X.shape)


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
#         self.s_w = []
#         self.s_b = []
#         self.eigen_vec = []
#         self.eigen_val = []
#         self.eigen_order = []
#         self.mu = []

#     def fit(self, X, y):

#         X = np.asarray(X)
#         y = np.asarray(y)

#         n, d = X.shape

#         self._classes = np.unique(y)

#         self.mu = np.mean(X, axis=0)

#         self.s_w = np.zeros((d, d))
#         self.s_b = np.zeros((d, d))

#         for k in self._classes:

#             X_c = X[y == k]

#             mu_k = np.mean(X_c, axis=0)

#             X_cen = X_c - mu_k

#             self.s_w += X_cen.T @ X_cen

#             diff = (mu_k - self.mu).reshape(-1, 1)

#             self.s_b += len(X_c) * (diff @ diff.T)

#         self.eigen_val, self.eigen_vec = np.linalg.eig(
#             np.linalg.inv(self.s_w) @ self.s_b
#         )

#         self.eigen_order = np.argsort(self.eigen_val)[::-1]

#     def transform(self, X, k):

#         X = np.asarray(X)

#         return (
#             (X - self.mu)
#             @ self.eigen_vec[:, self.eigen_order[:k]]
#         )


# # ==================================
# # 4. INITIALIZE LDA
# # ==================================

# lda = LDA()


# # ==================================
# # 5. FIT ON TRAINING DATA
# # ==================================

# lda.fit(X_train, y_train)


# # ==================================
# # 6. DIMENSIONALITY REDUCTION
# # ==================================

# X_train_new = lda.transform(X_train, k=2)

# X_test_new = lda.transform(X_test, k=2)


# print("\nAfter LDA:")

# print("Training shape:", X_train_new.shape)
# print("Testing shape:", X_test_new.shape)


# # ==================================
# # 7. TRAIN CLASSIFIER
# # ==================================

# model = LogisticRegression()

# model.fit(X_train_new, y_train)


# # ==================================
# # 8. PREDICTION
# # ==================================

# y_pred = model.predict(X_test_new)


# # ==================================
# # 9. EVALUATION
# # ==================================

# print("\nActual labels:")
# print(y_test.values)

# print("\nPredicted labels:")
# print(y_pred)

# print("\nAccuracy:")
# print(accuracy_score(y_test, y_pred))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
