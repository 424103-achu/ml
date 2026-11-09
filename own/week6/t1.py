# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)


# %%
data = np.loadtxt("inputDataset.txt")
x = data[:, :-1]
y = data[:, -1]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.30,
    random_state=50
)

# %%
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

lda.fit(X_train, y_train)
qda.fit(X_train, y_train)

y_pred_lda = lda.predict(X_test)
y_pred_qda = qda.predict(X_test)

# %%
print(classification_report(y_test, y_pred_lda))

# %%
print(classification_report(y_test, y_pred_qda))

# %%
cm_lda = confusion_matrix(y_test, y_pred_lda)

print(cm_lda)

# %%
cm_qda = confusion_matrix(y_test, y_pred_qda)

print(cm_qda)

# %%
results = pd.DataFrame({
    "Model": ["LDA", "QDA"],

    "Accuracy": [
        accuracy_score(y_test, y_pred_lda),
        accuracy_score(y_test, y_pred_qda)
    ],

    "Precision": [
        precision_score(y_test, y_pred_lda),
        precision_score(y_test, y_pred_qda)
    ],

    "Recall (Sensitivity)": [
        recall_score(y_test, y_pred_lda),
        recall_score(y_test, y_pred_qda)
    ]
})

print(results)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(
    cm_lda,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[0],
    xticklabels=["No CHD", "CHD"],
    yticklabels=["No CHD", "CHD"]
)

axes[0].set_title("LDA Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")


sns.heatmap(
    cm_qda,
    annot=True,
    fmt="d",
    cmap="Oranges",
    ax=axes[1],
    xticklabels=["No CHD", "CHD"],
    yticklabels=["No CHD", "CHD"]
)

axes[1].set_title("QDA Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# %%
print("LDA False Negatives:", cm_lda[1, 0])
print("QDA False Negatives:", cm_qda[1, 0])

# %%
