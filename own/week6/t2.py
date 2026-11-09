# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")

# %%
DATA_PATH = "Wine.csv"

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
df.head()
print("Columns:", list(df.columns))

# %%
target_col = "Customer_Segment"

X = df.drop(columns=[target_col])
y = df[target_col]

print("Target column:", target_col)
print("Number of features:", X.shape[1])
print("Classes:", sorted(y.unique()))
print(y.value_counts().sort_index().rename_axis("class").to_frame("count"))

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# %%
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training means after scaling (approximately 0):")
print(np.round(X_train_scaled.mean(axis=0), 4))
print("Training standard deviations after scaling (approximately 1):")
print(np.round(X_train_scaled.std(axis=0), 4))

# %%
lda = LinearDiscriminantAnalysis(n_components=2)

X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

print("X_train_lda shape:", X_train_lda.shape)
print("X_test_lda shape :", X_test_lda.shape)
print("Explained discriminant ratios:",
      np.round(lda.explained_variance_ratio_, 4))

# %%
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_lda, y_train)

y_pred = log_reg.predict(X_test_lda)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Test Accuracy:", round(accuracy_score(y_test, y_pred), 4))

# %%
plt.figure(figsize=(9, 7))

for cls in sorted(y_test.unique()):
    mask = y_test.to_numpy() == cls
    plt.scatter(
        X_test_lda[mask, 0],
        X_test_lda[mask, 1],
        label=f"Class {cls}",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4
    )

plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("Wine Classes in 2D LDA Space — Test Data")
plt.legend(title="Wine Class")
plt.tight_layout()
plt.show()
