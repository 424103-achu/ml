# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# %%
df = pd.read_csv('heart-disease.csv')
df.head()

# %%
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=9999)

# %%
scaler = StandardScaler()
X_trainnew = scaler.fit_transform(X_train)
X_testnew = scaler.transform(X_test)

# %%
error_rates = []
for k in range(1, 22):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_trainnew, y_train)
    preds = model.predict(X_testnew)
    error_rate = np.mean(preds != y_test)
    error_rates.append((k, error_rate))
    print(f"K={k} Error Rate: {error_rate:.2f}")

best_k = min(error_rates, key=lambda x: x[1])[0]
best_k

# %%
k = best_k
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_trainnew, y_train)

y_pred = knn.predict(X_testnew)
print(f"Accuracy: {accuracy_score(y_test,y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, zero_division=0))
