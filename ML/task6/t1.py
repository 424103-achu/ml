import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import seaborn as sns

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Discretize continuous features into 10 bins
dfnew = pd.DataFrame()

for col in iris.feature_names:
    dfnew[col] = pd.cut(df[col], bins=10, labels=False)

# Entropy
def entropy(p):
    p = np.array(p)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

# Mutual Information
def mutual_information(x, y):
    n = len(x)

    p_x = np.bincount(x) / n
    p_y = np.bincount(y) / n

    joint = np.zeros((10, 10))

    for xi, yi in zip(x, y):
        joint[xi, yi] += 1

    p_xy = joint / n

    return entropy(p_x) + entropy(p_y) - entropy(p_xy.flatten())

# Calculate MI matrix
features = dfnew.columns
n = len(features)

M = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        M[i, j] = mutual_information(
            dfnew.iloc[:, i].values,
            dfnew.iloc[:, j].values
        )

# Display matrix
Mnew = pd.DataFrame(M, index=features, columns=features)
print(Mnew)

# Heatmap
sns.heatmap(Mnew, annot=True)

# Find highest MI pair
pairs = Mnew.unstack().reset_index()
pairs.columns = ['Feature_A', 'Feature_B', 'MI_Value']

pairs = pairs[pairs['Feature_A'] != pairs['Feature_B']]

highest = pairs.loc[pairs['MI_Value'].idxmax()]

print("\nHighest MI:", highest['MI_Value'])
print("Features:", highest['Feature_A'], "and", highest['Feature_B'])