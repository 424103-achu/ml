import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task0"))  # data files live here
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split

csv_text = """CustomerID,Age,MonthlyCharges,TotalCharges,ContractType,Churn
1001,34,65.5,2227.0,One Year,No
1002,22,20.0,40.0,Month-to-month,Yes
1003,50,95.4,,Two Year,No
1004,63,110.2,6942.5,Two Year,No
1005,28,45.0,1350.0,Month-to-month,No
1006,41,,2010.3,One Year,No
1007,55,85.1,4255.0,Month-to-month,Yes
1008,19,25.0,75.0,Month-to-month,No
1009,38,70.8,,One Year,No
1010,47,100.5,5025.0,Two Year,Yes
1011,32,55.0,1100.0,Month-to-month,No
1012,25,20.1,60.3,Month-to-month,Yes
1013,58,105.3,6318.0,Two Year,No
1014,40,75.0,3000.0,One Year,No
1015,61,115.0,7015.0,Two Year,No
"""
df = pd.read_csv(StringIO(csv_text))
df.to_csv("churn.csv", index=False)
print("Original data:\n", df)

df = df.dropna()

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
df = pd.get_dummies(df, columns=["ContractType"], dtype=int)
print("\nCleaned data:\n", df)

X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nChurn train:", X_train.shape, " test:", X_test.shape)

wine = pd.read_csv("winequality-red.csv", sep=";")
wine = wine.dropna()
X = wine.drop(columns=["quality"])
y = wine["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Wine  train:", X_train.shape, " test:", X_test.shape)

try:
    air = pd.read_csv("http://www.stat.ufl.edu/~winner/data/airq402.dat",
                      sep=r"\s+", header=None)
    air = air.dropna()
    air = pd.get_dummies(air, dtype=int)
    train, test = train_test_split(air, test_size=0.2, random_state=42)
    print("Airfare train:", train.shape, " test:", test.shape)
except Exception as e:
    print("Airfare dataset could not be downloaded (offline?):", e)
