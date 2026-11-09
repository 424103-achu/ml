import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

torch.manual_seed(0)

df = pd.DataFrame({
    "age":    [25, 32, 47, 51, 23, 36, 60, 29, 41, 55, 34, 48, 27, 39, 58, 31, 44, 52, 26, 37],
    "salary": [30, 45, 80, 90, 25, 50, 95, 35, 70, 85, 48, 75, 32, 60, 92, 40, 72, 88, 28, 55],
    "city":   ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A",
               "B", "C", "A", "B", "C", "A", "B", "C", "A", "B"],
    "gender": ["M", "F", "M", "F", "F", "M", "M", "F", "M", "F",
               "M", "F", "F", "M", "F", "M", "F", "M", "F", "M"],
    "plan":   ["basic", "standard", "premium", "premium", "basic", "standard", "premium", "basic", "premium", "premium",
               "standard", "premium", "basic", "standard", "premium", "basic", "premium", "premium", "basic", "standard"],
})

print(df.head())
print(df.dtypes)

df = df.dropna()

df = pd.get_dummies(df, columns=["city", "gender"], dtype=int)

encoder = LabelEncoder()
df["plan"] = encoder.fit_transform(df["plan"])
print("\nClasses:", list(encoder.classes_))

X = df.drop("plan", axis=1).values
y = df["plan"].values
print("\nFeature columns:", list(df.drop("plan", axis=1).columns))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print("\nX_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


train_loader = DataLoader(MyDataset(X_train, y_train), batch_size=4, shuffle=True)


class MyModel(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 16)
        self.fc2 = nn.Linear(16, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = MyModel(n_features=X_train.shape[1], n_classes=len(encoder.classes_))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    for features, labels in train_loader:
        output = model(features)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        print("Epoch", epoch + 1, "Loss", round(loss.item(), 4))

model.eval()
with torch.no_grad():
    pred = torch.argmax(model(X_test), dim=1)
    accuracy = (pred == y_test).float().mean().item() * 100

print("\nTest accuracy:", round(accuracy, 2), "%")
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, pred, labels=range(len(encoder.classes_))))
print(classification_report(y_test, pred, labels=range(len(encoder.classes_)),
                            target_names=encoder.classes_, zero_division=0))

# ---- How to test with new data (no labels) ----
# New data must go through the SAME preprocessing as training:
# new_person = pd.DataFrame({"age": [45], "salary": [78], "city": ["C"], "gender": ["M"]})
# new_person = pd.get_dummies(new_person, columns=["city", "gender"], dtype=int)            # same one-hot
# new_person = new_person.reindex(columns=df.drop("plan", axis=1).columns, fill_value=0)   # same columns, same order
# new_x = torch.tensor(scaler.transform(new_person.values), dtype=torch.float32)          # same scaler (transform, NOT fit)
# model.eval()
# with torch.no_grad():
#     pred = torch.argmax(model(new_x), dim=1)
# print("New person plan:", encoder.inverse_transform(pred.numpy())[0])                   # number -> class name
