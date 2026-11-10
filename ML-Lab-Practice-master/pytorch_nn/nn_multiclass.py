import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

torch.manual_seed(0)

X0 = torch.randn(100, 4) + 0
X1 = torch.randn(100, 4) + 2
X2 = torch.randn(100, 4) + 4
X = torch.cat([X0, X1, X2])

y = torch.tensor([0] * 100 + [1] * 100 + [2] * 100)

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

print("Train:", X_train.shape, " Test:", X_test.shape)


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


train_loader = DataLoader(MyDataset(X_train, y_train), batch_size=16, shuffle=True)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


model = MyModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    model.train()
    for features, labels in train_loader:
        output = model(features)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print("Epoch", epoch + 1, "Loss", round(loss.item(), 4))

model.eval()
with torch.no_grad():
    pred = torch.argmax(model(X_test), dim=1)
    accuracy = (pred == y_test).float().mean().item() * 100

print("\nTest accuracy:", round(accuracy, 2), "%")
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# ---- How to test with new data (no labels) ----
# new_data = torch.tensor([[0.2, -0.5, 0.1, 0.3],
#                          [2.1, 1.8, 2.3, 1.9],
#                          [4.2, 3.9, 4.1, 3.7]])  # shape (points, 4 features)
# model.eval()
# with torch.no_grad():
#     output = model(new_data)
#     prob = torch.softmax(output, dim=1)  # one probability per class
#     pred = torch.argmax(output, dim=1)   # class with highest score
# print("Probabilities:", prob.round(decimals=3))
# print("Predicted classes:", pred.tolist())
