import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

torch.manual_seed(0)

# 2 classes: 0 = small points (1-5), 1 = big points (6-10)
X0 = torch.rand(50, 2) * 4 + 1
X1 = torch.rand(50, 2) * 4 + 6
X = torch.cat([X0, X1])
y = torch.tensor([[0.0]] * 50 + [[1.0]] * 50)  # float, shape (N, 1) for BCELoss

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y.squeeze())

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
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x


model = MyModel()
loss_fn = nn.BCELoss()
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
    prob = model(X_test)
    pred = (prob >= 0.5).float()
    accuracy = (pred == y_test).float().mean().item() * 100

print("\nTest accuracy:", round(accuracy, 2), "%")
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test.squeeze(), pred.squeeze()))
print(classification_report(y_test.squeeze(), pred.squeeze()))

# ---- How to test with new data (no labels) ----
# new_data = torch.tensor([[2, 3], [8, 9]], dtype=torch.float32)  # shape (points, 2 features)
# model.eval()
# with torch.no_grad():
#     prob = model(new_data)              # P(class 1) for each point, already sigmoid
#     pred = (prob >= 0.5).int()          # >= 0.5 -> class 1, else class 0
# print("Probabilities:", prob.squeeze().tolist())
# print("Predictions:", pred.squeeze().tolist())
