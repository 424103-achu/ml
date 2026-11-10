import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

torch.manual_seed(0)

# 3 classes: 0 = small (1-3), 1 = medium (4.5-6.5), 2 = big (8-10)
X0 = torch.rand(50, 2) * 2 + 1
X1 = torch.rand(50, 2) * 2 + 4.5
X2 = torch.rand(50, 2) * 2 + 8
X = torch.cat([X0, X1, X2])
y = torch.tensor([0] * 50 + [1] * 50 + [2] * 50, dtype=torch.long)  # 1-D, long

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
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 3)  # 3 outputs = 3 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)  # raw scores (logits), no softmax here


model = MyModel()
loss_fn = nn.CrossEntropyLoss()  # applies softmax internally
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
    pred = torch.argmax(model(X_test), dim=1)  # class with highest score
    accuracy = (pred == y_test).float().mean().item() * 100

print("\nTest accuracy:", round(accuracy, 2), "%")
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# ---- How to test with new data (no labels) ----
# new_data = torch.tensor([[2, 3], [5, 6], [9, 9]], dtype=torch.float32)  # shape (points, 2 features)
# model.eval()
# with torch.no_grad():
#     logits = model(new_data)
#     prob = torch.softmax(logits, dim=1)  # one probability per class, each row sums to 1
#     pred = torch.argmax(prob, dim=1)     # class with highest probability
# print("Probabilities:", prob.round(decimals=3))
# print("Predicted classes:", pred.tolist())
