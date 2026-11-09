import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

torch.manual_seed(0)

X = torch.rand(200, 3) * 10
y = 3 * X[:, 0] + 2 * X[:, 1] - 1 * X[:, 2] + 5 + torch.randn(200)
y = y.unsqueeze(1)

print("X shape:", X.shape, X.dtype)
print("y shape:", y.shape, y.dtype)

# no stratify for regression (y is continuous, not classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        x = self.linear(x)
        return x


model = LinearRegression()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(200):
    model.train()
    for features, targets in train_loader:
        output = model(features)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 40 == 0:
        print("Epoch", epoch + 1, "Loss", round(loss.item(), 4))

print("\nLearned weights:", [round(w, 2) for w in model.linear.weight.data[0].tolist()])
print("Learned bias:", round(model.linear.bias.item(), 2))
print("True weights:   [3, 2, -1]   True bias: 5")

# regression has no accuracy / confusion matrix -> use MSE and R2 on test data
model.eval()
with torch.no_grad():
    train_mse = loss_fn(model(X_train), y_train).item()
    pred = model(X_test)
    test_mse = loss_fn(pred, y_test).item()

print("\nTrain MSE:", round(train_mse, 4))
print("Test MSE:", round(test_mse, 4))
print("Test R2:", round(r2_score(y_test, pred), 4))

# ---- How to test with new data (no labels) ----
# new_data = torch.tensor([[1.0, 2.0, 3.0],
#                          [5.0, 5.0, 5.0]])  # shape (points, 3 features)
# model.eval()
# with torch.no_grad():
#     pred = model(new_data)               # predicted numbers, no softmax/argmax in regression
# print("Predictions:", [round(p, 2) for p in pred.squeeze().tolist()])
# True values for these points: 3*x1 + 2*x2 - x3 + 5  ->  [9, 25]
