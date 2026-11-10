import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task9"))  # data files live here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load XOR data (columns X, Y = inputs, Z = target)
data = pd.read_csv("Xor_Dataset.csv")
print("Samples:", len(data))
print(data.head())

X = data[["X", "Y"]].values.astype(np.float32)
y = data[["Z"]].values.astype(np.float32)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)
print("Train:", len(X_train), " Test:", len(X_test))


# Dataset class
class XORDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = XORDataset(X_train, y_train)
test_ds = XORDataset(X_test, y_test)


# 2-4-1 network with sigmoid activations, weights init N(0, 1)
class XORNet(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.hidden = nn.Linear(2, hidden)
        self.output = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.normal_(self.hidden.weight, 0.0, 1.0)
        nn.init.zeros_(self.hidden.bias)
        nn.init.normal_(self.output.weight, 0.0, 1.0)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        a1 = self.sigmoid(self.hidden(x))     # hidden layer
        return self.sigmoid(self.output(a1))  # output layer


def train_model(lr=0.5, epochs=100, batch_size=32, seed=1, verbose=True):
    torch.manual_seed(seed)
    model = XORNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)   # w = w - lr * dL/dw
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        losses.append(total / len(train_ds))
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:3d} | MSE = {losses[-1]:.6f}")
    return model, losses


# Train with lr = 0.5
print("\n=== Training (lr = 0.5, 100 epochs, batch 32) ===")
model, losses = train_model(lr=0.5)

# Loss plot
plt.figure(figsize=(7, 4))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("XOR training loss (PyTorch, lr = 0.5)")
plt.savefig("loss_plot_pytorch.png")

# Evaluate on the test set
print("\n=== Test-set evaluation ===")
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
model.eval()
probs, targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        probs.append(model(xb))
        targets.append(yb)
probs = torch.cat(probs).numpy().ravel()
y_true = torch.cat(targets).numpy().ravel().astype(int)
y_pred = (probs >= 0.5).astype(int)
print("Test MSE:", np.mean((probs - y_true) ** 2))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

# Network output for the 4 XOR patterns
print("=== Outputs for the 4 XOR patterns ===")
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
with torch.no_grad():
    out = model(torch.tensor(inputs, dtype=torch.float32)).numpy().ravel()
for (a, b), o in zip(inputs, out):
    print(f"{a} XOR {b} -> {o:.4f} -> class {int(o >= 0.5)}  (expected {a ^ b})")

# Learning-rate stress test
print("\n=== Learning-rate stress test ===")
plt.figure(figsize=(7, 4))
for lr in [0.05, 0.1, 0.5, 1.0, 5.0]:
    _, l = train_model(lr=lr, verbose=False)
    below = next((i for i, v in enumerate(l) if v < 0.01), None)
    print(f"lr={lr:<5} final MSE={l[-1]:.6f}  epochs to MSE<0.01: {below if below is not None else 'never'}")
    plt.plot(l, label=f"lr={lr}")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.yscale("log")
plt.title("Learning-rate comparison (PyTorch)")
plt.legend()
plt.savefig("lr_comparison_pytorch.png")

plt.show()
