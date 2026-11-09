import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

torch.manual_seed(0)


def make_images(n, label):
    images = torch.rand(n, 1, 8, 8) * 0.5
    for i in range(n):
        pos = torch.randint(0, 8, (1,)).item()
        if label == 0:
            images[i, 0, pos, :] = 1
        elif label == 1:
            images[i, 0, :, pos] = 1
        else:
            for k in range(8):
                images[i, 0, k, k] = 1
    return images


X = torch.cat([make_images(100, 0), make_images(100, 1), make_images(100, 2)])
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


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 2 * 2, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = MyCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        output = model(images)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 5 == 0:
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
# New images must have the same shape as training: (num_images, 1 channel, 8, 8)
# new_images = torch.cat([make_images(1, 0), make_images(1, 1), make_images(1, 2)])
# For a single image of shape (8, 8): new_images = img.reshape(1, 1, 8, 8)
# model.eval()
# with torch.no_grad():
#     output = model(new_images)
#     prob = torch.softmax(output, dim=1)  # one probability per class
#     pred = torch.argmax(output, dim=1)   # 0 = horizontal, 1 = vertical, 2 = diagonal
# print("Probabilities:", prob.round(decimals=3))
# print("Predicted classes:", pred.tolist())
