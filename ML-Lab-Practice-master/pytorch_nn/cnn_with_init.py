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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)


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
    def __init__(self, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 2 * 2, 32)
        self.fc2 = nn.Linear(32, 3)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.01)
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "elu":
            self.act = nn.ELU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "selu":
            self.act = nn.SELU()
        elif activation == "softplus":
            self.act = nn.Softplus()

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def make_optimizer(name, model):
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=0.01)
    elif name == "sgd_momentum":
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif name == "adam":
        return optim.Adam(model.parameters(), lr=0.01)
    elif name == "adamw":
        return optim.AdamW(model.parameters(), lr=0.01)
    elif name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=0.01)
    elif name == "adagrad":
        return optim.Adagrad(model.parameters(), lr=0.01)
    elif name == "adadelta":
        return optim.Adadelta(model.parameters(), lr=1.0)


def train_and_test(activation, optimizer_name):
    torch.manual_seed(0)
    model = MyCNN(activation)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = make_optimizer(optimizer_name, model)

    for epoch in range(20):
        model.train()
        for images, labels in train_loader:
            output = model(images)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(X_test), dim=1)
        accuracy = (pred == y_test).float().mean().item() * 100

    print(f"{activation:12s} {optimizer_name:14s} loss={loss.item():.4f}  test accuracy={accuracy:.2f}%")
    return pred


activations = ["relu", "leaky_relu", "sigmoid", "tanh", "elu", "gelu", "selu", "softplus"]
optimizers = ["sgd", "sgd_momentum", "adam", "adamw", "rmsprop", "adagrad", "adadelta"]

print("---- All activations (optimizer = adam) ----")
for act in activations:
    train_and_test(act, "adam")

print("\n---- All optimizers (activation = relu) ----")
for opt in optimizers:
    train_and_test("relu", opt)

print("\n---- Test results with relu + adam ----")
pred = train_and_test("relu", "adam")
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# ---- How to test with new data (no labels) ----
# 1. Make train_and_test() return the model too:  return model, pred
#    and call it as:  model, pred = train_and_test("relu", "adam")
# 2. New images must have shape (num_images, 1, 8, 8):
# new_images = torch.cat([make_images(1, 0), make_images(1, 1), make_images(1, 2)])
# model.eval()
# with torch.no_grad():
#     output = model(new_images)
#     prob = torch.softmax(output, dim=1)  # one probability per class
#     pred = torch.argmax(output, dim=1)   # 0 = horizontal, 1 = vertical, 2 = diagonal
# print("Probabilities:", prob.round(decimals=3))
# print("Predicted classes:", pred.tolist())
