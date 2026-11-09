import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

torch.manual_seed(0)

SEQ_LEN = 20


# Each sample is a sequence of 20 numbers (1 feature per time step)
# class 0 = going up, 1 = going down, 2 = wave (up and down)
def make_sequences(n, label):
    t = torch.linspace(0, 1, SEQ_LEN)
    seqs = []
    for i in range(n):
        if label == 0:
            s = t * (torch.rand(1) + 0.5)
        elif label == 1:
            s = -t * (torch.rand(1) + 0.5)
        else:
            s = torch.sin(t * 6.28 * (torch.rand(1) + 1)) * 0.5
        s = s + torch.randn(SEQ_LEN) * 0.1  # noise
        seqs.append(s)
    return torch.stack(seqs).unsqueeze(2)  # shape (n, seq_len, 1 feature)


X = torch.cat([make_sequences(100, 0), make_sequences(100, 1), make_sequences(100, 2)])
y = torch.tensor([0] * 100 + [1] * 100 + [2] * 100)

print("X shape:", X.shape)  # (samples, time steps, features per step)
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


class MyRNN(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "rnn":
            self.rnn = nn.RNN(input_size=1, hidden_size=32, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        # out: hidden state at EVERY time step -> (batch, seq_len, 32)
        # RNN returns (out, h), LSTM returns (out, (h, c)); we only need out
        out, _ = self.rnn(x)
        last = out[:, -1, :]  # hidden state at the LAST time step -> (batch, 32)
        return self.fc(last)  # raw scores (logits) for 3 classes


def train_and_test(kind):
    torch.manual_seed(0)
    model = MyRNN(kind)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"\n========== {kind.upper()} ==========")
    for epoch in range(30):
        model.train()
        for seqs, labels in train_loader:
            output = model(seqs)
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

    print("Test accuracy:", round(accuracy, 2), "%")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    return model


rnn_model = train_and_test("rnn")
lstm_model = train_and_test("lstm")

# ---- How to test with new data (no labels) ----
# New sequences must have shape (num_sequences, 20 time steps, 1 feature)
# new_seqs = torch.cat([make_sequences(1, 0), make_sequences(1, 1), make_sequences(1, 2)])
# For one plain list of 20 numbers: new_seqs = torch.tensor(values, dtype=torch.float32).reshape(1, 20, 1)
# lstm_model.eval()
# with torch.no_grad():
#     output = lstm_model(new_seqs)
#     prob = torch.softmax(output, dim=1)  # one probability per class
#     pred = torch.argmax(output, dim=1)   # 0 = up, 1 = down, 2 = wave
# print("Probabilities:", prob.round(decimals=3))
# print("Predicted classes:", pred.tolist())
