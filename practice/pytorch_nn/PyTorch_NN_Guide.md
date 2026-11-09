# Neural Networks in PyTorch: Step by Step

**Task used throughout:** binary classification. Each student has 2 features (`hours_studied`, `hours_slept`), and we predict **pass (1)** or **fail (0)**.

The full runnable script is [nn_pytorch.py](nn_pytorch.py) (`python nn_pytorch.py`).

---

## 1. Imports

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
```

| Import | Comes from | What it gives you |
|---|---|---|
| `torch` | core PyTorch package | Tensors (`torch.tensor`, `torch.rand`), math ops, `torch.no_grad()`, `torch.manual_seed()` |
| `torch.nn as nn` | `torch.nn` submodule | Building blocks: `nn.Module` (base class for models), `nn.Linear`, `nn.ReLU`, `nn.Sigmoid`, and loss functions (`nn.BCELoss`, `nn.MSELoss`, …) |
| `torch.optim as optim` | `torch.optim` submodule | Optimizers that update weights: `optim.SGD`, `optim.Adam`, … |
| `Dataset` | `torch.utils.data` | Abstract base class. You subclass it to describe *how to get one sample* |
| `DataLoader` | `torch.utils.data` | Wraps a `Dataset` and produces *batches*, with optional shuffling |

`nn` and `optim` are short aliases, so you write `nn.Linear` instead of `torch.nn.Linear`.

---

## 2. Sample Dataset

```python
torch.manual_seed(42)

N = 200
X = torch.rand(N, 2) * 10                                  # (200, 2)
y = ((X[:, 0] + 0.5 * X[:, 1]) > 8).float().unsqueeze(1)   # (200, 1)
X = X / 10.0                                               # scale to [0, 1]
```

- **X (features):** `torch.rand(200, 2)` gives 200 rows × 2 columns of random values in [0, 1). Multiplying by 10 turns them into hours (0 to 10).
- **y (labels):** a student passes if `studied + 0.5*slept > 8`. The comparison returns a `bool` tensor, and `.float()` turns it into `0.0` / `1.0`.
- **`.unsqueeze(1)`** changes shape `(200,)` into `(200, 1)`. The model outputs shape `(batch, 1)`, and the loss needs the labels in the same shape.
- **Scaling:** dividing by 10 keeps inputs small (0 to 1), which makes training faster and more stable.

**Data types (dtype):**

| dtype | Used for |
|---|---|
| `torch.float32` | Features, and labels for `BCELoss`, `BCEWithLogitsLoss` and `MSELoss`. NN weights are float32, so inputs must be too |
| `torch.long` (int64) | Class indices for `CrossEntropyLoss` (multiclass), e.g. `tensor([0, 2, 1])` |

If your data starts as a NumPy array or list, convert it like this:
```python
X = torch.tensor(numpy_X, dtype=torch.float32)
y = torch.tensor(numpy_y, dtype=torch.float32).unsqueeze(1)
```

**Dimensions:** `X.shape = (num_samples, num_features) = (200, 2)` and `y.shape = (200, 1)`.

---

## 3. Dataset Class

```python
class StudentDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
```

| Method | Called when | Job |
|---|---|---|
| `__init__` | `StudentDataset(X, y)` | Store the data (or load files, apply preprocessing) |
| `__len__` | `len(dataset)` | Return the total number of samples. `DataLoader` uses it to know how many indices exist |
| `__getitem__` | `dataset[i]` | Return **one** sample `(features, label)` at index `i` |

**Usage:**
```python
dataset = StudentDataset(X, y)
len(dataset)          # 200
f0, l0 = dataset[0]   # f0: tensor([0.8823, 0.9150]) shape (2,)
                      # l0: tensor([1.])             shape (1,)
```
A single sample has no batch dimension. The DataLoader adds it.

---

## 4. DataLoader

```python
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for features, labels in loader:
    print(features.shape, labels.shape)   # torch.Size([16, 2]) torch.Size([16, 1])
```

- **`batch_size=16`:** each iteration gives 16 samples stacked together. The model updates its weights once per batch.
- **`shuffle=True`:** the order is randomised every epoch, so the model doesn't learn the sample order. Use `shuffle=False` for test and validation data.
- **Iterating:** the DataLoader calls `__getitem__` 16 times and stacks the results:
  - features: `(16, 2)`, which is `(batch_size, num_features)`
  - labels: `(16, 1)`
- **Batches per epoch** = `ceil(200 / 16)` = **13** (`len(loader)`). The last batch has only 200 − 12×16 = 8 samples. Pass `drop_last=True` to skip it.

---

## 5. Neural Network Model Class

```python
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)    # 2 -> 8
        self.fc2 = nn.Linear(hidden1, hidden2)       # 8 -> 4
        self.fc3 = nn.Linear(hidden2, output_size)   # 4 -> 1
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = NeuralNet(input_size=2, hidden1=8, hidden2=4, output_size=1)
```

**`super().__init__()`** runs the `nn.Module` constructor. It sets up the machinery that registers layers and parameters, so `model.parameters()`, `model.train()` and `model.eval()` work. Call it **first**, before you create any layer, or PyTorch raises an error.

**`nn.Linear(in_features, out_features)`** is a fully connected layer that computes `y = x·Wᵀ + b`.
- `nn.Linear(2, 8)` has a weight matrix `W` of shape (8, 2) and bias `b` of shape (8), so 16 + 8 = 24 parameters.
- Weights start as small random values, and training adjusts them.

**Neurons:**
- **Input neurons = 2** must equal the number of features.
- **Hidden layers** are 8 and 4 neurons here. These are design choices: more neurons or layers can learn more complex patterns but take longer to train and can overfit.
- **Output neurons = 1** for binary classification (one probability). Use `num_classes` outputs for multiclass, and 1 output (no activation) for regression.

**Activation functions:**
- **ReLU** `max(0, x)` is used in hidden layers. It adds non-linearity. Without it, stacked Linear layers would collapse into one linear function.
- **Sigmoid** `1 / (1 + e^-x)` squashes the output into (0, 1), so it can be read as P(pass).

**`forward(x)`** defines how the data flows. You never call it directly. Write `model(x)` instead, which calls `forward` and also runs PyTorch hooks.

**Architecture (layer by layer):**

```
 Input (2)        Hidden 1 (8)        Hidden 2 (4)        Output (1)
 studied  ●──┐    ● ● ● ● ● ● ● ●     ● ● ● ●             ●  P(pass)
 slept    ●──┴──► Linear(2→8)  ──►    Linear(8→4)  ──►    Linear(4→1)
                  + ReLU              + ReLU              + Sigmoid
```

| Layer | Operation | Output shape | Params |
|---|---|---|---|
| Input | n/a | (batch, 2) | 0 |
| fc1 + ReLU | Linear(2, 8) | (batch, 8) | 2×8 + 8 = 24 |
| fc2 + ReLU | Linear(8, 4) | (batch, 4) | 8×4 + 4 = 36 |
| fc3 + Sigmoid | Linear(4, 1) | (batch, 1) | 4×1 + 1 = 5 |
| **Total** | | | **65** |

`print(model)` shows the layers, and `sum(p.numel() for p in model.parameters())` returns 65.

---

## 6. Loss Function and Optimizer

```python
criterion = nn.BCELoss()
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

### Loss functions: which one when?

| Problem | Last layer | Loss | Label shape / dtype |
|---|---|---|---|
| **Binary** classification | `Linear(…, 1)` + **Sigmoid** | `nn.BCELoss()` | `(N, 1)` float 0/1 |
| **Binary** classification (recommended) | `Linear(…, 1)`, **no** sigmoid (raw "logits") | `nn.BCEWithLogitsLoss()` | `(N, 1)` float 0/1 |
| **Multiclass** (C classes) | `Linear(…, C)`, **no** softmax | `nn.CrossEntropyLoss()` | `(N,)` **long** class indices 0…C−1 |
| **Regression** | `Linear(…, 1)`, no activation | `nn.MSELoss()` | `(N, 1)` float |

- **BCELoss** (Binary Cross Entropy) expects probabilities, so the model must end with Sigmoid.
- **BCEWithLogitsLoss** combines Sigmoid and BCE in one numerically stable step. Remove the sigmoid from `forward()` and apply `torch.sigmoid()` yourself only at prediction time.
- **CrossEntropyLoss** applies softmax internally, so don't add softmax to the model. Its labels are integers, not one-hot vectors.
- **MSELoss** is the mean of (prediction − target)², used for continuous targets.

```python
# initialisation examples
criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
```

### Optimizers

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)              # adaptive, a good default
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # classic gradient descent
```

- **`model.parameters()`** returns every trainable tensor (all W and b of fc1, fc2, fc3). The optimizer needs this list to know *which* tensors to update.
- **Learning rate (`lr`)** is the step size of each update: `w = w − lr × gradient`.
  - Too large: the loss jumps around or diverges.
  - Too small: training is very slow.
  - Typical values: Adam 0.001 to 0.01, SGD 0.01 to 0.1.
- **SGD** uses the same fixed step for every weight. Momentum adds "velocity" so it moves faster and more smoothly.
- **Adam** gives each weight its own adaptive step based on its gradient history. It usually converges faster with little tuning.

---

## 7. Training Loop

```python
epochs = 100
loss_history = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for features, labels in loader:
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

    epoch_loss = running_loss / len(dataset)
    loss_history.append(epoch_loss)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {epoch_loss:.4f}")
```

| Statement | What it does |
|---|---|
| `for epoch in range(epochs)` | **Epoch** = one full pass over all 200 samples. 100 epochs means the model sees the data 100 times |
| `model.train()` | Puts the model in training mode. This matters for Dropout and BatchNorm layers. Our model has neither, but calling it is good practice |
| `for features, labels in loader` | Loops over **batches** (13 per epoch). Each batch triggers one weight update, so 100 × 13 = 1300 updates in total |
| `outputs = model(features)` | **Forward propagation:** the batch (16, 2) flows through fc1→ReLU→fc2→ReLU→fc3→Sigmoid and produces probabilities (16, 1). PyTorch records every operation in a *computation graph* |
| `loss = criterion(outputs, labels)` | Measures how wrong the predictions are and returns one number (the mean loss over the batch) |
| `optimizer.zero_grad()` | Resets the gradients to zero. PyTorch **accumulates** gradients by default, so without this step each batch's gradients would add on top of the previous batch's |
| `loss.backward()` | **Backpropagation:** walks the computation graph backwards and computes `∂loss/∂w` for every parameter, storing it in `w.grad` |
| `optimizer.step()` | Updates every weight using its gradient. For SGD that is `w ← w − lr · w.grad`, and Adam uses its adaptive version |
| `loss.item()` | Converts the 1-element tensor into a Python float, which is cheap to store and detaches it from the graph |
| `* features.size(0)` | Weights each batch's loss by its size, so the smaller last batch counts correctly |
| `epoch_loss = running_loss / len(dataset)` | Average loss per sample for this epoch |
| `loss_history.append(...)` / `print` | Tracks progress, which you can plot later. The loss should go **down** over time |

### Backpropagation and weight updates (concept)

1. **Forward:** the input goes through the layers and becomes a prediction, and the loss compares it with the truth.
2. **Backward:** using the **chain rule**, PyTorch works out how much each weight contributed to the error, starting from the loss and moving back layer by layer:
   `∂L/∂W1 = ∂L/∂out · ∂out/∂h2 · ∂h2/∂h1 · ∂h1/∂W1`
3. **Update:** each weight moves a small step in the **opposite** direction of its gradient (downhill on the loss surface): `w_new = w_old − lr × ∂L/∂w`.
4. Repeating this over many batches and epochs moves the weights toward values that minimise the loss.

**Actual output from the script:**
```
Epoch [ 10/100]  Loss: 0.1532
Epoch [ 50/100]  Loss: 0.0446
Epoch [100/100]  Loss: 0.0187
Training accuracy: 99.50%
```

---

## 8. Prediction

```python
new_students = torch.tensor([[9.0, 8.0],
                             [2.0, 6.0],
                             [7.0, 3.0],
                             [5.0, 9.0]])
new_scaled = new_students / 10.0        # SAME preprocessing as training!

model.eval()
with torch.no_grad():
    probs = model(new_scaled)           # (4, 1) probabilities
    classes = (probs >= 0.5).int()      # threshold
```

- **`model.eval()`** switches to evaluation mode: Dropout is turned off and BatchNorm uses its stored statistics. Always call it before predicting.
- **`torch.no_grad()`** stops PyTorch from building the computation graph or computing gradients. Prediction doesn't need them, so it is faster and uses less memory.
- **Model prediction:** `model(new_scaled)` runs the forward pass. New data must get the **same scaling** as the training data.
- **Probability:** our model ends with Sigmoid, so the outputs are already probabilities. With `BCEWithLogitsLoss` (no sigmoid in the model), use `probs = torch.sigmoid(model(x))`.
- **Class conversion:**
  - Binary: `(probs >= 0.5)`. Change the threshold to trade precision for recall.
  - Multiclass: `probs = torch.softmax(logits, dim=1)` and `classes = torch.argmax(logits, dim=1)`.
  - Regression: the output *is* the prediction, `pred = model(x)` (undo any target scaling if you used one).

**Actual output:**
```
studied=9h slept=8h -> P(pass)=1.000 -> PASS
studied=2h slept=6h -> P(pass)=0.000 -> FAIL
studied=7h slept=3h -> P(pass)=0.917 -> PASS   (7 + 1.5 = 8.5 > 8)
studied=5h slept=9h -> P(pass)=0.999 -> PASS   (5 + 4.5 = 9.5 > 8)
```

### Quick reference: multiclass and regression variants

```python
# Multiclass (3 classes)
model = NeuralNet(input_size=2, hidden1=8, hidden2=4, output_size=3)   # remove sigmoid in forward
criterion = nn.CrossEntropyLoss()           # labels: torch.long, shape (N,)
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(logits, dim=1)

# Regression
model = NeuralNet(input_size=2, hidden1=8, hidden2=4, output_size=1)   # remove sigmoid in forward
criterion = nn.MSELoss()                    # labels: float32, shape (N, 1)
with torch.no_grad():
    pred_value = model(x)                   # continuous values
```

---

## 9. Complete Code

```python
# ---------------- 1. Imports ----------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)

# ---------------- 2. Sample Dataset ----------------
N = 200
X = torch.rand(N, 2) * 10                                  # hours studied, hours slept
y = ((X[:, 0] + 0.5 * X[:, 1]) > 8).float().unsqueeze(1)   # 1 = pass, 0 = fail
X = X / 10.0                                               # scale to [0, 1]


# ---------------- 3. Dataset Class ----------------
class StudentDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


dataset = StudentDataset(X, y)

# ---------------- 4. DataLoader ----------------
loader = DataLoader(dataset, batch_size=16, shuffle=True)


# ---------------- 5. Neural Network Class ----------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Model initialization
model = NeuralNet(input_size=2, hidden1=8, hidden2=4, output_size=1)
print(model)

# ---------------- 6. Loss Function and Optimizer ----------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---------------- 7. Training Loop ----------------
epochs = 100
loss_history = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for features, labels in loader:
        outputs = model(features)              # forward pass
        loss = criterion(outputs, labels)      # loss

        optimizer.zero_grad()                  # clear gradients
        loss.backward()                        # backpropagation
        optimizer.step()                       # update weights

        running_loss += loss.item() * features.size(0)

    epoch_loss = running_loss / len(dataset)
    loss_history.append(epoch_loss)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1:3d}/{epochs}]  Loss: {epoch_loss:.4f}")

# Training accuracy
model.eval()
with torch.no_grad():
    acc = ((model(X) >= 0.5).float() == y).float().mean().item()
print(f"Training accuracy: {acc * 100:.2f}%")

# ---------------- 8. Prediction ----------------
new_students = torch.tensor([[9.0, 8.0], [2.0, 6.0], [7.0, 3.0], [5.0, 9.0]])
new_scaled = new_students / 10.0

model.eval()
with torch.no_grad():
    probs = model(new_scaled)
    classes = (probs >= 0.5).int()

for x_raw, p, c in zip(new_students, probs, classes):
    label = "PASS" if c.item() == 1 else "FAIL"
    print(f"studied={x_raw[0]:.0f}h slept={x_raw[1]:.0f}h -> P(pass)={p.item():.3f} -> {label}")
```
