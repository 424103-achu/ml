import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../own/week11"))  # data files live here
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf   # plain TensorFlow only (no Keras): tf.Variable + GradientTape

np.random.seed(42)
tf.random.set_seed(42)

# Load data (5331 positive + 5331 negative reviews)
pos = open("rt-polarity.pos", encoding="latin-1").read().strip().split("\n")
neg = open("rt-polarity.neg", encoding="latin-1").read().strip().split("\n")
texts = pos + neg
labels = np.array([1] * len(pos) + [0] * len(neg))
print("Total samples:", len(texts), "| pos:", len(pos), "| neg:", len(neg))

# Shuffle and split 8500 train / 1100 val / 1062 test
idx = np.random.permutation(len(texts))
texts = [texts[i] for i in idx]
labels = labels[idx]
train_txt, val_txt, test_txt = texts[:8500], texts[8500:9600], texts[9600:]
y_train, y_val, y_test = labels[:8500], labels[8500:9600], labels[9600:]
print("Train:", len(train_txt), "| Val:", len(val_txt), "| Test:", len(test_txt))


# Task 1: tokenize (lowercase + split by space)
def tokenize(s):
    return s.lower().split()


# Build vocabulary from TRAIN data only, with special tokens
vocab = {"<pad>": 0, "<unk>": 1}
for s in train_txt:
    for w in tokenize(s):
        if w not in vocab:
            vocab[w] = len(vocab)
print("Vocabulary size:", len(vocab))

# Convert sentences to integer sequences, pad / truncate to max_len
max_len = 50


def encode(s):
    ids = [vocab.get(w, vocab["<unk>"]) for w in tokenize(s)][:max_len]  # truncate
    return ids + [vocab["<pad>"]] * (max_len - len(ids))                 # pad


X_train = np.array([encode(s) for s in train_txt], dtype=np.int32)
X_val = np.array([encode(s) for s in val_txt], dtype=np.int32)
X_test = np.array([encode(s) for s in test_txt], dtype=np.int32)
print("Example:", train_txt[0][:60], "...")
print("Encoded:", X_train[0][:15], "...")

# Dataset + batching with tf.data (TensorFlow's version of PyTorch Dataset/DataLoader)
batch_size = 64
to_float = lambda x, y: (x, tf.cast(y, tf.float32))
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(8500, seed=42).batch(batch_size).map(to_float)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).map(to_float)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).map(to_float)

# Task 2: models built from tf.Variables
emb_dim, hidden = 128, 128


def glorot(shape):
    limit = np.sqrt(6 / (shape[0] + shape[1]))
    return tf.Variable(tf.random.uniform(shape, -limit, limit))


# Task 3: loss (binary cross-entropy)
def bce_loss(logits, y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))


class SentimentRNN:
    # kind = "RNN" (simple RNN cell) or "LSTM"
    def __init__(self, kind, vocab_size, lr=1e-3):
        self.kind, self.lr = kind, lr
        gates = 1 if kind == "RNN" else 4          # LSTM has 4 gates: input, forget, cell, output
        self.p = {
            "E": tf.Variable(tf.random.normal([vocab_size, emb_dim], stddev=0.1)),   # embedding table
            "Wx": glorot([emb_dim, gates * hidden]),                                 # input -> hidden
            "Wh": glorot([hidden, gates * hidden]),                                  # hidden -> hidden
            "b": tf.Variable(tf.zeros([gates * hidden])),
            "Wo": glorot([hidden, 1]),                                               # linear classifier
            "bo": tf.Variable(tf.zeros([1])),
        }
        # Adam optimizer state (written by hand)
        self.m = {k: tf.Variable(tf.zeros_like(v)) for k, v in self.p.items()}
        self.v = {k: tf.Variable(tf.zeros_like(v)) for k, v in self.p.items()}
        self.t = tf.Variable(0.0)
        self.train_step = tf.function(self._train_step)
        self.logits = tf.function(lambda x: self.forward(x, training=False))

    def forward(self, x, training=False):
        p = self.p
        emb = tf.nn.embedding_lookup(p["E"], x)                 # (batch, 50, 128)
        mask = tf.cast(x != 0, tf.float32)                      # 0 at <pad> positions
        n = tf.shape(x)[0]
        h = tf.zeros([n, hidden])
        c = tf.zeros([n, hidden])
        for t in range(max_len):
            z = emb[:, t, :] @ p["Wx"] + h @ p["Wh"] + p["b"]
            if self.kind == "RNN":
                h_new = tf.tanh(z)                              # simple RNN cell
                c_new = c
            else:
                i, f, g, o = tf.split(z, 4, axis=1)             # LSTM cell
                c_new = tf.sigmoid(f) * c + tf.sigmoid(i) * tf.tanh(g)
                h_new = tf.sigmoid(o) * tf.tanh(c_new)
            m = mask[:, t:t + 1]
            h = m * h_new + (1 - m) * h                         # keep old state at padding
            c = m * c_new + (1 - m) * c
        if training:
            h = tf.nn.dropout(h, rate=0.5)
        return tf.squeeze(h @ p["Wo"] + p["bo"], axis=1)        # logits = final hidden state -> 1 output

    def adam_step(self, grads, b1=0.9, b2=0.999, eps=1e-7):
        self.t.assign_add(1.0)
        for k in self.p:
            g = tf.convert_to_tensor(grads[k])                  # embedding grad comes as IndexedSlices
            self.m[k].assign(b1 * self.m[k] + (1 - b1) * g)
            self.v[k].assign(b2 * self.v[k] + (1 - b2) * g * g)
            m_hat = self.m[k] / (1 - b1 ** self.t)
            v_hat = self.v[k] / (1 - b2 ** self.t)
            self.p[k].assign_sub(self.lr * m_hat / (tf.sqrt(v_hat) + eps))

    def _train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.forward(x, training=True)
            loss = bce_loss(logits, y)
        grads = tape.gradient(loss, self.p)
        self.adam_step(grads)
        acc = tf.reduce_mean(tf.cast(tf.cast(logits > 0, tf.float32) == y, tf.float32))
        return loss, acc

    def evaluate(self, ds):
        losses, correct, total = 0.0, 0, 0
        for x, y in ds:
            logits = self.logits(x)
            losses += float(bce_loss(logits, y)) * len(y)
            correct += int(tf.reduce_sum(tf.cast(tf.cast(logits > 0, tf.float32) == y, tf.int32)))
            total += len(y)
        return losses / total, correct / total

    def fit(self, train_ds, val_ds, epochs=12):
        # train, keep the best weights by validation loss
        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        best_val = np.inf
        for epoch in range(epochs):
            tl, ta, nb = 0.0, 0.0, 0
            for x, y in train_ds:
                loss, acc = self.train_step(x, y)
                tl += float(loss); ta += float(acc); nb += 1
            vl, va = self.evaluate(val_ds)
            self.history["loss"].append(tl / nb); self.history["accuracy"].append(ta / nb)
            self.history["val_loss"].append(vl); self.history["val_accuracy"].append(va)
            saved = ""
            if vl < best_val:                                     # save best checkpoint
                best_val = vl
                self.best_params = {k: v.numpy().copy() for k, v in self.p.items()}
                saved = "  <- best, saved"
            print(f"Epoch {epoch + 1:2d}: train_loss={tl / nb:.4f} train_acc={ta / nb:.4f} "
                  f"val_loss={vl:.4f} val_acc={va:.4f}{saved}")
        for k in self.p:                                          # restore best weights
            self.p[k].assign(self.best_params[k])

    def predict(self, X, batch_size=256):
        # 1 = positive, 0 = negative   (logit > 0  <=>  sigmoid > 0.5)
        logits = np.concatenate([self.logits(tf.constant(X[i:i + batch_size])).numpy()
                                 for i in range(0, len(X), batch_size)])
        return (logits > 0).astype(int)


# Train both models 12 epochs
epochs = 12
history = {}
models = {}
for kind in ["RNN", "LSTM"]:
    print(f"\n===== Training {kind} =====")
    model = SentimentRNN(kind, len(vocab))
    print("Parameters:", {k: tuple(v.shape) for k, v in model.p.items()},
          "| total =", sum(int(np.prod(v.shape)) for v in model.p.values()))
    model.fit(train_ds, val_ds, epochs)
    np.savez(f"best_{kind}_weights.npz", **model.best_params)
    history[kind] = model.history
    models[kind] = model
    best = int(np.argmin(model.history["val_loss"]))
    print(f"{kind} best epoch: {best + 1} (val_loss={model.history['val_loss'][best]:.4f})")

# Task 4: plots
ep = range(1, epochs + 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i, kind in enumerate(["RNN", "LSTM"]):
    ax[i].plot(ep, history[kind]["loss"], "o-", label="Train loss")
    ax[i].plot(ep, history[kind]["val_loss"], "s-", label="Val loss")
    ax[i].set_title(f"{kind} - Loss")
    ax[i].set_xlabel("Epoch")
    ax[i].set_ylabel("Binary cross-entropy")
    ax[i].legend()
plt.tight_layout()
plt.savefig("loss_curves.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i, kind in enumerate(["RNN", "LSTM"]):
    ax[i].plot(ep, history[kind]["accuracy"], "o-", label="Train acc")
    ax[i].plot(ep, history[kind]["val_accuracy"], "s-", label="Val acc")
    ax[i].set_title(f"{kind} - Accuracy")
    ax[i].set_xlabel("Epoch")
    ax[i].set_ylabel("Accuracy")
    ax[i].legend()
plt.tight_layout()
plt.savefig("accuracy_curves.png")


# Final test metrics (computed by hand)
def metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return acc, prec, rec, f1, [[int(tn), int(fp)], [int(fn), int(tp)]]


print("\n===== Test set (best checkpoints) =====")
print("Model | Accuracy | Precision | Recall | F1")
for kind in ["RNN", "LSTM"]:
    y_pred = models[kind].predict(X_test)
    acc, prec, rec, f1, cm = metrics(y_test, y_pred)
    print(f"{kind:5s} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f}   confusion [[TN,FP],[FN,TP]] = {cm}")

plt.show()
