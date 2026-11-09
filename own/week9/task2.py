# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")
data = np.genfromtxt("Xor_Dataset.csv", delimiter=",", skip_header=1)
X = data[:, :2]
y = data[:, 2:3]
N = len(X)

# %%


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_derivative(a):
    return a*(1.0-a)


def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)


def init_params(hidden, rng):
    return {
        "W1": rng.normal(0, 1, (2, hidden)), "b1": np.zeros((1, hidden)),
        "W2": rng.normal(0, 1, (hidden, 1)), "b2": np.zeros((1, 1)),
    }


def forward(p, X):
    a1 = sigmoid(X@p["W1"]+p["b1"])
    a2 = sigmoid(a1@p["W2"]+p["b2"])
    return a1, a2


def backward(p, X, y, a1, a2):
    n = len(X) 
    d2 = (2.0/n)*(a2-y)*sigmoid_derivative(a2)
    d1 = (d2@p["W2"].T)*sigmoid_derivative(a1)
    return {
        "W2": a1.T@d2, "b2": d2.sum(axis=0, keepdims=True),
        "W1": X.T@d1, "b1": d1.sum(axis=0, keepdims=True),
    }

# %%


def train(lr, epochs=10000, hidden=4, seed=1, log=False):
    p = init_params(hidden, np.random.default_rng(seed))
    losses = []
    for epoch in range(epochs+1):
        a1, a2 = forward(p, X)
        loss = mse(y, a2)
        losses.append(loss)
        if log and epoch % 1000 == 0:
            print(f"epoch {epoch:5d} MSE = {loss:6f}")
        g = backward(p, X, y, a1, a2)
        for k in p:
            p[k] -= lr*g[k]
    return p, losses


# %%
print(f"Loaded {N} samples from Xor_Dataset.csv\n")
print("Training with lr = 0.5, 4 hidden nodes, 10000 epochs")
params, loss_main = train(0.5, log=True)
print("\nFinal inference (XOR truth table)")
print(" x1  x2 | target | output | rounded")
for x in np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float):
    out = forward(params, x.reshape(1, 2))[1][0, 0]
    t = int(x[0]) ^ int(x[1])
    print(
        f"  {int(x[0])}   {int(x[1])} |   {t}    | {out:.4f} |   {round(out)}")


# %%
print("\nLearning-rate stress test (10000 epochs, same init)")
curves = {}
for lr in [0.05, 0.1, 0.5, 1.0, 5.0]:
    _, l = train(lr)
    curves[lr] = l
    first = next((i for i, v in enumerate(l) if v < 0.01), None)
    print(f"  lr={lr:<4}  final MSE={l[-1]:.6f}  epochs to MSE<0.01: "
          f"{first if first is not None else 'never'}")


# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(loss_main, color="#2a6fdb")
ax[0].set(title="Loss decay (lr = 0.5)", xlabel="Epoch", ylabel="MSE loss")
ax[0].grid(alpha=0.3)
for lr, l in curves.items():
    ax[1].plot(l, label=f"lr = {lr}")
ax[1].set(title="Learning-rate comparison",
          xlabel="Epoch", ylabel="MSE loss", yscale="log")
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_plot.png", dpi=150)
plt.show()
print("\nSaved loss_plot.png")
