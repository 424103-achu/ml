import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task9"))  # data files live here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf   # plain TensorFlow only (no Keras)

# Load XOR data (columns X, Y = inputs, Z = target)
data = pd.read_csv("Xor_Dataset.csv")
X = tf.constant(data[["X", "Y"]].values, dtype=tf.float32)
y = tf.constant(data[["Z"]].values, dtype=tf.float32)
print("Samples:", len(data))
print(data.head())


# Sigmoid and its derivative
def sigmoid(z):
    return 1.0 / (1.0 + tf.exp(-z))


def sigmoid_derivative(a):          # a = sigmoid(z); shown for the lab, GradientTape does this for us
    return a * (1.0 - a)


def mse(y_true, y_pred):
    return tf.reduce_mean((y_true - y_pred) ** 2)


class XORNet:
    # 2-4-1 network: weights as tf.Variables, init N(0, 1)
    def __init__(self, hidden=4, lr=0.5, epochs=10000, seed=1):
        self.lr, self.epochs = lr, epochs
        tf.random.set_seed(seed)
        self.p = {
            "W1": tf.Variable(tf.random.normal([2, hidden])), "b1": tf.Variable(tf.zeros([1, hidden])),
            "W2": tf.Variable(tf.random.normal([hidden, 1])), "b2": tf.Variable(tf.zeros([1, 1])),
        }

    def forward(self, X):
        a1 = sigmoid(X @ self.p["W1"] + self.p["b1"])     # hidden layer
        return sigmoid(a1 @ self.p["W2"] + self.p["b2"])  # output layer

    def fit(self, X, y, log_every=1000, verbose=True):
        # full-batch gradient descent, gradients from GradientTape
        @tf.function
        def step():
            with tf.GradientTape() as tape:
                loss = mse(y, self.forward(X))
            grads = tape.gradient(loss, self.p)
            for k in self.p:
                self.p[k].assign_sub(self.lr * grads[k])    # w = w - lr * dL/dw
            return loss

        self.losses = []
        for epoch in range(self.epochs):
            self.losses.append(float(step()))
            if verbose and (epoch % log_every == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch:5d} | MSE = {self.losses[-1]:.6f}")

    def predict_proba(self, X):
        return self.forward(tf.constant(X, dtype=tf.float32)).numpy().ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# Train with lr = 0.5
print("\n=== Training (lr = 0.5, 10000 epochs) ===")
model = XORNet(lr=0.5)
model.fit(X, y)

# Loss plot
plt.figure(figsize=(7, 4))
plt.plot(model.losses)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("XOR training loss (TensorFlow, lr = 0.5)")
plt.savefig("loss_plot_tf.png")

# Final inference on the 4 XOR inputs
print("\n=== Inference ===")
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
probs = model.predict_proba(inputs)
classes = model.predict(inputs)
for (a, b), o, c in zip(inputs, probs, classes):
    print(f"{a} XOR {b} -> {o:.4f} -> class {c}  (expected {a ^ b})")

# Learning-rate stress test
print("\n=== Learning-rate stress test ===")
plt.figure(figsize=(7, 4))
for lr in [0.05, 0.1, 0.5, 1.0, 5.0]:
    m = XORNet(lr=lr)
    m.fit(X, y, verbose=False)
    l = m.losses
    below = next((i for i, v in enumerate(l) if v < 0.01), None)
    print(f"lr={lr:<5} final MSE={l[-1]:.6f}  epochs to MSE<0.01: {below if below is not None else 'never'}")
    plt.plot(l, label=f"lr={lr}")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.yscale("log")
plt.title("Learning-rate comparison")
plt.legend()
plt.savefig("lr_comparison_tf.png")

plt.show()
