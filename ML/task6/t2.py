import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

print(df.head())
print("Shape:", df.shape)
print("Classes:", iris.target_names)


# Softmax
def softmax(logits):
    exp_scores = np.exp(logits - np.max(logits))
    return exp_scores / np.sum(exp_scores)


# Cross-Entropy
def cross_entropy(p, q):
    q = np.clip(q, 1e-15, 1.0)
    return -np.sum(p * np.log(q))


# KL Divergence
def kl_divergence(p, q):
    p = np.clip(p, 1e-15, 1.0)
    q = np.clip(q, 1e-15, 1.0)
    return np.sum(p * np.log(p / q))


# True class: class 0
p_true = np.array([1.0, 0.0, 0.0])

# Generate logits
weights = np.linspace(-2, 5, 100)

ce_losses = []
kl_losses = []

for w in weights:
    logits = np.array([w, 1.0, 0.5])
    q = softmax(logits)

    ce_losses.append(cross_entropy(p_true, q))
    kl_losses.append(kl_divergence(p_true, q))

# Plot
plt.plot(ce_losses, label="Cross-Entropy")
plt.plot(kl_losses, "--", label="KL Divergence")
plt.xlabel("Prediction Step")
plt.ylabel("Loss")
plt.title("Cross-Entropy vs KL Divergence")
plt.legend()
plt.grid()
plt.show()

# Compare
print("CE and KL are equal:", np.allclose(ce_losses, kl_losses))
print("Mean difference:", np.mean(np.abs(np.array(ce_losses) - kl_losses)))
