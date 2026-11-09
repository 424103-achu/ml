import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task6"))  # data files live here
import numpy as np
import matplotlib.pyplot as plt

eps = 1e-15


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / np.sum(e)


print("softmax([2,1,0.5]) =", softmax(np.array([2, 1, 0.5])).round(3),
      " sum =", softmax(np.array([2, 1, 0.5])).sum())


def cross_entropy(p, q):
    q = np.clip(q, eps, 1)
    return -np.sum(p * np.log(q))


def kl_divergence(p, q):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))


p = np.array([1, 0, 0])
simulated_logits = [np.array([w, 1.0, 0.5]) for w in np.linspace(-2, 5, 100)]

ce, kl = [], []
for z in simulated_logits:
    q = softmax(z)
    ce.append(cross_entropy(p, q))
    kl.append(kl_divergence(p, q))

print("Max |CE - KL| =", np.max(np.abs(np.array(ce) - np.array(kl))))
print("For a one-hot p, entropy H(p) = 0, so CE = H(p) + KL = KL.")

plt.plot(ce, label="Cross-Entropy", lw=3)
plt.plot(kl, "--", label="KL Divergence", lw=2)
plt.xlabel("Prediction step (w increasing)")
plt.ylabel("Loss")
plt.title("Cross-Entropy and KL Divergence drop together")
plt.legend()
plt.show()
