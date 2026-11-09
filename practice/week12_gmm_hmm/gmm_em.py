import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../own/week12"))  # outputs saved here
import time
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# Generate data: two overlapping 2-D Gaussians (true labels kept only for evaluation)
def make_data(distance=3.0, n=300, seed=0):
    rng = np.random.default_rng(seed)
    X1 = rng.multivariate_normal([0, 0], [[1.0, 0.6], [0.6, 1.0]], n)
    X2 = rng.multivariate_normal([distance, distance / 2], [[1.0, -0.4], [-0.4, 1.5]], n)
    return np.vstack([X1, X2]), np.array([0] * n + [1] * n)


# Gaussian pdf (multivariate, by hand)
def gaussian_pdf(X, mean, cov):
    d = X.shape[1]
    diff = X - mean
    inv = np.linalg.inv(cov)
    expo = -0.5 * np.sum(diff @ inv * diff, axis=1)
    return np.exp(expo) / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))


# K-means (used for initialization)
class KMeans:
    def __init__(self, K, iters=100, seed=0):
        self.K, self.iters, self.seed = K, iters, seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        centers = X[rng.choice(len(X), self.K, replace=False)]
        for _ in range(self.iters):
            labels = self.assign(X, centers)
            new = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                            for k in range(self.K)])
            if np.allclose(new, centers):
                break
            centers = new
        self.centers, self.labels = centers, labels

    @staticmethod
    def assign(X, centers):
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return dist.argmin(axis=1)

    def predict(self, X):
        return self.assign(X, self.centers)              # nearest centroid


# Gaussian Mixture Model trained with EM
class GMM:
    def __init__(self, K, init="kmeans", max_iter=200, tol=1e-6, seed=0):
        self.K, self.init, self.max_iter, self.tol, self.seed = K, init, max_iter, tol, seed

    # Two initialization strategies
    def init_random(self, X):
        rng = np.random.default_rng(self.seed)
        lo, hi = X.min(axis=0), X.max(axis=0)
        self.means = rng.uniform(lo, hi, size=(self.K, X.shape[1]))  # random means inside data range
        self.covs = np.array([np.eye(X.shape[1]) for _ in range(self.K)])
        self.weights = np.ones(self.K) / self.K

    def init_kmeans(self, X):
        km = KMeans(self.K, seed=self.seed)
        km.fit(X)
        covs, weights = [], []
        for k in range(self.K):
            Xk = X[km.labels == k]
            covs.append(np.cov(Xk.T) + 1e-6 * np.eye(X.shape[1]))
            weights.append(len(Xk) / len(X))
        self.weights, self.means, self.covs = np.array(weights), km.centers, np.array(covs)

    def e_step(self, X):
        # responsibilities r[i, k] = P(component k | x_i)
        dens = np.column_stack([self.weights[k] * gaussian_pdf(X, self.means[k], self.covs[k])
                                for k in range(self.K)])
        total = dens.sum(axis=1, keepdims=True) + 1e-300
        return dens / total, np.sum(np.log(total))

    def m_step(self, X, r):
        # update weights, means, covariances
        n, d = X.shape
        Nk = r.sum(axis=0) + 1e-10
        self.weights = Nk / n
        self.means = (r.T @ X) / Nk[:, None]
        covs = []
        for k in range(self.K):
            diff = X - self.means[k]
            covs.append((r[:, k, None] * diff).T @ diff / Nk[k] + 1e-6 * np.eye(d))
        self.covs = np.array(covs)

    def fit(self, X):
        if self.init == "kmeans":
            self.init_kmeans(X)
        else:
            self.init_random(X)
        self.log_liks = []
        start = time.time()
        for it in range(self.max_iter):
            r, ll = self.e_step(X)
            self.log_liks.append(ll)
            self.m_step(X, r)
            if it > 0 and abs(self.log_liks[-1] - self.log_liks[-2]) < self.tol:   # converged
                break
        self.time = time.time() - start
        self.iters = len(self.log_liks)
        self.labels = r.argmax(axis=1)

    def predict(self, X):
        r, _ = self.e_step(X)
        return r.argmax(axis=1)                          # most responsible component


def gmm_em(X, K, init="kmeans", seed=0):
    model = GMM(K, init=init, seed=seed)
    model.fit(X)
    return model


# Metrics (all by hand)
def silhouette(X, labels):
    if len(np.unique(labels)) < 2:
        return 0.0
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == c].mean() for c in np.unique(labels) if c != labels[i])
        s[i] = (b - a) / max(a, b)
    return s.mean()


def contingency(y, c):
    return np.array([[np.sum((y == a) & (c == b)) for b in np.unique(c)] for a in np.unique(y)])


def ari(y, c):
    comb2 = lambda x: x * (x - 1) / 2
    T = contingency(y, c)
    sum_ij = comb2(T).sum()
    sum_a = comb2(T.sum(axis=1)).sum()
    sum_b = comb2(T.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb2(len(y))
    max_idx = (sum_a + sum_b) / 2
    return (sum_ij - expected) / (max_idx - expected)


def nmi(y, c):
    T = contingency(y, c) / len(y)
    pa, pb = T.sum(axis=1), T.sum(axis=0)
    nz = T > 0
    mi = np.sum(T[nz] * np.log(T[nz] / np.outer(pa, pb)[nz]))
    ha = -np.sum(pa * np.log(pa))
    hb = -np.sum(pb * np.log(pb))
    return mi / np.sqrt(ha * hb) if ha * hb > 0 else 0.0


def cluster_accuracy(y, c):
    # best matching between cluster ids and true labels
    best = 0
    for perm in permutations(np.unique(c)):
        mapping = {cl: i for i, cl in enumerate(perm)}
        best = max(best, np.mean(np.array([mapping[v] for v in c]) == y))
    return best


def evaluate(X, y, res):
    return {"loglik": res.log_liks[-1], "silhouette": silhouette(X, res.labels),
            "ARI": ari(y, res.labels), "NMI": nmi(y, res.labels),
            "iters": res.iters, "time": res.time}


def draw_ellipses(ax, res, color="k"):
    for mean, cov in zip(res.means, res.covs):
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        for nsig in [1, 2]:
            w, h = 2 * nsig * np.sqrt(vals[1]), 2 * nsig * np.sqrt(vals[0])
            ax.add_patch(Ellipse(mean, w, h, angle=angle, fill=False, color=color, lw=1.5))
        ax.plot(*mean, "kx", ms=10, mew=3)


def print_row(name, m):
    print(f"{name:22s} loglik={m['loglik']:9.2f}  sil={m['silhouette']:.3f}  ARI={m['ARI']:.3f}  "
          f"NMI={m['NMI']:.3f}  iters={m['iters']:3d}  time={m['time'] * 1000:.1f}ms")


# Output 1: original dataset
X, y = make_data(distance=3.0)
plt.figure(figsize=(6, 5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=10, label="Gaussian 1")
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=10, label="Gaussian 2")
plt.title("Output 1: Original dataset (true labels)")
plt.legend()
plt.savefig("gmm_out1_original.png")

# Experiment 1: random vs K-means initialization
print("\n=== Experiment 1: initialization strategy (K=2) ===")
res_rand = gmm_em(X, 2, init="random", seed=3)
res_km = gmm_em(X, 2, init="kmeans", seed=3)
print_row("Random init", evaluate(X, y, res_rand))
print_row("K-means init", evaluate(X, y, res_km))
print("K-means init -> weights:", res_km.weights.round(3))
print("means:\n", res_km.means.round(3))
print("covariances:\n", res_km.covs.round(3))

# Output 2: final clustering with ellipses
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a, res, name in [(ax[0], res_rand, "Random init"), (ax[1], res_km, "K-means init")]:
    a.scatter(X[:, 0], X[:, 1], c=res.labels, cmap="coolwarm", s=10)
    draw_ellipses(a, res)
    a.set_title(f"Output 2: GMM clustering ({name})")
plt.tight_layout()
plt.savefig("gmm_out2_clustering.png")

# Output 3: convergence graph
plt.figure(figsize=(7, 4))
plt.plot(res_rand.log_liks, "o-", ms=3, label=f"Random init ({res_rand.iters} iters)")
plt.plot(res_km.log_liks, "s-", ms=3, label=f"K-means init ({res_km.iters} iters)")
plt.xlabel("EM iteration")
plt.ylabel("Log-likelihood")
plt.title("Output 3: Convergence")
plt.legend()
plt.savefig("gmm_out3_convergence.png")

# Experiment 2: number of components K = 2..5
print("\n=== Experiment 2: number of components ===")
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
for i, K in enumerate([2, 3, 4, 5]):
    res = gmm_em(X, K, init="kmeans", seed=0)
    m = evaluate(X, y, res)
    n_params = K * (1 + 2 + 3) - 1  # weights + means + covariances (2-D)
    bic = -2 * m["loglik"] + n_params * np.log(len(X))
    print_row(f"K={K}", m)
    print(f"{'':22s} BIC={bic:.2f}")
    ax[i].scatter(X[:, 0], X[:, 1], c=res.labels, cmap="tab10", s=10)
    draw_ellipses(ax[i], res)
    ax[i].set_title(f"K={K}  loglik={m['loglik']:.1f}  BIC={bic:.1f}")
plt.suptitle("Output 4: Number of components comparison")
plt.tight_layout()
plt.savefig("gmm_out4_components.png")

# Experiment 3: cluster overlap
print("\n=== Experiment 3: cluster overlap ===")
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
for i, dist in enumerate([6.0, 4.0, 2.5, 1.0]):
    Xo, yo = make_data(distance=dist)
    res = gmm_em(Xo, 2, init="kmeans", seed=0)
    m = evaluate(Xo, yo, res)
    print_row(f"distance={dist}", m)
    print(f"{'':22s} accuracy={cluster_accuracy(yo, res.labels):.3f}")
    ax[i].scatter(Xo[:, 0], Xo[:, 1], c=res.labels, cmap="coolwarm", s=10)
    draw_ellipses(ax[i], res)
    ax[i].set_title(f"distance={dist}  ARI={m['ARI']:.2f}")
plt.suptitle("Experiment 3: effect of overlap")
plt.tight_layout()
plt.savefig("gmm_exp3_overlap.png")

# Experiment 4: 10 random initializations
print("\n=== Experiment 4: 10 random initializations ===")
lls, accs = [], []
for s in range(10):
    res = gmm_em(X, 2, init="random", seed=s)
    acc = cluster_accuracy(y, res.labels)
    lls.append(res.log_liks[-1])
    accs.append(acc)
    print(f"run {s}: loglik={lls[-1]:.2f}  accuracy={acc:.3f}  iters={res.iters}")
print(f"log-likelihood: mean={np.mean(lls):.2f}  std={np.std(lls):.2f}")
print(f"accuracy:       mean={np.mean(accs):.3f}  std={np.std(accs):.3f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(range(10), lls)
ax[0].set_title("Final log-likelihood per random run")
ax[0].set_ylim(min(lls) - 5, max(lls) + 5)
ax[1].bar(range(10), accs)
ax[1].set_title("Clustering accuracy per random run")
ax[1].set_ylim(0, 1)
plt.tight_layout()
plt.savefig("gmm_exp4_variability.png")

plt.show()
