import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../own/week12"))  # data files live here
import time
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

USERS = ["000", "001", "002"]
STEP = 10          # keep every 10th GPS point (points are only ~1-5 s apart)

# Task 1: load GPS trajectories straight from the zip
zf = zipfile.ZipFile("archive (3).zip")
frames = []
for info in zf.infolist():
    parts = info.filename.split("/")   # Geolife Trajectories 1.3/Data/<user>/Trajectory/<file>.plt
    if len(parts) == 5 and parts[2] in USERS and parts[4].endswith(".plt"):
        df = pd.read_csv(zf.open(info), skiprows=6, header=None,
                         names=["lat", "lon", "zero", "alt", "days", "date", "time"])
        df["user"] = parts[2]
        frames.append(df.iloc[::STEP])
data = pd.concat(frames, ignore_index=True)[["user", "lat", "lon", "alt", "date", "time"]]
print("Loaded points:", len(data))

# Remove missing / invalid coordinates (outside valid range, zeros, or far outside Beijing)
data = data.dropna()
data = data[(data.lat.between(-90, 90)) & (data.lon.between(-180, 180)) & (data.lat != 0) & (data.lon != 0)]
data = data[data.lat.between(39.4, 41.1) & data.lon.between(115.4, 117.6)]

# Chronological order per user
data["timestamp"] = pd.to_datetime(data["date"] + " " + data["time"])
data = data.sort_values(["user", "timestamp"]).reset_index(drop=True)
print("Clean points:", len(data))
print(data.groupby("user").timestamp.agg(["count", "min", "max"]))

coords = data[["lat", "lon"]].values


# Task 2: K-Means on (lat, lon), from scratch
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


# Turn each user's point stream into a sequence of visited locations.
# Consecutive points in the same cluster are merged into ONE visit, so "next location"
# means the next DIFFERENT region the user goes to (otherwise the answer is almost always "same").
def build_sequences(labels):
    seqs = {}
    for u in USERS:
        idx = np.where(data.user.values == u)[0]
        lab = labels[idx]
        keep = np.r_[True, lab[1:] != lab[:-1]]
        seqs[u] = {"obs": lab[keep], "lat": coords[idx[keep], 0], "lon": coords[idx[keep], 1]}
    return seqs


clusterings = {}
labels_by_K = {}
for K in [5, 10, 15, 20]:
    km = KMeans(K)
    km.fit(coords)
    centers, labels = km.centers, km.labels
    clusterings[K] = (centers, build_sequences(labels))
    labels_by_K[K] = labels
    print(f"K={K:2d}: visit-sequence lengths", {u: len(s["obs"]) for u, s in clusterings[K][1].items()})

# Output 1: geographical clusters + centroids
centers10, _ = clusterings[10]
labels10 = labels_by_K[10]
plt.figure(figsize=(8, 7))
plt.scatter(coords[:, 1], coords[:, 0], c=labels10, cmap="tab10", s=1)
plt.scatter(centers10[:, 1], centers10[:, 0], c="k", marker="X", s=150, label="centroids")
for k, (la, lo) in enumerate(centers10):
    plt.text(lo, la, f" {k}", fontsize=12, weight="bold")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Output 1: GPS points grouped into K=10 clusters")
plt.legend()
plt.savefig("hmm_out1_clusters.png")


# Task 3: discrete HMM trained with Baum-Welch (scaled forward-backward), from scratch
class HMM:
    def __init__(self, N, M, iters=50, seed=0):
        self.N, self.M, self.iters, self.seed = N, M, iters, seed   # N hidden states, M symbols

    def forward(self, obs):
        T = len(obs)
        alpha = np.zeros((T, self.N))
        c = np.zeros(T)
        alpha[0] = self.pi * self.B[:, obs[0]]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]
        return alpha, c                        # log-likelihood = sum(log c)

    def backward(self, obs, c):
        T = len(obs)
        beta = np.zeros((T, self.N))
        beta[-1] = 1
        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (self.B[:, obs[t + 1]] * beta[t + 1]) / c[t + 1]
        return beta

    def fit(self, seqs):
        # Baum-Welch on a list of observation sequences
        N, M = self.N, self.M
        rng = np.random.default_rng(self.seed)
        self.pi = rng.random(N); self.pi /= self.pi.sum()
        self.A = rng.random((N, N)); self.A /= self.A.sum(axis=1, keepdims=True)
        self.B = rng.random((N, M)); self.B /= self.B.sum(axis=1, keepdims=True)
        for _ in range(self.iters):
            pi_acc = np.zeros(N); A_num = np.zeros((N, N)); B_num = np.zeros((N, M)); gamma_sum = np.zeros(N)
            for obs in seqs:
                alpha, c = self.forward(obs)
                beta = self.backward(obs, c)
                gamma = alpha * beta                                     # P(state at t | all obs)
                gamma /= gamma.sum(axis=1, keepdims=True)
                xi = (alpha[:-1, :, None] * self.A[None] * (self.B[:, obs[1:]].T * beta[1:])[:, None, :]) / c[1:, None, None]
                pi_acc += gamma[0]
                A_num += xi.sum(axis=0)
                gamma_sum += gamma[:-1].sum(axis=0)
                for t, o in enumerate(obs):
                    B_num[:, o] += gamma[t]
            self.pi = pi_acc / pi_acc.sum()
            self.A = A_num / (gamma_sum[:, None] + 1e-12) + 1e-6
            self.A /= self.A.sum(axis=1, keepdims=True)
            self.B = B_num + 1e-6                                         # small smoothing: no zero emissions
            self.B /= self.B.sum(axis=1, keepdims=True)

    def viterbi(self, obs):
        T = len(obs)
        logA, logB = np.log(self.A + 1e-300), np.log(self.B + 1e-300)
        delta = np.log(self.pi + 1e-300) + logB[:, obs[0]]
        back = np.zeros((T, self.N), dtype=int)
        for t in range(1, T):
            scores = delta[:, None] + logA
            back[t] = scores.argmax(axis=0)
            delta = scores.max(axis=0) + logB[:, obs[t]]
        path = [int(delta.argmax())]
        for t in range(T - 1, 0, -1):
            path.append(back[t, path[-1]])
        return [int(x) for x in path[::-1]]

    # Task 4: prediction. For each test position t we only use obs[0..t-horizon]:
    #   forward  -> P(current hidden state | history)
    #   @ A^h    -> P(hidden state h steps ahead)
    #   @ B      -> P(next location cluster)
    def predict_proba(self, obs, start, horizon=1):
        alpha, _ = self.forward(obs)
        Ah = np.linalg.matrix_power(self.A, horizon)
        return np.array([alpha[t - horizon] @ Ah @ self.B for t in range(start, len(obs))])

    def predict(self, obs, start, horizon=1):
        return self.predict_proba(obs, start, horizon).argmax(axis=1)   # most probable cluster


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def scores(actual, dists, act_lat, act_lon, centers):
    top1 = dists.argmax(axis=1)
    top3 = np.argsort(-dists, axis=1)[:, :3]
    precs, recs, f1s = [], [], []
    for k in np.unique(np.r_[actual, top1]):             # macro precision / recall / F1
        tp = np.sum((top1 == k) & (actual == k))
        p = tp / max(np.sum(top1 == k), 1)
        r = tp / max(np.sum(actual == k), 1)
        precs.append(p); recs.append(r); f1s.append(2 * p * r / (p + r) if p + r else 0)
    dist_km = haversine_km(act_lat, act_lon, centers[top1, 0], centers[top1, 1])
    return {"top1": np.mean(top1 == actual), "top3": np.mean([a in t for a, t in zip(actual, top3)]),
            "precision": np.mean(precs), "recall": np.mean(recs), "f1": np.mean(f1s),
            "haversine_km": dist_km.mean(), "pred": top1}


# One experiment run: returns metrics averaged over the users
def run(K=10, N=5, train_frac=0.8, shared=False, horizon=1, verbose=False):
    centers, seqs = clusterings[K]
    start_time = time.time()
    cut = {u: int(0.8 * len(s["obs"])) for u, s in seqs.items()}            # test = last 20 % always
    train = {u: s["obs"][: int(train_frac * len(s["obs"]))] for u, s in seqs.items()}
    models = {}
    if shared:
        shared_model = HMM(N, K)
        shared_model.fit(list(train.values()))
        models = {u: shared_model for u in USERS}
    else:
        for u in USERS:
            models[u] = HMM(N, K)
            models[u].fit([train[u]])
    results = {}
    for u in USERS:
        s = seqs[u]
        dists = models[u].predict_proba(s["obs"], start=cut[u], horizon=horizon)
        results[u] = scores(s["obs"][cut[u]:], dists, s["lat"][cut[u]:], s["lon"][cut[u]:], centers)
        if verbose:
            r = results[u]
            print(f"  user {u}: top1={r['top1']:.3f} top3={r['top3']:.3f} P={r['precision']:.3f} "
                  f"R={r['recall']:.3f} F1={r['f1']:.3f} dist={r['haversine_km']:.2f} km")
    avg = {m: np.mean([results[u][m] for u in USERS]) for m in ["top1", "top3", "precision", "recall", "f1", "haversine_km"]}
    avg["time"] = time.time() - start_time
    return avg, results, models


def show(name, m):
    print(f"{name:22s} top1={m['top1']:.3f} top3={m['top3']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
          f"F1={m['f1']:.3f} dist={m['haversine_km']:.2f} km time={m['time']:.1f}s")


# Main model: K=10 clusters, N=5 hidden states, user-specific HMMs, 80:20 split
print("\n=== Main model (K=10, N=5, per-user HMM) ===")
main, main_res, main_models = run(K=10, N=5, verbose=True)
show("Average", main)

model = main_models["000"]
pi, A, B = model.pi, model.A, model.B
print("\nUser 000 HMM parameters")
print("Initial probabilities pi:", pi.round(3))
print("Transition matrix A:\n", A.round(3))
print("Emission matrix B:\n", B.round(3))

# Example: predict the next location for user 000 after the training history
centers, seqs = clusterings[10]
obs = seqs["000"]["obs"]
cut = int(0.8 * len(obs))
history = obs[:cut]
alpha, _ = model.forward(history)
state_now = alpha[-1]
state_next = state_now @ A
loc_next = state_next @ B
path = model.viterbi(history)
best = int(loc_next.argmax())
print("\nLast 10 visited clusters :", history[-10:])
print("Viterbi hidden states    :", path[-10:])
print("P(current hidden state)  :", state_now.round(3))
print("P(next hidden state)     :", state_next.round(3))
print("P(next location cluster) :", loc_next.round(3))
print(f"Most probable next location: cluster {best} -> lat={centers[best, 0]:.5f}, lon={centers[best, 1]:.5f}")
print(f"Actual next location       : cluster {obs[cut]}")

# Output 2: HMM transition diagram (user 000)
plt.figure(figsize=(7, 7))
Nst = A.shape[0]
ang = np.linspace(0, 2 * np.pi, Nst, endpoint=False)
pos = np.c_[np.cos(ang), np.sin(ang)] * 3
for i in range(Nst):
    for j in range(Nst):
        if A[i, j] < 0.05:
            continue
        if i == j:
            plt.text(*(pos[i] * 1.28), f"self {A[i, i]:.2f}", ha="center", fontsize=9, color="purple")
            continue
        p1, p2 = pos[i], pos[j]
        offset = np.array([p2[1] - p1[1], p1[0] - p2[0]]) * 0.06   # shift so i->j and j->i don't overlap
        plt.annotate("", xy=p2 + offset - (p2 - p1) * 0.12, xytext=p1 + offset + (p2 - p1) * 0.12,
                     arrowprops=dict(arrowstyle="->", lw=1 + 4 * A[i, j], color="gray"))
        mid = p1 + (p2 - p1) * 0.3 + offset * 2.5             # label near the source node
        plt.text(*mid, f"{A[i, j]:.2f}", fontsize=8, ha="center", bbox=dict(fc="white", ec="none", pad=1))
for i in range(Nst):
    plt.gca().add_patch(plt.Circle(pos[i], 0.45, color="skyblue", zorder=3))
    plt.text(*pos[i], f"S{i}", ha="center", va="center", fontsize=13, weight="bold", zorder=4)
plt.xlim(-4.5, 4.5); plt.ylim(-4.5, 4.5); plt.axis("off")
plt.title("Output 2: HMM transition diagram (user 000, arrows for P >= 0.05)")
plt.savefig("hmm_out2_transitions.png")

# Output 3: actual vs predicted next locations (user 000, first 40 test steps)
s = seqs["000"]
pred = main_res["000"]["pred"][:40]
act_lat, act_lon = s["lat"][cut:cut + 40], s["lon"][cut:cut + 40]
plt.figure(figsize=(8, 7))
plt.scatter(coords[:, 1], coords[:, 0], c="lightgray", s=1)
plt.scatter(centers[:, 1], centers[:, 0], c="k", marker="X", s=100, label="cluster centroids")
for la, lo, p in zip(act_lat, act_lon, pred):
    plt.plot([lo, centers[p, 1]], [la, centers[p, 0]], "r-", lw=0.5)
plt.scatter(act_lon, act_lat, c="blue", s=25, label="actual next location")
plt.scatter(centers[pred, 1], centers[pred, 0], facecolors="none", edgecolors="red", s=200, label="predicted (centroid)")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.title("Output 3: Actual vs predicted next locations (user 000)")
plt.legend()
plt.savefig("hmm_out3_actual_vs_pred.png")

# Experiment 1: number of geographical clusters
print("\n=== Experiment 1: number of clusters K (N=5) ===")
exp1 = {}
for K in [5, 10, 15, 20]:
    exp1[K], _, _ = run(K=K, N=5)
    show(f"K={K}", exp1[K])

# Experiment 2: amount of training history
print("\n=== Experiment 2: training fraction (K=10, N=5) ===")
exp2 = {}
for frac in [0.5, 0.6, 0.8]:
    exp2[frac], _, _ = run(K=10, N=5, train_frac=frac)
    show(f"train {int(frac * 100)}%", exp2[frac])

# Experiment 3: number of hidden states
print("\n=== Experiment 3: hidden states N (K=10) ===")
exp3 = {}
for N in [2, 3, 5, 8]:
    exp3[N], _, _ = run(K=10, N=N)
    show(f"N={N}", exp3[N])

# Experiment 4: user-specific vs shared HMM
print("\n=== Experiment 4: user-specific vs shared HMM (K=10, N=5) ===")
print("User-specific:")
exp4_user, _, _ = run(K=10, N=5, verbose=True)
print("Shared:")
exp4_shared, _, _ = run(K=10, N=5, shared=True, verbose=True)
show("User-specific avg", exp4_user)
show("Shared avg", exp4_shared)

# Experiment 5: prediction horizon (1 = next location, 2 = the one after, ...)
print("\n=== Experiment 5: prediction horizon (K=10, N=5) ===")
exp5 = {}
for h in [1, 2, 3, 5]:
    exp5[h], _, _ = run(K=10, N=5, horizon=h)
    show(f"{h} step(s) ahead", exp5[h])

# Output 4: experimental comparison
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
for a, exp, xlabel in [(ax[0, 0], exp1, "K-Means clusters K"), (ax[0, 1], exp3, "Hidden states N"),
                       (ax[1, 0], exp2, "Training fraction"), (ax[1, 1], exp5, "Prediction horizon (steps)")]:
    xs = list(exp.keys())
    a.plot(xs, [exp[x]["top1"] for x in xs], "o-", label="Top-1")
    a.plot(xs, [exp[x]["top3"] for x in xs], "s-", label="Top-3")
    a.plot(xs, [exp[x]["f1"] for x in xs], "^-", label="Macro F1")
    a.set_xlabel(xlabel); a.set_ylabel("Score"); a.set_xticks(xs); a.set_ylim(0, 1); a.legend()
plt.suptitle("Output 4: Prediction accuracy across experiments")
plt.tight_layout()
plt.savefig("hmm_out4_experiments.png")

plt.show()
