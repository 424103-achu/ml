import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task4"))  # data files live here
import ast
import numpy as np


# SVD without np.linalg.svd (from ML-Lab-Practice-master/SVD/svd.py)
def svd(A):
    m, n = A.shape
    AT = A.T
    B = AT @ A

    eigenvalues, eigenvectors = np.linalg.eigh(B)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues = np.maximum(eigenvalues, 0)
    singular_values = np.sqrt(eigenvalues)

    tol = np.finfo(float).eps * max(m, n) * singular_values[0]
    mask = singular_values > tol
    singular_values = singular_values[mask]

    V = eigenvectors[:, mask]
    U = (A @ V) / singular_values[None, :]

    sigma = np.zeros((m, n))
    k = len(singular_values)
    sigma[:k, :k] = np.diag(singular_values)

    return U, sigma, V.T


def truncated_svd(A, k):
    U, sigma, V = svd(A)
    return U[:, :k], sigma[:k, :k], V[:k, :]


def makeDocumentTermMatrix(data):
    vocab = sorted(set(word for doc in data for word in doc))
    index = {word: i for i, word in enumerate(vocab)}
    M = np.zeros((len(vocab), len(data)))
    for j, doc in enumerate(data):
        for word in doc:
            M[index[word], j] += 1
    return M, vocab


def log_entropy(M):
    n_docs = M.shape[1]
    local = np.log2(M + 1)
    p = M / M.sum(axis=1, keepdims=True)
    plogp = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)
    g = 1 + plogp.sum(axis=1) / np.log2(n_docs)
    return local * g.reshape(-1, 1)


class LSA:
    def __init__(self, k=2):
        self.k = k

    def fit(self, docs):
        self.M, self.vocab = makeDocumentTermMatrix(docs)
        self.X = log_entropy(self.M)
        U, S, Vt = truncated_svd(self.X, self.k)
        self.S, self.Vt = np.diag(S), Vt
        self.words_2d = self.X @ self.Vt.T                   # every word in k-dim concept space

    def predict(self, target_word, top_n=3):
        # top_n words closest (cosine similarity) to target_word
        i = self.vocab.index(target_word)
        v = self.words_2d[i]
        norms = np.linalg.norm(self.words_2d, axis=1) * np.linalg.norm(v) + 1e-12
        sims = self.words_2d @ v / norms
        sims[i] = -np.inf
        best = np.argsort(sims)[::-1][:top_n]
        return [(self.vocab[j], round(float(sims[j]), 3)) for j in best]


text = open("dataset.txt", encoding="utf-8").read()
dataset = ast.literal_eval(text.split("=", 1)[1].strip())

docs = []
for d in dataset:
    w = d["words"]
    docs += [w[i:i + 10] for i in range(0, len(w), 10)]
print("Documents:", len(docs))

model = LSA(k=2)
model.fit(docs)
print("Matrix shape (words x docs):", model.X.shape)
print("Singular values:", model.S.round(3))

target = next((w for w in ["machine", "data"] if w in model.vocab), None)
if target is None:
    target = model.vocab[int(model.M.sum(axis=1).argmax())]
    print('"machine"/"data" not in this corpus, using most frequent word instead.')
print(f"\nTop 3 words closest to '{target}':", model.predict(target))
