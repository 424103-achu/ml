import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ML/task4"))  # data files live here
import numpy as np
import pandas as pd


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


class SVDRecommender:
    def __init__(self, k=2):
        self.k = k

    def fit(self, R):
        # R = students x courses rating matrix (NaN = not rated)
        # (every student here rated only ONE course, so subtracting the student's
        #  mean would make the matrix all zeros -> use raw ratings, 0 = not rated)
        self.R = R
        self.U_k, self.S_k, self.Vt_k = truncated_svd(R.fillna(0).values, self.k)
        R_hat = self.U_k @ self.S_k @ self.Vt_k
        self.R_hat = pd.DataFrame(R_hat, index=R.index, columns=R.columns)

    def predict(self, student_id, top_n=3):
        # top_n unseen courses with the highest predicted rating
        scores = self.R_hat.loc[student_id]
        seen = self.R.loc[student_id].dropna().index
        return scores.drop(seen).sort_values(ascending=False).head(top_n)


df = pd.read_csv("udemy_courses_with_student_ids_and_ratings.csv")

R = df.pivot_table(index="student_id", columns="course_id", values="course_rating")
print("Matrix R shape (students x courses):", R.shape)

model = SVDRecommender(k=2)
model.fit(R)
print("U_k:", model.U_k.shape, " S_k:", model.S_k.shape, " Vt_k:", model.Vt_k.shape)

titles = df.drop_duplicates("course_id").set_index("course_id")["course_title"]


def recommend_top_courses(student_id, top_n=3):
    scores = model.predict(student_id, top_n)
    return pd.DataFrame({"course_title": titles[scores.index].values,
                         "predicted_score": scores.values.round(3)})


student = R.index[0]
print("\nTop 3 courses for", student)
print(recommend_top_courses(student, top_n=3))
