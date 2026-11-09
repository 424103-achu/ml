
# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv('udemy_courses_with_student_ids_and_ratings.csv')
R = df.pivot(index='student_id', columns='course_id', values='course_rating')
R.head()


# %%
mean = R.mean(axis=1)
R_new = R.sub(mean, axis=0).fillna(0)

# %%
U, S, Vt = np.linalg.svd(R_new, full_matrices=False)

k = 2
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]

R_new1 = np.dot(np.dot(U_k, S_k), Vt_k)
R_new2 = pd.DataFrame(R_new1, index=R.index,
                      columns=R.columns).add(mean, axis=0)

course_titles = (
    df[["course_id", "course_title"]].drop_duplicates().set_index("course_id")[
        "course_title"
    ]
)

# %%


def recommend_top_courses(student_id, top_n=3):

    student_scores = R_new2.loc[student_id]
    interacted_courses = R.loc[student_id].dropna().index
    unseen_scores = student_scores.drop(labels=interacted_courses)
    top_course_ids = unseen_scores.sort_values(ascending=False).head(top_n)

    recommendations = pd.DataFrame(
        {
            "Course ID": top_course_ids.index,
            "Course Title": top_course_ids.index.map(course_titles),
            "Predicted Rating": top_course_ids.values.round(2),
        }
    )

    return recommendations


recommendations = recommend_top_courses('STU01FBPRI1', top_n=3)
print(recommendations)
