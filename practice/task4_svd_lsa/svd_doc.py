import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Sample documents
documents = [
    "Machine learning is a part of artificial intelligence",
    "Deep learning uses neural networks",
    "Machine learning uses algorithms and data",
    "Artificial intelligence is used in robotics",
    "Neural networks are used in deep learning"
]

# Step 2: Convert documents into TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

print("TF-IDF Matrix Shape:", X.shape)

# Step 3: Apply SVD
lsa = TruncatedSVD(n_components=2, random_state=42)
X_lsa = lsa.fit_transform(X)

print("\nDocument Vectors:")
print(X_lsa)

# Step 4: Display topics using important words
terms = vectorizer.get_feature_names_out()

for i, component in enumerate(lsa.components_):
    words = component.argsort()[-5:][::-1]
    print(f"\nTopic {i + 1}:")
    print([terms[j] for j in words])

# Step 5: Calculate document similarity
similarity = cosine_similarity(X_lsa)

print("\nDocument Similarity:")
print(np.round(similarity, 2))

# Step 6: Find most similar document to document 1
scores = similarity[0]
scores[0] = -1  # Exclude document 1 itself

closest = np.argmax(scores)

print("\nDocument 1:")
print(documents[0])

print("\nMost Similar Document:")
print(documents[closest])