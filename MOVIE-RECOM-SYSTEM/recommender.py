import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("tmdb_5000_movies.csv")

# We use the 'overview' column (movie descriptions)
df = df[["title", "overview"]].dropna()

# -------------------------
# 2. TF-IDF Vectorization
# -------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["overview"])

# -------------------------
# 3. Cosine Similarity
# -------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -------------------------
# 4. Movie index lookup
# -------------------------
indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

# -------------------------
# 5. Recommendation function
# -------------------------
def recommend(movie_title, num_recommendations=5):
    if movie_title not in indices:
        return f"Movie '{movie_title}' not found in dataset."

    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]

    movie_indices = [i[0] for i in sim_scores]
    return df["title"].iloc[movie_indices].tolist()

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    movie = "Tangled"
    print(f"Recommendations for '{movie}':")
    print(recommend(movie))