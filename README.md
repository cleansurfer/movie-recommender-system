# movie-recommender-system
A content-based movie recommendation system using Python and TF-IDF.

-This project implements a content-based movie recommendation system using
Python, Pandas, Scikit-Learn, TF-IDF vectorization, and Cosine Similarity.
-It predicts what movies a user is likely to enjoy based on past behavior and movie similarity



---> How it works:
- The model reads a dataset of movies
- Converts movie descriptions into TF-IDF vectors
- Computes cosine similarity between movies
- Recommends movies based on similarity of descriptions



---> Features:
1. Content-Based Filtering:
   - Recommends movies based on Genres / Keywords / Movie descriptions / TF-IDF similarity
2. Collaborative Filtering:
   - Using User–item interaction matrix / Cosine similarity
3. Hybrid Recommendation System
   - Combines content + collaborative for better results
4. Data Preprocessing:
   - Cleaning text
   - Removing duplicates
   - Vectorization (TF-IDF)
   - Movie metadata normalization




---> Technology:
- Python
- Pandas
- Scikit-Learn
- TF-IDF Vectorizer
- Cosine Similarity




---> How to run the project:
1. Install dependencies:



---> Future works:
- Add deep-learning recommendations (Neural Collaborative Filtering)
- Add user-personalized dashboard
- Deploy API using FastAPI
