# content_based.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movies from u.item
columns = ["movie_id","title","release_date","video_release_date","IMDb_URL",
           "unknown","Action","Adventure","Animation","Children","Comedy",
           "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
           "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]

movies = pd.read_csv("data/u.item", sep="|", names=columns, encoding='latin-1')

# Create a 'genres_text' column combining genres with value 1
genre_cols = columns[5:]
movies['genres_text'] = movies[genre_cols].apply(lambda row: ' '.join([g for g in genre_cols if row[g]==1]), axis=1)

# Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_text'])

# Cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get top N similar movies (content-based)
def get_similar_movies(title, movies_df=movies, cosine_matrix=cosine_sim_matrix, top_n=10):
    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # skip the movie itself
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()
