# hybrid_recommender.py
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split
from content_based import movies, get_similar_movies

# Load ratings from u.data
ratings = pd.read_csv("data/u.data", sep="\t", names=['user_id','movie_id','rating','timestamp'])

# Prepare Surprise dataset
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['user_id','movie_id','rating']], reader)
trainset = data.build_full_trainset()

# Train SVD
algo = SVD()
algo.fit(trainset)

# Hybrid recommender: content-based + collaborative filtering
def hybrid_recommend_user(movie_title, user_id, top_n=10):
    # Step 1: Get content-based similar movies
    similar_movies = get_similar_movies(movie_title, top_n=top_n*2)  # get extra to filter later
    
    # Step 2: Predict ratings for the user
    predictions = []
    for title in similar_movies:
        movie_id = movies[movies['title'] == title]['movie_id'].values[0]
        pred = algo.predict(user_id, movie_id).est
        predictions.append((title, pred))
    
    # Step 3: Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N titles
    top_recs = [p[0] for p in predictions[:top_n]]
    return top_recs
