import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# Load ratings
ratings = pd.read_csv(r"data\u.data", sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
svd_model = SVD()
cross_validate(svd_model, data, measures=['RMSE','MAE'], cv=3, verbose=True)

def predict_rating(user_id, movie_id):
    return svd_model.predict(user_id, movie_id).est
