import streamlit as st
import pandas as pd
from hybrid_recommender import hybrid_recommend_user
from content_based import movies  # to show genres

# Title
st.title("🎬 Netflix-Style Hybrid Movie Recommender")

# Sidebar: Select user ID and movie
st.sidebar.header("User Inputs")
user_id = st.sidebar.number_input("Select User ID", min_value=1, max_value=943, value=1)
movie_name = st.sidebar.selectbox("Select a Movie", movies['title'].tolist())

top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# Button to get recommendations
if st.sidebar.button("Get Recommendations"):
    recommendations = hybrid_recommend_user(movie_name, user_id, top_n=top_n)
    
    st.subheader(f"Top {top_n} Hybrid Recommendations for User {user_id} based on '{movie_name}'")
    rec_df = pd.DataFrame(recommendations, columns=['Recommended Movies'])
    st.dataframe(rec_df)
    
    # Add chart: content similarity vs predicted rating
    st.subheader("Recommendation Scores (Hybrid)")
    sim_scores = []
    pred_scores = []
    for title in recommendations:
        # Get content similarity
        content_score = 0
        movie_row = movies[movies['title'] == title]
        if not movie_row.empty:
            idx = movie_row.index[0]
            content_score = 1  # can adjust if using full cosine similarity matrix
        sim_scores.append(content_score)
        
        # Predicted rating
        pred = hybrid_recommend_user.movie_prediction(user_id, title) if hasattr(hybrid_recommend_user,'movie_prediction') else 4
        pred_scores.append(pred)
    
    chart_df = pd.DataFrame({
        'Movie': recommendations,
        'Content Similarity': sim_scores,
        'Predicted Rating': pred_scores
    }).set_index('Movie')
    
    st.bar_chart(chart_df)
