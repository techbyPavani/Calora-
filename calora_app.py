# calora_app.py
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from hybrid_recommender import hybrid_recommend_user
from content_based import movies

# Page setup
st.set_page_config(page_title="Calora - Hybrid Recommender", layout="wide")
st.title("🎬 Calora - Hybrid Movie Recommender")
st.markdown("Discover movies you'll love, just like Netflix!")

# Sidebar
st.sidebar.header("User Inputs")
user_id = st.sidebar.number_input("Select User ID", min_value=1, max_value=943, value=1)
movie_name = st.sidebar.selectbox("Select a Movie", movies['title'].tolist())
top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# Function to generate a placeholder poster
def generate_poster(title, width=200, height=300):
    img = Image.new('RGB', (width, height), color=(30,30,30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    # Wrap text
    lines = []
    words = title.split()
    line = ""
    for word in words:
        if len(line + word) < 15:
            line += word + " "
        else:
            lines.append(line.strip())
            line = word + " "
    lines.append(line.strip())
    
    # Draw text
    y_text = height//2 - len(lines)*10
    for line in lines:
        w, h = draw.textbbox((0,0), line, font=font)[2:]
        draw.text(((width - w)/2, y_text), line, font=font, fill=(255,255,255))
        y_text += 20
    return img

# Recommendations
if st.sidebar.button("Get Recommendations"):
    recommendations = hybrid_recommend_user(movie_name, user_id, top_n=top_n)
    
    # Display as Netflix-style cards
    st.subheader(f"Top {top_n} Hybrid Recommendations for User {user_id}")
    rows = (len(recommendations) + 4)//5
    for r in range(rows):
        cols = st.columns(5)
        for i in range(5):
            idx = r*5 + i
            if idx >= len(recommendations):
                break
            title = recommendations[idx]
            col = cols[i]
            col.image(generate_poster(title), use_container_width=True)
            col.markdown(f"**{title}**")
            # Genre display
            genre = movies[movies['title']==title]['genres_text'].values[0]
            col.markdown(f"*{genre}*")
            # Predicted rating
            movie_id = movies[movies['title']==title]['movie_id'].values[0]
            from hybrid_recommender import algo
            rating = round(algo.predict(user_id, movie_id).est, 1)
            col.markdown(f"⭐ Predicted Rating: {rating}")

    # Bar chart of predicted ratings
    pred_ratings = []
    for title in recommendations:
        movie_id = movies[movies['title']==title]['movie_id'].values[0]
        pred_ratings.append(round(algo.predict(user_id, movie_id).est,1))
    
    chart_df = pd.DataFrame({'Movie': recommendations, 'Predicted Rating': pred_ratings}).set_index('Movie')
    st.subheader("Overall Predicted Ratings")
    st.bar_chart(chart_df)
