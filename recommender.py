from hybrid_recommender import hybrid_recommend_user


if __name__ == "__main__":
    movie_name = 'Star Wars (1977)'
    user_id = 1  # demo user
    recommendations = hybrid_recommend_user(movie_name, user_id)
    
    print(f"Top {len(recommendations)} hybrid recommendations for user {user_id} based on '{movie_name}':")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Optional: save to CSV
    import pandas as pd
    df = pd.DataFrame(recommendations, columns=['Recommended Movies'])
    df.to_csv('hybrid_recommendations.csv', index=False)
    print("Recommendations saved to hybrid_recommendations.csv")
