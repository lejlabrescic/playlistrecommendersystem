import pandas as pd
import numpy as np
import extractTrackLyrics
import SentimentAnalysis as sa
import Clustering as c
import extractTrackGenre as etg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



df = pd.read_csv("Spotify_Youtube.csv")

df.isnull().sum()
data_cleaned = df.dropna()
df = data_cleaned
df.isnull().sum()

df.drop(columns=["Unnamed: 0"], inplace=True)
df.reset_index(drop=True, inplace=True)

df['song_uri'] = df['Uri'].str.split(':').str[-1]
# print(df)
#df['lyrics'] = df.head(4).apply(lambda row: fetch_lyrics(row['Track'], row['Artist']), axis=1)
#df['genres'] = df['song_uri'].apply(etg.get_track_genre_for_df)

# df['polarity'], df['subjectivity'] = sa.sentiment_analysis(df, 'song_lyrics')


selected_columns = ['Liveness', 'Energy', 'Danceability', 'Likes', 'Stream']
train_data, test_data = train_test_split(df[selected_columns], test_size=0.2, random_state=42)
train_data, kmeans_model = c.kmeans_clustering(train_data)
train_data, dbscan_model = c.dbscan_clustering(train_data)

song_recommendations = c.recommend_songs(df, 'JUST DANCE HARDSTYLE', 5)
# print("Recommended Songs:")
# print(np.array(song_recommendations))
# print(song_recommendations.columns)

# ground_truth = df[df['Track'].isin(song_recommendations['Track'])]['Likes'].apply(lambda x: 1 if x > 50 else 0)

# accuracy = accuracy_score(ground_truth, song_recommendations['Likes'] > 50)

recommended_songs_df = df[df['Track'].isin(song_recommendations['Track'])][['Track', 'Likes']]
recommended_songs_df['Liked'] = recommended_songs_df['Likes'] > 50
accuracy = recommended_songs_df['Liked'].mean()

print(f"Accuracy: {accuracy}")


