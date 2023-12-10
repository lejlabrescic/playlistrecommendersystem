import pandas as pd
import numpy as np
import extractTrackLyrics
import SentimentAnalysis as sa
import Clustering as c
import extractTrackGenre as etg

df = pd.read_csv("Spotify_Youtube.csv")

df.isnull().sum()
data_cleaned = df.dropna()
df = data_cleaned
df.isnull().sum()

df.drop(columns=["Unnamed: 0"], inplace=True)
df.reset_index(drop=True, inplace=True)

df['song_uri'] = df['Uri'].str.split(':').str[-1]
df['song_lyrics'] = df.apply(lambda row: extractTrackLyrics.fetch_lyrics(row['Track'], row['Artist']), axis=1)
df['track_genre'] = etg.get_track_genre_for_df(df)
df['polarity'], df['subjectivity'] = sa.sentiment_analysis(df, 'song_lyrics')

song_recommendations = c.recommend_songs(df, 'JUST DANCE HARDSTYLE', 5)
print("Recommended Songs:")
print(np.array(song_recommendations))


