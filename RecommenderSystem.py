import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from sklearn.cluster import KMeans, DBSCAN
from statistics import median
import numpy as np
import spotipy
import re
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

import SentimentAnalysis as sa
import Clustering as c
import extractTrackUri as etu
import extractTrackGenre as etg

df = pd.read_csv("Spotify_Youtube.csv")

df.isnull().sum()
data_cleaned = df.dropna()
df = data_cleaned
df.isnull().sum()

df.drop(columns=["Unnamed: 0"], inplace=True)
df.reset_index(drop=True, inplace=True)
df['song_uri'] = df['Uri'].str.split(':').str[-1]






df['song_lyrics'] = etu.extract_song_lyrics(df['Description'])


df['track_genre'] = etg.get_track_genre_for_df(df)
df['sentiment_score'] = sa.sentimentAnalysis(df)


song_recommendations = c.recommend_songs(df, 'JUST DANCE HARDSTYLE', 5)
print("Recommended Songs:")
print(np.array(song_recommendations))


