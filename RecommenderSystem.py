import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import spotipy
import re
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

# Load variables from the .env file
load_dotenv()
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
df = pd.read_csv("Spotify_Youtube.csv")

df.isnull().sum()
data_cleaned = df.dropna()
df = data_cleaned
df.isnull().sum()

df.drop(columns=["Unnamed: 0"], inplace=True)
df.reset_index(drop=True, inplace=True)
df['song_uri'] = df['Uri'].str.split(':').str[-1]


client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def extract_song_lyrics(description):
    if isinstance(description, str):
        match = re.search(r'Lyrics:(.*)', description, flags=re.DOTALL)

        if match:
            lyrics = match.group(1).strip()
            return lyrics
        match = re.search(r'LYRICS(.*)', description, flags=re.DOTALL)
        if match:
            lyrics = match.group(1).strip()
            return lyrics
    return None

df['song_lyrics'] = df['Description'].apply(extract_song_lyrics)

def get_track_genre_for_df(df):
    genres_list = []
    for index, row in df.iterrows():
        track_uri = row['song_uri']
        try:
            track_info = sp.track(track_uri)
            if track_info['artists']:
                artist_id = track_info['artists'][0]['id']
                artist_info = sp.artist(artist_id)

                if artist_info['genres']:
                    genres_list.append(artist_info['genres'])
                    print(genres_list)
                else:
                    genres_list.append(None)
            else:
                genres_list.append(None)
        except Exception as e:
            print(f"Error fetching genres for track {track_uri}: {e}")
            genres_list.append(None)

    return genres_list


df['track_genre'] = get_track_genre_for_df(df)

#print(df)



from sklearn.cluster import KMeans, DBSCAN
from statistics import median
import numpy as np


def kmeans_clustering(df):
    selected_columns = ['track_genre', 'Energy', 'sentiment_score', 'Likes', 'Stream']
    X = df[selected_columns].values

    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(X)
    df['KMeans_Cluster'] = clusters

    return df, kmeans


def dbscan_clustering(df):
    selected_columns = ['track_genre', 'Energy', 'sentiment_score', 'Likes', 'Stream']
    X = df[selected_columns].values

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X)
    df['DBSCAN_Cluster'] = clusters

    return df, dbscan


def recommend_songs(dataset, song_name, num_recommendations):
    data = dataset.copy()

    input_song = data[data['Track'] == song_name].iloc[0]

    selected_columns = ['track_genre', 'Energy', 'sentiment_score', 'Likes', 'Stream']
    input_features = np.array(input_song[selected_columns]).reshape(1, -1)

    kmeans_model = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    data, kmeans_model = kmeans_clustering(data)
    kmeans_input_cluster = kmeans_model.predict(input_features)[0]

    dbscan_model = DBSCAN(eps=0.5, min_samples=5)
    data, dbscan_model = dbscan_clustering(data)
    dbscan_input_cluster = dbscan_model.fit_predict(input_features)[0]

    consensus_cluster = int(median([kmeans_input_cluster, dbscan_input_cluster]))

    consensus_cluster_data = data[data['KMeans_Cluster'] == consensus_cluster]

    sorted_consensus_cluster_data = consensus_cluster_data.sort_values(by='Likes', ascending=False)

    sorted_consensus_cluster_data = sorted_consensus_cluster_data[sorted_consensus_cluster_data['Track'] != song_name]

    recommended_songs = sorted_consensus_cluster_data.head(num_recommendations)

    return recommended_songs[['Artist', 'Track']]


song_recommendations = recommend_songs(df, 'JUST DANCE HARDSTYLE', 5)
print("Recommended Songs:")
print(np.array(song_recommendations))


