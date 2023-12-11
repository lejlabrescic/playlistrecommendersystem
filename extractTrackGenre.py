import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os


def get_track_genre_for_df(df):
    load_dotenv()
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
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

    genres_list = df['song_uri'].apply(get_track_genre_for_df)
    return genres_list
