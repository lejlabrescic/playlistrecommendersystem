import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
def get_track_genre_for_df(song_uri):
    load_dotenv()
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    track_info = sp.track(song_uri)
    try:
        if track_info['artists']:
            artist_id = track_info['artists'][0]['id']
            artist_info = sp.artist(artist_id)
            if artist_info['genres']:
                print(artist_info['genres'])
                return artist_info['genres']
    except Exception as e:
        print(f"Error fetching genres for track {song_uri}: {e}")

    return None
