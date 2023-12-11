import lyricsgenius
from dotenv import load_dotenv
import os


def fetch_lyrics(track_name, artist):
    load_dotenv()
    genius_api_token = os.getenv("GENIUS_API_TOKEN")
    try:
        genius = lyricsgenius.Genius(genius_api_token)
        song = genius.search_song(track_name, artist)
        if song:
            return song.lyrics
        else:
            return None
    except Exception as e:
        print("An error occurred while fetching lyrics: ", e)
        return None
