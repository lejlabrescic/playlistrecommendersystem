import re
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