from nltk.sentiment.vader import SentimentIntensityAnalyzer
def sentimentAnalysis(df):
    sid = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['song_lyrics'].apply(lambda x: sid.polarity_scores(x)['compound']);
    return df['sentiment_score']