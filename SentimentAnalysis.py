from textblob import TextBlob
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


def getAnalysis(score, task="polarity"):
    if task == "subjectivity":
        if score < 1 / 3:
            return "low"
        elif score > 1 / 3:
            return "high"
        else:
            return "medium"
    else:
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'


def sentiment_analysis(df, text_col):
    df['subjectivity'] = df[text_col].apply(getSubjectivity).apply(lambda x: getAnalysis(x, "subjectivity"))
    df['polarity'] = df[text_col].apply(getPolarity).apply(getAnalysis)
    return df