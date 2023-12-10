from textblob import TextBlob
def getSubjectivity(lyrics):
  return TextBlob(lyrics).sentiment.subjectivity
def getPolarity(lyrics):
  return TextBlob(lyrics).sentiment.polarity
def getAnalysis(score, task="polarity"):
  if task == "subjectivity":
    if score < 1/3:
      return "low"
    elif score > 1/3:
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

def sentiment_analysis(df, lyrics):
  df['subjectivity'] = df[lyrics].apply(getSubjectivity).apply(lambda x: getAnalysis(x,"subjectivity"))
  df['polarity'] = df[lyrics].apply(getPolarity).apply(getAnalysis)
  return df['polarity'], df['subjectivity']
