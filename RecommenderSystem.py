import pandas as pd
import numpy as np
import extractTrackLyrics
import SentimentAnalysis as sa
import Clustering as c
import extractTrackGenre as etg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
df = pd.read_csv("Spotify_Youtube.csv")

df.isnull().sum()
data_cleaned = df.dropna()
df = data_cleaned
df.isnull().sum()

df.drop(columns=["Unnamed: 0"], inplace=True)
df.reset_index(drop=True, inplace=True)

df['song_uri'] = df['Uri'].str.split(':').str[-1]
# print(df)
#df['lyrics'] = df.head(4).apply(lambda row: fetch_lyrics(row['Track'], row['Artist']), axis=1)
#df['genres'] = df['song_uri'].apply(etg.get_track_genre_for_df)

# df['polarity'], df['subjectivity'] = sa.sentiment_analysis(df, 'song_lyrics')


selected_columns = ['Liveness', 'Energy', 'Danceability', 'Likes', 'Stream']
train_data, test_data = train_test_split(df[selected_columns], test_size=0.2, random_state=42)
train_data, kmeans_model = c.kmeans_clustering(train_data)
train_data, dbscan_model = c.dbscan_clustering(train_data)
# sns.pairplot(train_data) 
# plt.show()

song_recommendations = c.recommend_songs(df, 'JUST DANCE HARDSTYLE', 5)
# print("Recommended Songs:")
# print(np.array(song_recommendations))
# print(song_recommendations.columns)

#not forking for some reason 
# ground_truth = df[df['Track'].isin(song_recommendations['Track'])]['Likes'].apply(lambda x: 1 if x > 50 else 0)
#not forking for some reason 
# accuracy = accuracy_score(ground_truth, song_recommendations['Likes'] > 50) 

recommended_songs_df = df[df['Track'].isin(song_recommendations['Track'])][['Track', 'Likes']]
recommended_songs_df['Liked'] = recommended_songs_df['Likes'] > 50
accuracy = recommended_songs_df['Liked'].mean()

# print(f"Accuracy: {accuracy}")

# most_streamed=df.groupby("Track")["Stream"].mean()
# print(most_streamed)

# most_streamed1=most_streamed.nlargest(5)
# print(most_streamed1)

# fig = px.bar(most_streamed1,x=most_streamed1,title="5 Most Streamed Songs")
# fig.show()
# most_streamed_albumtype=df.groupby(["Album_type","Track"])["Stream"].mean().sort_values(ascending=False)
# print(most_streamed_albumtype)
# pd.concat([most_streamed_albumtype.unstack(0).nlargest(5,['album']) ,most_streamed_albumtype.unstack(0).nlargest(5,['single'])]).plot.bar()
# plt.show()

# most_played_artist_spotify=df.groupby("Artist")["Stream"].mean().sort_values(ascending=False)
# print(most_played_artist_spotify)
# most_played_artist_spotify1=most_played_artist_spotify.nlargest(5)

# fig1=px.bar(most_played_artist_spotify1,title="Most played artist on spotify")
# fig1.show()

# f=df.groupby("Artist")[["Stream","Views"]].mean()
# print(f)

# print(f.nlargest(10,["Stream","Views"]))

#cluser evaluation metrics 
# silhouette_avg = silhouette_score(train_data[selected_columns], train_data['KMeans_Cluster'])
# print(f"Silhouette Score: {silhouette_avg}")  #prema rezultatima okej su definisani
# print(f"KMeans Inertia: {kmeans_model.inertia_}") #prema outputu nama je velik broj sto je manji to znaci da su tacke blize cluseru, znaci da su nama daleko

#recommender evaluatin ne valja nam bas clusering ili cu nam daleko ili su oberfitted
# ground_truth = recommended_songs_df['Liked']
# predicted_labels = recommended_songs_df['Likes'] > 50

# precision = precision_score(ground_truth, predicted_labels)
# recall = recall_score(ground_truth, predicted_labels)
# f1 = f1_score(ground_truth, predicted_labels)

# print(f"Precision: {precision}") # rasult: all the instances predicted as positive (Liked) were indeed positive.
# print(f"Recall: {recall}") #result:  model correctly identified all positive instances in the dataset.
# print(f"F1-Score: {f1}") #a perfect balance between precision and recall

#ovi rezultati znace da nam je data set ili nije u balansu pa lase dobije ove pozitivne vjv njih ima vise
#a mozda sam ja samo glupa pa sam koristila isto u treniranju i testiranju :)

# Visualizing recommender results
# fig = px.bar(recommended_songs_df, x='Track', y='Likes', color='Liked', title='Recommender Results')
# fig.show()