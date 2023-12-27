import pandas as pd
import numpy as np
import extractTrackLyrics
import SentimentAnalysis as sa
import Clustering as c
import extractTrackGenre as etg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
test_data = test_data.reset_index(drop=True)
# print(test_data.columns)
train_data, kmeans_model = c.kmeans_clustering(train_data)
train_data, dbscan_model = c.dbscan_clustering(train_data)

# print(test_data)
#print(c.recommend_songs(test_data, "On Melancholy Hill", 5))

#testing pokusaj 

# song_name_to_test = 'Shape of You'
# num_recommendations = 5

# song_recommendations = c.recommend_songs(test_data, song_name_to_test, num_recommendations)
# print(f"Recommended Songs for '{song_name_to_test}':")
# print(song_recommendations[['Artist', 'Track']])

# sns.pairplot(train_data) 
# plt.show()


song_recommendations = c.recommend_songs(test_data, "On Melancholy Hill", 5)
# print("Recommended Songs:")
# print(np.array(song_recommendations))
# print(song_recommendations.columns)


# recommended_songs_df = df[train_data['Track'].isin(song_recommendations['Track'])][['Track', 'Danceability']]
# recommended_songs_df['Danceability'] = recommended_songs_df['Danceability'] > 50
# accuracy = recommended_songs_df['Danceability'].mean()

# print(f"Accuracy: {accuracy}")

threshold = 0.6
recommended_indices = df[df['Track'].isin(song_recommendations['Track'])].index
# merged_data = pd.merge(df, song_recommendations[['Track']], on='Track', how='inner')

# recommended_indices = merged_data.index


# Use the extracted indices for boolean indexing
true_labels = df.loc[recommended_indices, 'Danceability'] > threshold

predicted_labels = song_recommendations['Danceability'] > threshold
true_labels = true_labels.head(len(predicted_labels))
# min_length = min(len(true_labels), len(predicted_labels))
# true_labels = true_labels.head(min_length)
# predicted_labels = predicted_labels.head(min_length)

precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

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


# Visualizing recommender results
# fig = px.bar(recommended_songs_df, x='Track', y='Likes', color='Liked', title='Recommender Results')
# fig.show()
 

#RMSE 
# X = df[selected_columns]  
# y = df['Likes']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# regression_model = LinearRegression()
# regression_model.fit(X_train, y_train)
# y_pred = regression_model.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)
# print("Mean Absolute Error (MAE):", mae)
# print("Root Mean Squared Error (RMSE):", rmse)