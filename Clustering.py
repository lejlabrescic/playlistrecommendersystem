from sklearn.cluster import KMeans, DBSCAN
from statistics import median
import numpy as np

from sklearn.metrics import silhouette_score
def kmeans_clustering(df):
    selected_columns = ['Liveness', 'Energy', 'Danceability', 'Likes', 'Stream']
    X = df[selected_columns].values

    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(X)
    df['KMeans_Cluster'] = clusters
    return df, kmeans


def dbscan_clustering(df):
    selected_columns = ['Liveness', 'Energy', 'Danceability', 'Likes', 'Stream']
    X = df[selected_columns].values

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X)
    df['DBSCAN_Cluster'] = clusters
    return df, dbscan


def recommend_songs(dataset, song_name, num_recommendations):
    data = dataset.copy()

    input_song = data[data['Track'] == song_name].iloc[0]

    selected_columns = ['Liveness', 'Energy', 'Danceability', 'Likes', 'Stream']
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

    return recommended_songs[['Artist', 'Track']];

