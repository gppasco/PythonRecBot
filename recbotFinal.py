import pandas as pd 
import numpy as np
import os
import sys
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import spotify_secrets

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn import metrics
from scipy.spatial.distance import cosine

import matplotlib.pyplot as plt

print("Done")

if len(sys.argv) != 4:
	raise ValueError("use case: python recbot.py \"[playlist name]\" [number of songs] [new playlist name]")

clientID = spotify_secrets.clientID
clientSecret = spotify_secrets.clientSecret
redirectURL = 'http://localhost:8888/callback/'  # We/Spotify suggest http://localhost:8888/callback/ or http://localhost/

# receive the following warning
# __main__:1: DeprecationWarning: You're using 'as_dict = True'.get_access_token will return the token string directly in future 
#  versions. Please adjust your code accordingly, or use get_cached_token instead.
# At this point, I am taken to the user authorization and grant access with the 'user-top-read' scope

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=clientID,
                                               client_secret=clientSecret,
                                               redirect_uri=redirectURL,
                                               scope="user-library-read user-read-currently-playing playlist-modify-public playlist-read-private user-read-recently-played"))


# Obtain the ID of the playlist given by the user
me = sp.current_user()

playlists = sp.current_user_playlists(limit=30, offset=0)
for playlist in playlists['items']:
	if playlist['name'] == sys.argv[1]:
		playlist_id = playlist['id']

print(f"Loading Playlist (ID: {playlist_id})")

df = pd.DataFrame(columns = ['track_name', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])

# Create a list of (up to) the first 100 songs in the playlist, gather data on songs
# Build a pandas dataframe from the data
results = sp.playlist_tracks(playlist_id=playlist_id, fields=None, limit=100, offset=0, market=None)
for idx, item in enumerate(results['items']):
	audio_feats = sp.audio_features(item['track']['id'])[0]
	df = df.append({'track_id' : item['track']['id'],
		'track_name' : item['track']['name'],
		'danceability' : audio_feats['danceability'],
		'energy' : audio_feats['energy'],
		'key' : audio_feats['key'],
		'loudness' : audio_feats['loudness'],
		'mode' : audio_feats['mode'],
		'speechiness' : audio_feats['speechiness'],
		'acousticness' : audio_feats['acousticness'],
		'instrumentalness' : audio_feats['instrumentalness'],
		'liveness' : audio_feats['liveness'],
		'valence' : audio_feats['valence'],
		'tempo' : audio_feats['tempo']}, ignore_index=True)

print("Done loading playlist")

print(df.head(10)[['track_name', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']])

# ****************************** STANDARDIZING/FITTING DATA *****************************

# Create a df containing only the columns with numerical values
X_orig = (df[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']])
X = MinMaxScaler().fit_transform(X_orig)

# K-means clustering
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# Find the optimal number of clusters (for this playlist, it's 5)
# List inertia 
inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(2, 11), inertias)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

n_clusters = 7

# Now that we have the number of clusters, cluster the data
kmeans = KMeans(
		n_clusters = n_clusters,
		init = "k-means++",
		n_init = 50,
		max_iter = 500)
kmeans.fit(X)

X_orig['cluster'] = kmeans.predict(X)

cluster_centers = kmeans.cluster_centers_

# ****************************** SCORING SONGS *****************************

# Find the "closest" thing

print("Scoring songs...")

all_songs = pd.read_csv('tracks_features.csv')

#all_songs = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv')
all_songs = all_songs[['id', 'name', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
all_songs = all_songs.drop_duplicates()

all_songs = all_songs.sample(1000)

# Remove songs that are already in the sample playlist
all_songs = all_songs.loc[~all_songs.id.isin(df.track_id)]

# Return the cluster assignment for each song in the database
all_songs['cluster'] = kmeans.predict(MinMaxScaler().fit_transform(all_songs.select_dtypes(np.number)))

all_songs['distance'] = ""

for index, row in all_songs.iterrows():
	row_vals = [row['danceability'], row['energy'], row['loudness'], row['speechiness'], row['acousticness'], row['instrumentalness'], row['liveness'], row['valence'], row['tempo']]
	all_songs.loc[index, 'distance'] = cosine(row_vals, cluster_centers[row['cluster']])

print("Done scoring songs")

# ****************************** FINDING RECOMMENDATIONS *****************************

print("Finding recommendations...")
num_songs = int(sys.argv[2])
all_songs = all_songs.sort_values(by='distance')

recs = all_songs.head(num_songs)

print(recs)

print("Done finding recommendations")

# ****************************** CREATING PLAYLIST *****************************
playlist_name = sys.argv[3]
playlist_description = f"A playlist created by RecBot. Based on the playlist \"{sys.argv[1]}\""

playlist = sp.user_playlist_create(me['display_name'], playlist_name, public=True, collaborative = False, description =playlist_description)

sp.user_playlist_add_tracks(me['display_name'], playlist['id'], list(recs['id']))