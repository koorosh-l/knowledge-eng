import stumpy
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.neighbors import KNeighborsClassifier
import sklearn.neighbors

m = 640
steam_df = pd.read_csv("https://zenodo.org/record/4273921/files/STUMPY_Basics_steamgen.csv?download=1")
mp = stumpy.stump(steam_df["steam flow"], m)

knn = sklearn.neighbors.NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(mp)

dist, ix = knn.kneighbors(mp)

# for i in range(len(mp)):
#     print(f"p {i}: {ix[i]}")
#     print(f"dist {i}: {dist[i]}")
# figure out a way to use dist and ix as features to(probably just passing dist)

# import tensorflow as tf

# x = dist.reshape(-1, 1)

# dataset = tf.data.Dataset.from_tensor_slices(dist)

# kmeans = tf.keras.layers.Normalization()
# kmeans.adapt(dataset)
# clusters = kmeans(dataset)
