from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from model.cluster import MyKMeans
import numpy as np
import matplotlib.pyplot as plt

data = make_blobs(n_samples=1000, n_features=2, centers=10, random_state=42)

X, y = data[0], data[-1]

model01 = MyKMeans(n_clusters=5, n_epoches=100, threhold=1e-5)
centroids, labels01 = model01.fit(X)

model02 = KMeans(n_clusters=5)
model02.fit(X)
labels02 = model02.predict(X)

# visualization
plt.figure(figsize=(8, 8), dpi=80)

color = ["Red", "Blue", "Purple", "Orange", "Pink"]
for i in range(5):
    plt.scatter(X[:, 0][labels01==i], X[:, -1][labels01==i], color=color[i])
    
plt.figure(figsize=(8, 8), dpi=80)
for i in range(5):
    plt.scatter(X[:, 0][labels02==i], X[:, -1][labels02==i], color=color[i])
plt.show()

