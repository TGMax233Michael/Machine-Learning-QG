import numpy as np
import matplotlib.pyplot as plt

X = np.random.randint(0, 100, size=(50, 2))
n_clusters = 2
n_iterations = 100

# random choose two data -> initial centroids
centroids_index = np.random.choice(X.shape[0], n_clusters, replace=False)
centroids = X[centroids_index]
labels = np.zeros(shape=(X.shape[0]))

for i in range(n_iterations):
    # calc distance -> d = sqrt((x1-x2)**2 + (y1-y2)**2)
    distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))

    # get labels
    labels = np.argmin(distances, axis=1)

    # new centroids
    new_centroids = np.array([X[i == labels].mean(axis=0) for i in range(n_clusters)])

    # decide iter
    if np.allclose(new_centroids, centroids, 0, 1e-2):
        break
    
    centroids = new_centroids
    
print(centroids)
print(labels)

# visualization
plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(X[:, 0][labels==0], X[:, -1][labels==0], color="RED")
plt.scatter(X[:, 0][labels==1], X[:, -1][labels==1], color="BLUE")
for i in range(n_clusters):
    plt.scatter(centroids[i, 0], centroids[i, -1], color="GREEN")
plt.show()