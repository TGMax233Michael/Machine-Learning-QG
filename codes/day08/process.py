import numpy as np

X = np.array([[1, 2], 
              [1.5, 1.8], 
              [5, 8], 
              [8, 8], 
              [1, 0.6], 
              [9, 11]])

centroids_index = np.random.choice(X.shape[0], 2, replace=False)
centroids = X[centroids_index]
# print(centroids)
# print(centroids.shape)

# print(X[:, np.newaxis])
# print(X[:, np.newaxis].shape)

# Calc distance
distances = np.sqrt(np.sum((X[:, np.newaxis]-centroids) ** 2, axis=2))

# print(f"distances:\n{distances}")
labels = np.argmin(distances, axis=1)
# print(f"labels: {labels}")

new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(2)])

print(centroids)
print(new_centroids)

# Whether the threshold is reached
# |a - b| < (atol + rtol * |b|)
# when rtol == 0 -> |a - b| < atol
print(np.allclose(new_centroids, centroids, atol=2, rtol=0)) 