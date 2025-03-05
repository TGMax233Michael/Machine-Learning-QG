import numpy as np

class KMeans:
    def __init__(self, n_clusters, n_epoches=100, threhold=1e-2):
        self.n_clusters = n_clusters
        self.n_epoches = n_epoches
        self.threhold = threhold
    
    def _init_centroids(self, X: np.ndarray):
        return X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
    
    def _calc_distance(self, X: np.ndarray, centroids: np.ndarray):
        return np.sqrt(np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2))
    
    def fit(self, X: np.ndarray):
        centroids = self._init_centroids(X)
        labels = np.zeros(shape=(X.shape[0]))
        
        for epoch in range(self.n_epoches):
            distances = self._calc_distance(X, centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([np.mean(X[i == labels], axis=0) for i in range(self.n_clusters)])
            
            if np.allclose(new_centroids, centroids, rtol=0, alol=self.threhold):
                break
            centroids = new_centroids
            
        return centroids, labels