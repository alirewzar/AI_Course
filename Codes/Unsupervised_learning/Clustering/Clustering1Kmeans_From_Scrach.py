import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:,:2]

def Visualize(X, centroids, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x')
    plt.show()
    
def kmeans(X, k, iterations=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(iterations):
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        Visualize(X, centroids, labels)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

k = 3
centroids, labels = kmeans(X, k)