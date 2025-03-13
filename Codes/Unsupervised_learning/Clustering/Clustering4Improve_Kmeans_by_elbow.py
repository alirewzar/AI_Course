#Elbow Method

#The Elbow Method is one of the most commonly used techniques to determine the optimal number of clusters (K) 
#in K-means clustering. It is based on inertia, which measures the sum of squared distances between each 
#data point and its nearest centroid.
#Inertia decreases as the number of clusters increases. This is because, with more clusters, each point is 
#likely to be closer to its assigned centroid.
#The goal is to find the "elbow" in the plot of inertia vs. K. The elbow point represents the point where 
#adding more clusters does not result in a significant decrease in inertia. This indicates that further 
#increasing K is not improving the model substantially.
#The point where the inertia curve starts flattening is considered the optimal number of clusters.

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

iris = datasets.load_iris()
X = iris.data
y = iris.target

plt.scatter(X[:, 0], X[:, 1])
plt.show()

# Inertia (Elbow Method)
def compute_inertia(X, max_k=10):
    inertia_vals = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia_vals.append(kmeans.inertia_)
    return inertia_vals

inertia_vals = compute_inertia(X, max_k=10)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia_vals, marker='o', color="red")
plt.axvline(x=3, ls='--')
plt.title('Inertia (Elbow Method)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()