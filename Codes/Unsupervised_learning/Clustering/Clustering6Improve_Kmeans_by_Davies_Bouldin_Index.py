#Davies-Bouldin Index (DB Index)

#The Davies-Bouldin Index (DB Index) is another internal evaluation metric for clustering algorithms. It measures the average 
#similarity between each cluster and the cluster that is most similar to it. The similarity is defined as a ratio of the within-cluster 
#distances to the between-cluster distances.

#Formula:
#         N
#DB= 1/N  ∑   max ((si+sj)/dij) 
#        i=1  j≠i

#Where:

#N is the number of clusters.
#si  is the average distance between each point in the i-th cluster and the centroid of that cluster (intra-cluster distance).
#dij  is the distance between the centroids of the i-th and j-th clusters (inter-cluster distance).
#Davies-Bouldin Score Range: The index is always positive, and lower values are better.
#A lower DB index indicates that the clusters are well-separated and compact.
#Unlike inertia, the Davies-Bouldin index does not rely on a visual method, and it tends to give a more objective evaluation of cluster quality.

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score


iris = datasets.load_iris()
X = iris.data
y = iris.target

# Davies-Bouldin Index
def compute_db_index(X, max_k=10):
    db_vals = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        db_vals.append(davies_bouldin_score(X, labels))
    return db_vals

db_vals = compute_db_index(X, max_k=10)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), db_vals, marker='o', color='red')
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('DB Index')
plt.show()