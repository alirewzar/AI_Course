#Silhouette Score

#The Silhouette Score measures how well each data point fits within its assigned cluster compared to 
#how well it fits in other clusters. The silhouette score for a data point is calculated as:
#S=(b−a)/max(a,b) 
   
#Where:

#a is the mean intra-cluster distance (the average distance between a point and other points within the same cluster).
#b is the mean nearest-cluster distance (the average distance between a point and the points in the nearest cluster).
#Silhouette Score Range: The score ranges from -1 to 1.
#A value close to 1 indicates that the point is well clustered.
#A value close to 0 means that the point is on or very close to the decision boundary between clusters.
#A negative value indicates that the point might have been assigned to the wrong cluster.
#The optimal K is the one that maximizes the silhouette score, indicating that the clusters are well-separated and compact.

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

iris = datasets.load_iris()
X = iris.data
y = iris.target

plt.scatter(X[:, 0], X[:, 1])
plt.show()

sil = []
max_k = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, max_k+1):
  kmeans = KMeans(n_clusters = k)
  kmeans.fit(X)
  labels = kmeans.labels_
  sil.append(silhouette_score(X, labels, metric = 'euclidean'))

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sil, marker='o')
plt.title('Silhouette Score For Different K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()

fig, ax = plt.subplots(4, 2, figsize=(15,8))
for k in range(2, 10):
    #Create KMeans instance for different number of clusters
    km = KMeans(n_clusters=k, random_state=42)
    q, mod = divmod(k, 2)

    #Create SilhouetteVisualizer instance with KMeans instance
    #Fit the visualizer
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(X)

plt.show()

'''
A Silhouette Plot is a graphical representation of the silhouette coefficient for each data point. It shows:

Silhouette width for each point: A wider silhouette indicates better clustering for that point.
Separation between clusters: The plot shows how distinct the clusters are from one another. Large gaps between 
clusters suggest better-defined clusters.

In a Silhouette Plot:

Each cluster is represented by a horizontal bar of silhouette scores for the points in that cluster.
The points are ordered by silhouette score within each cluster.
A red dashed line shows the average silhouette score for all points.
Interpreting Silhouette Analysis

Positive Silhouette Coefficients: A positive silhouette score indicates that the data point is well within the boundary of its cluster.
Negative Silhouette Coefficients: A negative silhouette score means the point is likely in the wrong cluster.
Cluster Separation: The gaps between the clusters in the silhouette plot indicate how distinct the clusters are. Large gaps suggest 
better-defined clusters, while overlapping bars suggest poorly separated clusters.
Choosing the Optimal Number of Clusters: Silhouette analysis can help choose the optimal number of clusters by comparing 
the average silhouette score for different numbers of clusters (K). The best K is typically the one with the highest silhouette score.
When to Use Silhouette Analysis

Cluster Quality Evaluation: It is used to assess the quality of clustering and determine whether the clusters are well-separated and cohesive.
Determining Optimal Number of Clusters: By plotting the silhouette score for different values of K (e.g., in K-means), you can 
identify the number of clusters that maximizes the average silhouette score.
Example of Silhouette Analysis in K-means

Let’s assume you have clustered data using K-means with different numbers of clusters. After applying Silhouette Analysis, you might see:

High silhouette score (close to +1): Clusters are well-defined, with little overlap.
Low silhouette score (close to 0): Clusters are not well-separated; some data points are likely to be near the cluster boundaries.
Negative silhouette score: Some data points are potentially assigned to the wrong clusters.
In practice, Silhouette Analysis is a valuable tool for understanding the clustering structure of the data and for determining 
the optimal number of clusters.
'''
