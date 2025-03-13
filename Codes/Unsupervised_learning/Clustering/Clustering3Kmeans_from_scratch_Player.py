import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from IPython.display import clear_output

players = pd.read_csv("./AI/Codes/Unsupervised_learning/Clustering/players_22.csv")
display(players.head())

features = ["overall", "potential", "wage_eur", "value_eur", "age"]

players = players.dropna(subset=features)

data = players[features].copy()
display(data.head())

#Min-Max Normalization: Min-max normalization is one of the most common ways to normalize 
#data. For every feature, the minimum value of that feature gets transformed into a 0, the 
#maximum value gets transformed into a 1, and every other value gets 
#transformed into a decimal between 0 and 1.
data = ((data - data.min()) / (data.max() - data.min()))

display('\n \n', data.describe())


#There are various methods of assigning  k  - centroid initially. We used random selection for initialing centroids.
def random_centroids(data, k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)

centroids = random_centroids(data, 5)

display(centroids)

#We find the euclidean distance from each point to all the centroids.
#d(p,q)= √((q1−p1)^2+(q2−p2)^2)
#Now, In order to know which cluster an data items belongs to, we are calculating 
#euclidean distance from the data items to each centroid. Data item closest to the 
#cluster belongs to that respective cluster. If each cluster centroid is denoted by  ci , 
#then each data point  x  is assigned to a cluster based on argmin dist(ci,x)^2

def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)

labels = get_labels(data, centroids)

display(labels.value_counts())

#Finding the new centroid from the clustered group of points:
def new_centroids(data, labels, k):
    centroids = data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids

#We've now completed the K Means scratch code of this Machine Learning tutorial series. Now lets test our code by clustering:
def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.figure(figsize=(10, 10))
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
    plt.show()

max_iterations = 100
centroid_count = 3

centroids = random_centroids(data, centroid_count)
old_centroids = pd.DataFrame()
iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids

    labels = get_labels(data, centroids)
    centroids = new_centroids(data, labels, centroid_count)
    plot_clusters(data, labels, centroids, iteration)
    iteration += 1

display(centroids)

kmeans = KMeans(3)
kmeans.fit(data)