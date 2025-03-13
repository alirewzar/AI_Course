import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

home_data = pd.read_csv('./AI/Codes/Unsupervised_learning/Clustering/housing.csv', usecols = ['longitude', 'latitude', 'median_house_value'])
print(home_data.head())

sns.scatterplot(data = home_data, x = 'longitude', y = 'latitude', hue = 'median_house_value')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(home_data[['latitude', 'longitude']], home_data[['median_house_value']], test_size=0.33, random_state=0)

X_train_norm = normalize(X_train)
X_test_norm = normalize(X_test)

#Fitting and Evaluating the Model
#For the first iteration, we will arbitrarily choose a number of clusters (referred to as k) of 3. 
#Building and fitting models in sklearn is very simple. We will create an instance of KMeans, define 
#the number of clusters using the n_clusters attribute, set n_init, which defines the number of iterations 
#the algorithm will run with different centroid seeds, to “auto”, and we will set the random_state to 0 so 
#we get the same result each time we run the code. We can then fit the model to the normalized training data using the fit() method.

kmeans = KMeans(n_clusters = 3, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)

sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = kmeans.labels_)
plt.show()

sns.boxplot(x = kmeans.labels_, y = y_train['median_house_value'], hue=kmeans.labels_)
plt.show()


#Choosing the best number of clusters
#The weakness of k-means clustering is that we don’t know how many clusters we need by just running the model. 
#We need to test ranges of values and make a decision on the best value of k. We typically make a decision using 
#the Elbow method to determine the optimal number of clusters where we are both not overfitting the data 
#with too many clusters, and also not underfitting with too few.

K = range(2, 8)
fits = []

for k in K:
    # train the model for current value of k on training data
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(X_train_norm)

    # append the model to fits
    fits.append(model)

# Create figure with 2 rows and 3 columns
f, axes = plt.subplots(2, 3, figsize=(18, 10))
f.suptitle('K-Means Clustering', fontsize=24)

# Assuming fits is a list of KMeans models (length 6)
# Plotting each clustering result in its respective subplot
sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=fits[0].labels_, ax=axes[0,0])
sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=fits[1].labels_, ax=axes[0,1])
sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=fits[2].labels_, ax=axes[0,2])
sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=fits[3].labels_, ax=axes[1,0])
sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=fits[4].labels_, ax=axes[1,1])
sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=fits[5].labels_, ax=axes[1,2])
# Optional improvements
# Adjust layout to prevent overlap
plt.tight_layout()
# Add padding between title and plots
f.subplots_adjust(top=0.9)
# Display the plot
plt.show()

# Calculate inertia (within-cluster sum of squares) for different k values
inertia = []
k_range = range(1, 11)  # Testing k from 1 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans.fit(X_train_norm)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

#To determine the optimal number of clusters, we have to select the value of k at the “Elbow”. 
#In the next two section, we have explained in detail about the elbow. Thus for the given data, 
#we conclude that the optimal number of clusters for the data is 5. We see k = 5 is probably 
#the best we can do without overfitting.
kmeans = KMeans(n_clusters = 5, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)

# Create a figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # Adjust figsize as needed
fig.suptitle('K-Means Clustering Analysis', fontsize=16)

# Plot 1: Scatter plot on the first axis
sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=kmeans.labels_, ax=ax1)
ax1.set_title('Cluster Distribution')

# Plot 2: Box plot on the second axis
sns.boxplot(x=kmeans.labels_, y=y_train['median_house_value'], hue=kmeans.labels_, ax=ax2)
ax2.set_title('Median House Value by Cluster')

# Adjust layout to prevent overlap
plt.tight_layout()
fig.subplots_adjust(top=0.85)  # Make room for the suptitle

# Display the plots
plt.show()