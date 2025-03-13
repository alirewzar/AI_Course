# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset (replace the path with the correct one if needed)
df = pd.read_csv('./AI/Codes/Unsupervised_learning/Clustering/Mall_Customers.csv')

# Display the first 5 rows of the dataset
print(df.head())

# Drop the 'CustomerID' column as it's irrelevant for clustering
df = df.drop(columns=['CustomerID'])

# Optionally encode the 'Gender' column (1 for Male, 0 for Female)
df['Genre'] = df['Genre'].map({'Male': 1, 'Female': 0})

# Select features to cluster on (e.g., Age, Annual Income, and Spending Score)
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering with 5 clusters initially
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=0)
kmeans.fit(X_scaled)

# Get the cluster labels
labels = kmeans.labels_

# Get the centroids
centroids = kmeans.cluster_centers_

# Inverse transform the centroids back to the original scale for plotting
centroids_original = scaler.inverse_transform(centroids)

# Add the cluster labels to the original dataset for analysis
df['Cluster'] = labels

# Plot clusters using Annual Income and Spending Score and add centroids
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=labels, cmap='rainbow', s=50)
plt.scatter(centroids_original[:, 1], centroids_original[:, 2], c='black', s=200, marker='X', label='Centroids')
plt.title("K-means Clustering (Annual Income vs Spending Score) with Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# Reduce the dataset to 2D using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Reduce the centroids using the same PCA transformation
centroids_pca = pca.transform(centroids)

# Plot the clusters in the reduced 2D space with centroids
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='rainbow', s=50)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='black', s=200, marker='X', label='Centroids')
plt.title("K-means Clustering (PCA Projection) with Centroids")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# 3D visualization using Age, Annual Income, and Spending Score
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot
ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'], c=labels, cmap='rainbow', s=50)

# Plot the centroids in 3D
ax.scatter(centroids_original[:, 0], centroids_original[:, 1], centroids_original[:, 2],
           c='black', s=200, marker='X', label='Centroids')

# Set plot titles and labels
ax.set_title("3D K-means Clustering (Age, Annual Income, Spending Score)")
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
ax.legend()

plt.show()

# Final K-means clustering with the optimal number of clusters (replace 5 with optimal k)
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Add the cluster labels to the original dataset
df['Cluster'] = labels

# Analyze the centroids
centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(scaler.inverse_transform(centroids), columns=X.columns)
centroids_df['Cluster'] = range(len(centroids_df))
print(centroids_df)

# Visualize the centroid values
centroids_df.set_index('Cluster').plot(kind='bar', figsize=(12, 6))
plt.title("Centroid Feature Values for Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Mean Feature Values")
plt.xticks(rotation=0)
plt.show()
