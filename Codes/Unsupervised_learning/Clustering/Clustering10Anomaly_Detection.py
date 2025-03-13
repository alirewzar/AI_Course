from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data

# Standardize the data
X_scaled = StandardScaler().fit_transform(X)

# Apply DBSCAN for anomaly detection
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_scaled)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters_dbscan, cmap='viridis')
plt.title('Anomaly Detection using DBSCAN')
plt.xlabel('Standardized Feature 1')
plt.ylabel('Standardized Feature 2')
plt.colorbar(label='Cluster')
plt.show()
