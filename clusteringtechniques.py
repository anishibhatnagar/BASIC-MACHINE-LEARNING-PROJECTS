# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------
# 1. KMeans Clustering
# ----------------------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Visualization of KMeans
plt.figure(figsize=(8, 6))
for label in np.unique(kmeans_labels):
    plt.scatter(X_scaled[kmeans_labels == label, 0], X_scaled[kmeans_labels == label, 1], label=f'Cluster {label}')
plt.title('KMeans Clustering on Iris Dataset')
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Sepal Width (standardized)')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------
# 2. Hierarchical Clustering
# ----------------------------------------

# Dendrogram
plt.figure(figsize=(10, 7))
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram (Iris)')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.grid(True)
plt.show()

# Agglomerative Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Visualization of Hierarchical Clustering
plt.figure(figsize=(8, 6))
for label in np.unique(hierarchical_labels):
    plt.scatter(X_scaled[hierarchical_labels == label, 0], X_scaled[hierarchical_labels == label, 1], label=f'Cluster {label}')
plt.title('Hierarchical Clustering on Iris Dataset')
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Sepal Width (standardized)')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------
# 3. DBSCAN Clustering
# ----------------------------------------

dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Visualization of DBSCAN
plt.figure(figsize=(8, 6))
unique_labels = np.unique(dbscan_labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    label_name = f'Cluster {label}' if label != -1 else 'Noise'
    plt.scatter(X_scaled[dbscan_labels == label, 0], X_scaled[dbscan_labels == label, 1],
                label=label_name, c=[color], edgecolor='k')
plt.title('DBSCAN Clustering on Iris Dataset')
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Sepal Width (standardized)')
plt.legend()
plt.grid(True)
plt.show()
