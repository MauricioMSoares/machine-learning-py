import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# Generate random data points
x,_ = make_blobs(n_samples=300, centers=4, random_state=42)

#Plot the data points
plt.scatter(x[:,0], x[:,1], s=50)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Data Points")
plt.show()

# Create a K-Means object with k=4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)

# Fit the K-Means to the data
kmeans.fit(x)

# Get the cluster labels assigned to each data point
labels = kmeans.labels_

# Get the clusters centers
centers = kmeans.cluster_centers_

# Plot the data points with cluster assignment
plt.scatter(x[:,0], x[:,1], c=labels, cmap="viridis")

# Plot the cluster centers
plt.scatter(centers[:,0], centers[:,1], marker="o", c="red", s=200, alpha=0.7)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.show()
