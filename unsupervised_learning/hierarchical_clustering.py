import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


np.random.seed(0)
x = np.random.rand(20, 2)

clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0).fit(x)

linked = linkage(x, method="ward")
dendrogram(linked, orientation="top")
plt.xlabel("Data points")
plt.ylabel("Distance")
plt.title("Dendrogram")
plt.show()
