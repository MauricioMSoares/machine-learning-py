import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
print(iris.head())

x = iris.iloc[:,:-1].values
y = iris.iloc[:,-1].values

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)
print(x_pca)

le = LabelEncoder()
y_num = le.fit_transform(y)

plt.scatter(x_pca[:,0], x_pca[:,1], c=y_num, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
