import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# Generating a synthetic dataset
np.random.seed(0)
x = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = [0] * 20 + [1] * 20

# Visualizing the dataset
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# Training the model
clf = svm.SVC(kernel="linear")
clf.fit(x, y)

# Plotting the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot decision boundary
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# Plotting support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Make a prediction
np.random.seed(1)
x_test = np.random.randn(10, 2)
y_prediction = clf.predict(x_test)
print("Predictions: ", y_prediction)
