# This covers an example of an underfitting model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Generate non-linear data
np.random.seed(0)
x = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)

# Fit a Linear Regression model
model = LinearRegression()
model.fit(x, y)

# Predict on training data
y_pred = model.predict(x)

# Plot the data and the linear regression line
plt.scatter(x, y, color="b", label="Actual")
plt.plot(x, y_pred, color="r", label="Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Underfitting Model")
plt.legend()
plt.show()

# -------------------- Fixing Method -------------------- #

from sklearn.preprocessing import PolynomialFeatures

# Generate Polynomial Features
poly_features = PolynomialFeatures(degree = 3)
x_poly = poly_features.fit_transform(x)

# Fit a polynomial regression model
model = LinearRegression()
model.fit(x_poly, y)

y_pred = model.predict(x_poly)

plt.scatter(x, y, color="b", label="Actual")
plt.plot(x, y_pred, color="r", label="Polynomial Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Underfitting Model Fixed - Polynomial Regression")
plt.legend()
plt.show()
