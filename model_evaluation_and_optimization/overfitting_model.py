from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Generate a synthetic dataset with 2 classes (moon)
x, y = make_moons(n_samples=200, noise=0.3, random_state=42)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fit a decision tree classifier with a high max_depth (overfitting)
model = DecisionTreeClassifier(max_depth=10)
model.fit(x_train, y_train)

# Predict on training and test data
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Train accuracy: ", train_accuracy)
print("Test accuracy: ", test_accuracy)

# -------------------- Fixing Method -------------------- #

# Fit a decision tree classifier with a lower max_depth (regularization)
model = DecisionTreeClassifier(max_depth=3)
model.fit(x_train, y_train)

# Predict on training and test data
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Train accuracy: ", train_accuracy)
print("Test accuracy: ", test_accuracy)
