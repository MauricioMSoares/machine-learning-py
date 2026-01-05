import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV


# Load the dataset
data = load_breast_cancer()
x = data.data
y = data.target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10]
}

# Create the random forest classifier instance
clf = RandomForestClassifier()

# Perform Grid search using cross validation
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring="accuracy")
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the model using the best hyperparameters
clf_best = RandomForestClassifier(**best_params)
clf_best.fit(x_train, y_train)

# Evaluate the model on the test set
test_accuracy = clf_best.score(x_test, y_test)
print("Accuracy on test: ", test_accuracy)

# Create and train the classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train, y_train)

# Save the model
joblib.dump(clf_best, "Example.joblib")

# Load the model
model = joblib.load("Example.joblib")

# Use the model for prediction
y_pred = model.predict(x_test)
