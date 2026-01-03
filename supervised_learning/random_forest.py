import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load the dataset
data = load_breast_cancer()
x = data.data
y = data.target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Create and train a random classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train, y_train)

# Make predictions on the test set
predictions = rf_classifier.predict(x_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Get feature importance
importances = rf_classifier.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances[::-1])

# Rearrange feature names to match the sorted feature importances
feature_names = data.feature_names[indices]

# Create bar plot of feature importances
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(x.shape[1]), importances[indices])
plt.xticks(range(x.shape[1]), feature_names, rotation=90)
plt.show()
