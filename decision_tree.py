from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
import os


# Load the iris dataset
iris = load_iris()
x = iris.data
y = iris.target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Create a decision tree classifier with maximum depth of 3
clf = DecisionTreeClassifier(max_depth=3)

# Train the classifier on the training data
clf.fit(x_train, y_train)

# Make predictions on the testing data
y_prediction = clf.predict(x_test)

# Calculate the accuracy score of the classifier
accuracy = accuracy_score(y_test, y_prediction)
print("Accuracy score: ", accuracy)

# Visualize the decision tree using graphviz
# Currently generates the iris_tree file, but is unable to open the visualization due to not having Graphviz installed
os.environ["PATH"] += os.pathsep + os.path.abspath("C:\Program Files (x86)\Graphviz\bin")
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph.render("iris_tree", view=True)
