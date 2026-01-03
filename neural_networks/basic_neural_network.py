# Deprecated

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x_train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_train_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Define number of neurons in each layer
n_input = 2  # number of features
n_hidden = 4  # number of hidden units
n_output = 1  # number of output units

# Define placeholders for input and output
x = tf.placeholder(tf.float32, shape=[None, n_input])
y = tf.placeholder(tf.float32, shape=[None, n_output])

# Define weights and biases for each layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden]))  # weights from input to hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden]))  # biases for hidden layer
w2 = tf.Variable(tf.random_normal([n_hidden, n_output]))  # weights from hidden to output layer
b2 = tf.Variable(tf.random_normal([n_output]))  # biases for output layer

# Define activation functions for each layer
z1 = tf.add(tf.matmul(x, w1), b1)  # linear combination for hidden layer
a1 = tf.nn.sigmoid(z1)  # sigmoid activation for hidden layer
z2 = tf.add(tf.matmul(a1, w2), b2)  # linear combination for output layer
a2 = tf.nn.sigmoid(z2)  # sigmoid activation for output layer# define cost function and optimizer
cost = tf.reduce_mean(tf.square(y - a2))  # mean squared error
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)  # gradient descent optimizer

# Create a session object
sess = tf.Session()

# Initialize all variables
sess.run(tf.global_variables_initializer())

# Train for 1000 epochs
for epoch in range(1000):
    # Run optimization and calculate cost
    _, c = sess.run([optimizer, cost], feed_dict={x: x_train_data, y: y_train_data})
    # Print cost every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Cost: {c}")

# Our model on unseen data
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)  # test input
y_test = np.array([[0], [1], [1], [0]], dtype=np.float32)  # Test output

# Predict using our trained model
y_pred = sess.run(a2, feed_dict={x: x_test})

# Print predictions and actual values
print(f"Predictions: {y_pred}")
print(f"Actual: {y_test}")

# Plot predictions and actual values
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred[:, 0], cmap='RdBu')
plt.title("Predictions")
plt.show()

plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test[:, 0], cmap='RdBu')
plt.title("Actual")
plt.show()

# Close session
sess.close()