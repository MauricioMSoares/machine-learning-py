import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Load the MNIST Datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the dataset
x_train = x_train.reshape((-1, 28, 28, 1)) / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(10, activation="sotfmax")
])

# Compile and train the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Select random images from the test
random_index = np.random.choice(len(x_test), size=5, replace=False)
images = x_test[random_index]
true_labels = np.argmax(y_test[random_index], axis=1)

# Make predictions using the trained model
predictions = np.argmax(model.predits(images), axis=1)

fig, axes = plt.subplots(1, 5, figsize=(12, 4))
for i, (image, true_label, predicted_label) in enumerate(zip(images, true_labels, predictions)):
    axes[i].imshow(image.squeeze(), cmap="gray")
    axes[i].set_title(f"True: {true_label}, Predicted: {predicted_label}")
    axes[i].axis("off")
plt.tight_layout()
