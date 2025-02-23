import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import cv2

# Step 1: Load the dataset (EMNIST or MNIST)
# Here we use MNIST for simplicity, as it contains hand-written digits (0-9).
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Step 2: Preprocessing (normalization and reshaping)
# Reshape data to match the input dimensions for CNN (28x28x1)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Step 3: Build the Convolutional Neural Network (CNN) Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 for digits (0-9)
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Step 6: Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Step 7: Save the model
model.save("handwritten_character_model.h5")

# Step 8: Use the model for predictions
def predict_character(image):
    # Preprocess the image to match model input
    img_resized = cv2.resize(image, (28, 28))  # Resize to 28x28
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_normalized = img_gray.reshape(1, 28, 28, 1).astype('float32') / 255  # Normalize

    # Predict the character
    prediction = model.predict(img_normalized)
    return np.argmax(prediction)

# Example usage with a new image
# Load an image from file (use OpenCV or PIL to load a custom image of a character)
# image = cv2.imread("path_to_handwritten_image.png")
# predicted_label = predict_character(image)
# print(f"Predicted Label: {predicted_label}")
