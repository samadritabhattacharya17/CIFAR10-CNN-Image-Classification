# ==========================================================
# CIFAR-10 Image Classification using CNN with Data Augmentation
# ==========================================================
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Load Dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Training Images :", x_train.shape)
print("Testing Images  :", x_test.shape)

# -----------------------------
# Normalize Images
# -----------------------------
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

# -----------------------------
# Display Sample Images
# -----------------------------
plt.figure(figsize=(10,5))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# Data Augmentation
# -----------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen.fit(x_train)
# -----------------------------
# Build CNN Model
# -----------------------------
model = Sequential()

# First Convolution Block
model.add(Conv2D(32, (3,3), activation='relu', padding='same',
                 input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))

# Second Convolution Block
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))

# Third Convolution Block
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))

# Fully Connected Layers
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

# -----------------------------
# Model Summary
# -----------------------------
model.summary()

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# -----------------------------
# Train the Model
# -----------------------------
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=20,
    validation_data=(x_test, y_test)
)
# -----------------------------
# Save Model
# -----------------------------
model.save("model.keras")

print("\nModel saved successfully as model.keras")
# -----------------------------
# Evaluate Model
# -----------------------------
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Loss :", test_loss)
print("Test Accuracy :", test_accuracy)
# -----------------------------
# Plot Accuracy
# -----------------------------
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("plots/accuracy.png")
plt.show()
# -----------------------------
# Plot Loss
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("plots/loss.png")
plt.show()
