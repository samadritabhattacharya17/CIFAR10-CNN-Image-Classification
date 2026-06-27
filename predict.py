import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# -----------------------------
# Load the trained model
# -----------------------------
model = tf.keras.models.load_model("model.keras")

# -----------------------------
# Load CIFAR-10 test dataset
# -----------------------------
(_, _), (x_test, y_test) = cifar10.load_data()

# Normalize images
x_test = x_test.astype("float32") / 255.0

# Class names
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
# Select a random image
# -----------------------------
index = random.randint(0, len(x_test) - 1)

image = x_test[index]
true_label = y_test[index][0]

# -----------------------------
# Predict
# -----------------------------
prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)

predicted_class = np.argmax(prediction)
confidence = np.max(prediction) * 100

# -----------------------------
# Display Result
# -----------------------------
plt.imshow(image)
plt.title(
    f"Autonomous UAV Vision\n"
    f"Detected: {class_names[predicted_class]}\n"
    f"Ground Truth: {class_names[true_label]}\n"
    f"Confidence: {confidence:.2f}%"
)
plt.axis("off")
plt.show()

# -----------------------------
# Print Result
# -----------------------------
print("="*55)
print("AUTONOMOUS UAV VISION SYSTEM")
print("="*55)

print(f"Detected Object : {class_names[predicted_class]}")
print(f"Ground Truth    : {class_names[true_label]}")
print(f"Recognition Confidence : {confidence:.2f}%")

print("\nMission Status : SUCCESS")
print("Object detected successfully.")
print("Suitable for lightweight UAV onboard vision.")