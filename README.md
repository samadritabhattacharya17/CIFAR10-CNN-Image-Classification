# CIFAR10-CNN-Image-Classification
# CIFAR-10 Image Classification using CNN with Data Augmentation

## Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the CIFAR-10 dataset into ten different object categories. Data augmentation techniques are applied to improve the model's generalization and reduce overfitting.

## Features

* Image classification using CNN
* Data preprocessing and normalization
* Data augmentation
* Model training and evaluation
* Accuracy and loss visualization
* Prediction on unseen images
* Model saved using Keras

## Dataset

* CIFAR-10
* 60,000 RGB images
* 10 object classes
* Image size: 32 × 32 pixels

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn

## Results

* Test Accuracy: **76.16%**
* Training Epochs: **20**
* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy

## Project Structure

```text
train.py
predict.py
model.keras
requirements.txt
plots/
predictions/
```

## Future Improvements

* Transfer Learning using ResNet50
* Vision Transformer (ViT)
* Model Quantization
* Real-time webcam prediction
* Deployment using Streamlit

## Author

Samadrita Bhattacharjee
