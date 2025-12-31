# Face Age Prediction using CNN (Regression)

This project predicts a person’s age from a given face image using a Convolutional Neural Network (CNN).

## Project Overview
The model takes a single face image as input and predicts the approximate age of the person present in the image.  
It is built using Deep Learning (CNN) and trained as a **regression problem**.

## How it Works
1. Input face image is loaded.
2. Image is resized to 224 × 224.
3. Image is preprocessed using the same method used during training.
4. The trained CNN model predicts the age.
5. The predicted age is displayed as output.

## Input
- A single face image of a person (JPEG / PNG).

## Output
- Predicted age of the person in years.

## Model Details
- Algorithm: Convolutional Neural Network (CNN)
- Problem Type: Regression
- Input Shape: (224, 224, 3)
- Output: Continuous value (Age)

## Inference
The trained model is saved and loaded using Keras.
You can give any face image as input and get the predicted age.

## Use Case
- Face-based age estimation
- Computer Vision learning project
- Deep Learning regression example

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy

## Note
The predicted age is an approximate value and depends on the quality of the input image and training data.
