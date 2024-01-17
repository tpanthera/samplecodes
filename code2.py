import cv2
import numpy as np
from sklearn.externals import joblib  # For loading the logistic regression model

# Load the logistic regression model (replace 'your_model.pkl' with the actual filename)
logistic_regression_model = joblib.load('your_model.pkl')

def preprocess_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image (adjust dimensions based on your model's input size)
    img_resized = cv2.resize(img, (28, 28))  # Assuming a 28x28 pixel image for simplicity

    # Flatten the image to a 1D array
    img_flattened = img_resized.flatten()

    # Normalize pixel values (if your model was trained on normalized data)
    img_normalized = img_flattened / 255.0  # Assuming original pixel values range from 0 to 255

    return img_normalized

def make_prediction(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Reshape the processed image to match the expected input shape of the model
    processed_image_reshaped = processed_image.reshape(1, -1)

    # Make a prediction using the logistic regression model
    prediction = logistic_regression_model.predict(processed_image_reshaped)

    return prediction

# Replace 'your_image.jpg' with the actual image file path
image_path = 'your_image.jpg'

# Make a prediction
result = make_prediction(image_path)

# Display the result
print("Prediction:", result)
