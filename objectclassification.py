# Save this code in a file, for example, "cat_dog_detection_app.py"

"""
uvicorn object_classification:app --reload --port 8004
streamlit run object_classification.py

"""
import streamlit as st
import requests
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import torch
from torchvision import models, transforms
import torch.nn.functional as F

# Streamlit UI
st.title("Object Detection App")

# User uploads an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Convert the image to bytes
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    # Make a request to FastAPI for cat and dog detection
    response = requests.post("http://127.0.0.1:8004/predict", files={"file": ("image.jpg", image_bytes)})

    # Display the prediction result
    if response.status_code == 200:
        prediction_result = response.json()["prediction"]
        st.success(f"Prediction: {prediction_result}")
    else:
        st.error("Failed to get prediction from the model")


# FastAPI Server

app = FastAPI()

# Load the pre-trained ResNet-18 model for cat and dog detection
model = models.resnet18(pretrained=True)
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dummy function for cat and dog detection
def detect_cat_dog(image_bytes):
    # Convert image bytes to PIL Image
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Apply the transformation
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    # Make the prediction
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)
    print(f"predicted_idx is {predicted_idx.item()}")
    predicted_class = predicted_idx.item()

    # Map class index to label
    # You can customize this part based on your requirements
    class_mapping = {
        0: "Class 0",
        1: "Class 1",
        985: "daisy",
        248: "Dog",
        284:"Cat",
        361:"Cat",
        281:"Cat",
        716:"dandelion",
        903:"Rose",
        991: "flower"


        # Add more class mappings as needed
    }
    
    predicted_class_label = class_mapping.get(predicted_class, "Unknown Class")

    return predicted_class_label

@app.post("/predict")
def make_prediction(file: UploadFile = File(...)):
    # Perform cat and dog detection
    prediction = detect_cat_dog(file.file.read())

    return {"prediction": prediction}
