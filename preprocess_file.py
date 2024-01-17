from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import numpy as np
from joblib import load

app = FastAPI()

# Load the logistic regression model
model = load("logistic_regression_model.joblib")

# FastAPI route to handle file upload
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
  """
  - BytesIO(contents): Creates a BytesIO object from the binary content (contents). BytesIO is an in-memory stream for binary data.
  - Image.open(BytesIO(contents)): Opens an image using the Image.open method. The image data is read from the BytesIO object.
  - .convert("L"): Converts the image to grayscale mode. In PIL, "L" mode represents a single-channel (luminance) image, 
  where pixel values are represented by intensity (brightness) ranging from 0 to 255. 
  Grayscale images have only one channel compared to RGB images, which have three channels (red, green, blue).
  
  """
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image).flatten() / 255.0

    # Make prediction
    prediction = model.predict([image_array])[0]

    return JSONResponse(content={"prediction": int(prediction)})
