import numpy as np
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from PIL import Image

# Load the individual models
model_fruit = load_model("models/fruit_classification_model_5buah.h5")
model_apple = load_model("models/Apple_ripeness_model.h5")
model_durian = load_model("models/Durian_ripeness_model.h5")
model_grape = load_model("models/Grape_ripeness_model.h5")
model_strawberry = load_model("models/Strawberry_ripeness_model.h5")
model_dragonfruit = load_model("models/DragonFruit_ripeness_model.h5")

# Initialize FastAPI app
app = FastAPI()

# Fruit class mapping
fruit_classes = ["apple", "durian", "grape", "strawberry", "dragonfruit"]

# Function to load and preprocess image
def load_and_preprocess_image(image: BytesIO, target_size=(128, 128)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image to [0, 1]
    return img_array

# Function to predict ripeness for specific fruits
def predict_ripeness(fruit_label, input_image):
    if fruit_label == "apple":
        model = model_apple
    elif fruit_label == "durian":
        model = model_durian
    elif fruit_label == "grape":
        model = model_grape
    elif fruit_label == "strawberry":
        model = model_strawberry
    elif fruit_label == "dragonfruit":
        model = model_dragonfruit
    else:
        return "Unsupported fruit"

    prediction = model.predict(input_image)
    predicted_label = (prediction > 0.5).astype(int)
    return "Ripe" if predicted_label == 1 else "Unripe"

# API endpoint to upload and predict
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image = await file.read()
    image = BytesIO(image)

    # Load and preprocess the image
    input_image = load_and_preprocess_image(image)

    # Fruit classification model prediction
    predictions = model_fruit.predict(input_image)
    predicted_labels = np.argmax(predictions, axis=1)
    fruit_label = fruit_classes[predicted_labels[0]]

    # Predict ripeness based on the fruit type
    ripeness = predict_ripeness(fruit_label, input_image)

    # Return results
    return {
        "predicted_fruit": fruit_label,
        "ripeness": ripeness
    }
