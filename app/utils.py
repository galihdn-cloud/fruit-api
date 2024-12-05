import tensorflow as tf
import numpy as np
from PIL import Image

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3

# Load model function
def load_model():
    model = tf.keras.models.load_model("models/fruit_classification_model_5buah.h5")
    return model

# Preprocess the image before feeding it into the model
def preprocess_image(img: Image.Image):
    # Convert image to RGB in case it's in other mode like RGBA or grayscale
    img = img.convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array
