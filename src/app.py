# app.py
# A simple web app to showcase the crop stress detection model.

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

# --- CONFIGURATION (adjust paths if necessary) ---
# Assuming this app.py is in the 'src' folder
MODEL_PATH = os.path.join("..", "models", "crop_stress_model.h5")
CLASS_INDICES_PATH = os.path.join("..", "models", "class_indices.json")
IMAGE_SIZE = (224, 224)

# --- MODEL AND CLASS LOADING ---
# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model_and_classes():
    """Loads the trained model and class names."""
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    # Invert the dictionary to map index to class name
    class_names = {v: k for k, v in class_indices.items()}
    return model, class_names

model, class_names = load_model_and_classes()

# --- HELPER FUNCTION ---
def preprocess_image(image):
    """Preprocesses the uploaded image."""
    img = image.resize(IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# --- STREAMLIT INTERFACE ---
st.title("🌱 Crop Stress Detection")
st.write(
    "Upload an image of a plant leaf, and the model will predict its condition."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and make a prediction
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions)

    st.success(f"**Prediction:** {predicted_class.replace('_', ' ').title()}")
    st.info(f"**Confidence:** {confidence:.2%}")