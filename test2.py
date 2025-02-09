import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the pre-trained model
MODEL_PATH = 'sign_language_model.h5'
model = load_model(MODEL_PATH)

# Image parameters
img_height, img_width = 128, 128

# Function to make predictions
def predict_class(img_file):
    img = image.load_img(img_file, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    class_label = class_names[class_idx]  # Get class label based on index
    return class_label

# Extract class names from the model training
class_names = ['Hello', 'I Love You', 'No', 'Stop', 'Thank You', 'Thumbs Down', 'Thumbs Up', 'Victory', 'Yes']

# Streamlit UI
st.title("Sign Language Classification")
st.write("Upload an image or take a photo to predict the corresponding sign language class.")

# Option Selection
option = st.radio("Select Input Method:", ("Upload Image", "Take Photo"))

if option == "Upload Image":
    # File uploader
    uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_container_width=True)
        st.write("Processing...")
        
        # Save uploaded image temporarily
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_image.read())
            
        # Perform prediction
        prediction = predict_class("temp_image.png")
        st.write(f"Predicted Class: **{prediction}**")

elif option == "Take Photo":
    # Camera input
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        st.image(camera_image, caption='Captured Image', use_container_width=True)
        st.write("Processing...")
        
        # Save the captured image temporarily
        with open("temp_shot.png", "wb") as f:
            f.write(camera_image.read())
            
        # Perform prediction
        prediction = predict_class("temp_shot.png")
        st.write(f"Predicted Class: **{prediction}**")
