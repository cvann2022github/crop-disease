# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Loading the Model
model = load_model('plant_disease_model.h5')

# Name of Classes (updated to include CoSev classes)
CLASS_NAMES = (
    'Tomato-Bacterial_spot', 'Potato-Early_blight', 'Corn-Common_rust',
    'Cotton-Curl_Stage1', 'Cotton-Curl_Stage1_Stage2_Sooty', 
    'Cotton-Curl_Stage1_Sooty', 'Cotton-Curl_Stage2', 
    'Cotton-Curl_Stage2_Sooty', 'Cotton-Healthy', 'Cotton-Leaf_Enation'
)

# Setting Title of App
st.title("AI Crop Disease Prevention and Management System")
st.markdown("Upload an image of the plant leaf")

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an OpenCV image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(f"Image Shape: {opencv_image.shape}")
        
        # Resizing the image to match the input shape of the model
        opencv_image = cv2.resize(opencv_image, (256, 256))
        
        # Convert image to 4 dimensions (1, 256, 256, 3) as required by the model
        opencv_image = np.expand_dims(opencv_image, axis=0)
        
        # Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        
        # Display the result
        st.title(f"This is a {result.split('-')[0]} leaf with {result.split('-')[1]}")
