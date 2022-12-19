#importing the libraries
import streamlit as st
import joblib
from PIL import Image
from skimage.transform import resize
import numpy as np
import h5py
import time
from keras.models import load_model

# Load the model from the .h5 file
#model = h5py.File('food_vision.h5', 'r')
model = load_model('food_vision.h5')

# Create a Streamlit user interface
st.title('Food Vision')

# Get the input image from the user
image_file = st.file_uploader('Select an image:')

# Load and preprocess the input image
if image_file is not None:
    image = Image.open(image_file)
    image = image.resize((224, 224))  # resize the image to the model's input size
    image_array = np.array(image)  # convert the image to a numpy array
    image_array = image_array / 255.0  # normalize the pixel values
    image_array = np.expand_dims(image_array, axis=0)  # add a batch dimension

    # Use the model to make a prediction
    prediction = model.predict(image_array)

    # Display the prediction in the Streamlit app
    predicted_class = np.argmax(prediction)
    st.write(f'Predicted class: {predicted_class}')
