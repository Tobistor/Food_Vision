#importing the libraries
import streamlit as st
import joblib
from PIL import Image
from skimage.transform import resize
import numpy as np
import h5py
import time
from keras.models import load_model

# An array of all the class names
class_names = ['apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare','beet_salad','beignets',
               'bibimbap','bread_pudding','breakfast_burrito','bruschetta','caesar_salad','cannoli','caprese_salad',
               'carrot_cake','ceviche','cheese_plate','cheesecake','chicken_curry','chicken_quesadilla','chicken_wings',
               'chocolate_cake','chocolate_mousse','churros','clam_chowder','club_sandwich','crab_cakes','creme_brulee',
               'croque_madame','cup_cakes','deviled_eggs','donuts','dumplings','edamame','eggs_benedict','escargots',
               'falafel','filet_mignon','fish_and_chips','foie_gras','french_fries','french_onion_soup','french_toast',
               'fried_calamari','fried_rice','frozen_yogurt','garlic_bread','gnocchi','greek_salad','grilled_cheese_sandwich',
               'grilled_salmon','guacamole','gyoza','hamburger','hot_and_sour_soup','hot_dog','huevos_rancheros','hummus',
               'ice_cream','lasagna','lobster_bisque','lobster_roll_sandwich','macaroni_and_cheese','macarons','miso_soup',
               'mussels','nachos','omelette','onion_rings','oysters','pad_thai','paella','pancakes','panna_cotta','peking_duck',
               'pho','pizza','pork_chop','poutine','prime_rib','pulled_pork_sandwich','ramen','ravioli','red_velvet_cake','risotto',
               'samosa','sashimi','scallops','seaweed_salad','shrimp_and_grits','spaghetti_bolognese','spaghetti_carbonara','spring_rolls',
               'steak','strawberry_shortcake','sushi','tacos','takoyaki','tiramisu','tuna_tartare','waffles']

# Load the model from the .h5 file
#model = h5py.File('food_vision.h5', 'r')
model = load_model('food_vision.h5')

# Create a Streamlit user interface
st.title('Food Vision')

# Get the input image from the user
image_file = st.file_uploader('Select an image:')

# Load and preprocess the input image
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
    st.write(f'Predicted class: {class_names[predicted_class]}')
