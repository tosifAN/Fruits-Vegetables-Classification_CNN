import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image  # You may need this for better image handling

# Streamlit header
st.header('Image Classification Model')

# Load model - make sure the path uses raw string or double backslashes
model = load_model('Image_classify.keras')

# List of categories
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Image settings
img_height = 180
img_width = 180

# Image file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess the image
        image_load = Image.open(uploaded_file)
        image_resized = image_load.resize((img_height, img_width))  # Resize the image
        img_arr = tf.keras.utils.img_to_array(image_resized)
        img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension

        # Predict
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict[0])

        # Display the image
        st.image(image_load, width=200)

        # Display the prediction and accuracy
        st.write('Veg/Fruit in image is: ' + data_cat[np.argmax(score)])
        st.write('With accuracy of: {:.2f}%'.format(np.max(score) * 100))

    except Exception as e:
        st.write(f"Error: {str(e)}")
