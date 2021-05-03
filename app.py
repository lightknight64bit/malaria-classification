from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import streamlit as st


def teachable_machine_classification(img, weights_file):
    
    model = keras.models.load_model(weights_file)

  
    data = np.ndarray(shape=(1, 112, 112, 3), dtype=np.float32)
    image = img
    
    size = (112, 112)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)
    
    normalized_image_array = (image_array.astype(np.float32) / 255.0) 

    data[0] = normalized_image_array

    
    prediction = model.predict(data)
    return np.argmax(prediction) 

uploaded_file = st.file_uploader("Upload cell image", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded cell image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'model/final_model.h5')
    if label == 0:
        st.write("This is a malaria infected cell")
    else:
        st.write("This is a non malaria infected cell")