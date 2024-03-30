import tensorflow as tf
import streamlit as st
from PIL import Image
import os
import numpy as np

working_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(working_dir, "Bone_Break_Classification.h5")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

class_names = ['Avulsion fracture', 'Comminuted fracture', 'Fracture Dislocation', 'Greenstick fracture', 'Hairline Fracture', 'Impacted fracture', 'Longitudinal fracture', 'Oblique fracture', 'Pathological fracture', 'Spiral Fracture']

img_height = 256  
img_width = 256 

# Preprocess image
def preprocess_img(image):
    image = image.resize((img_height, img_width))
    image = np.array(image)
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
st.title('Bone Break Classification')

uploaded_image = st.file_uploader('Upload an Image', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button('Classify'):
            # Preprocess the uploaded image
            img_arr = preprocess_img(image)
            # Make a prediction using the pre-trained model
            result = model.predict(img_arr)
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]
            st.success(f'{prediction}')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
