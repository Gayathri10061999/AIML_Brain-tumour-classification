import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from config import IMG_SIZE, MODEL_PATH

model = load_model(MODEL_PATH)

class_names = ['glioma', 'meningioma', 'pituitary', 'no tumor']

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

st.title("Brain Tumor Classification")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess_image(image)
    prediction = model.predict(img)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"### Prediction: {pred_class}")
    st.write(f"Confidence: {confidence:.2f}")
