import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# --------------------------
# Model Download from Google Drive
# --------------------------
MODEL_PATH = "plant_disease_model.h5"
FILE_ID = "1kHDAJuDNkk2ctcSn4GZHy0levdMoqpR6"  # 🔁 Replace with your real Google Drive File ID
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.text("Downloading model from Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)

# --------------------------
# Load Model
# --------------------------
model = tf.keras.models.load_model(MODEL_PATH)
st.success("Model loaded successfully!")

# --------------------------
# Class Names
# --------------------------
class_names = ['Eggplant', 'Pumpkin', 'Rose', 'Mango']

# --------------------------
# Streamlit App UI
# --------------------------
st.title("🌿 Plant Disease Detector")
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize and preprocess
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("🔍 Result")
    st.write(f"**Class:** {class_names[class_index]}")
    st.write(f"**Confidence:** {confidence:.2f}")
