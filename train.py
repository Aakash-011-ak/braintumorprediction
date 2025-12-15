import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

st.title("🧠 Brain Tumor Detection using CNN (Google Drive Model)")

# ------------------------------------------------
# 1. Google Drive Model Download
# ------------------------------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=139lIdp8m1L0I5Yl8IKDkIRNAtU8lPRbm"
MODEL_PATH = "model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model... Please wait ⏳")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")

download_model()

# ------------------------------------------------
# 2. Load the Model
# ------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

st.success("Model Loaded Successfully ✔️")

# ------------------------------------------------
# 3. Upload Image
# ------------------------------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI Image", width=300)

    # Preprocessing
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # ------------------------------------------------
    # 4. Prediction
    # ------------------------------------------------
    if st.button("Predict"):
        prediction = model.predict(img_array)[0][0]

        st.write("### Prediction Score:", float(prediction))

        if prediction >= 0.5:
            st.error("🧠 **Tumor Detected**")
        else:
            st.success("✔️ No Tumor Detected")
