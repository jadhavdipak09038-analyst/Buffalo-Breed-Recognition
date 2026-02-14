# app.py - Improved Buffalo Breed Recognition (Streamlit)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Buffalo Breed Recognition", layout="centered")
st.title("🐃 Buffalo Breed Recognition System")

# ---------------- MODEL PATH ----------------
MODEL_PATH = r"C:\Users\HP\Downloads\Buffalo_Breed_Recognition\buffalo_breed_model.h5"

# ---------------- CLASS NAMES ----------------
class_names = [
    'Alambadi','Amritmahal','Ayrshire','Banni','Bargur','Bhadawari','Brown_Swiss','Dangi','Deoni','Gir',
    'Guernsey','Hallikar','Hariana','Holstein_Friesian','Jaffrabadi','Jersey','Kangayam','Kankrej','Kasargod',
    'Kenkatha','Kherigarh','Khillari','Krishna_Valley','Malnad_gidda','Mehsana','Murrah','Nagori','Nagpuri',
    'Nili_Ravi','Nimari','Ongole','Pulikulam','Rathi','Red_Dane','Red_Sindhi','Sahiwal','Surti','Tharparkar',
    'Toda','Umblachery','Vechur'
]

IMG_SIZE = (224, 224)
TOP_K = 3

# ---------------- LOAD MODEL ----------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    model = tf.keras.models.load_model(path, compile=False)
    return model

try:
    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Could not load model.\n\n{e}")
    st.stop()

st.write("Upload a buffalo/cattle image (JPG, JPEG, PNG).")

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(pil_img):
    img = pil_img.convert("RGB")
    img = ImageOps.fit(img, IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img

# ---------------- FILE UPLOAD ----------------
uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded is not None:

    pil = Image.open(uploaded)
    arr, shown = preprocess_image(pil)

    st.image(shown, caption="Uploaded Image", use_column_width=True)

    # ---------------- PREDICTION ----------------
    preds = model.predict(arr)[0]

    best_idx = int(np.argmax(preds))
    best_confidence = float(preds[best_idx])
    best_name = class_names[best_idx]

    st.markdown("## 🏆 Prediction Result")
    st.success(f"Predicted Breed: **{best_name}**")
    st.write(f"Confidence: {best_confidence*100:.2f}%")

    # Low confidence warning (no rejection)
    if best_confidence < 0.30:
        st.warning("⚠ Model confidence is low. Prediction may not be accurate.")

    # ---------------- TOP K ----------------
    st.markdown("### 🔝 Top 3 Predictions")
    top_idx = np.argsort(preds)[-TOP_K:][::-1]

    for rank, idx in enumerate(top_idx, start=1):
        st.write(f"{rank}. **{class_names[idx]}** — {preds[idx]*100:.2f}%")

    # ---------------- BAR CHART ----------------
    st.markdown("### 📊 Confidence Chart (Top 3)")

