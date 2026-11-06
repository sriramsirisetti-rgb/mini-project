import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Plant Leaf Disease Detection", page_icon="🌿", layout="centered")
MODEL_PATH = Path(__file__).parent / "plant_leaf_disease_cnn.keras"
CLASS_PATH = Path(__file__).parent / "class_names.json"
IMG_SIZE = (128, 128)

st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload a clear image of a plant leaf. The trained CNN model will predict the most likely disease class.")

@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)

if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.warning("The trained model files are not present yet.")
    st.markdown(
        "To enable predictions, run **train_model.py** to create `plant_leaf_disease_cnn.keras` "
        "and `class_names.json`, then upload both files to the same GitHub repository as `app.py`."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    class_names = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
except Exception as exc:
    st.error(f"Could not load the model or class labels: {exc}")
    st.stop()

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded leaf image", use_container_width=True)

        resized = image.resize(IMG_SIZE)
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)

        if st.button("Predict disease", type="primary", use_container_width=True):
            with st.spinner("Analyzing leaf image..."):
                probabilities = model.predict(image_batch, verbose=0)[0]
                top_indices = np.argsort(probabilities)[::-1][:3]

            best_index = int(top_indices[0])
            st.subheader("Prediction")
            st.success(f"{class_names[best_index].replace('___', ' — ').replace('_', ' ')}")
            st.metric("Confidence", f"{probabilities[best_index] * 100:.2f}%")

            st.subheader("Top 3 predictions")
            for index in top_indices:
                label = class_names[int(index)].replace("___", " — ").replace("_", " ")
                confidence = float(probabilities[int(index)])
                st.write(f"**{label}** — {confidence * 100:.2f}%")
                st.progress(min(max(confidence, 0.0), 1.0))
            st.caption("This is an AI model prediction, not a confirmed agricultural diagnosis.")
    except Exception as exc:
        st.error(f"Unable to process this image: {exc}")
