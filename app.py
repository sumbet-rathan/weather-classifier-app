import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import zipfile
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Weather Classifier", layout="centered")
st.title("🌦️ Weather Classifier (DenseNet121 + Grad-CAM)")

@st.cache_resource
def load_all():
    # 1. Load class mapping
    with open("weather_classes.json", "r") as f:
        labels = json.load(f)
    idx_to_class = {int(k): v for k, v in labels.items()}

    # 2. Download and unpack SavedModel archive
    zip_path = hf_hub_download(
        repo_id="RATHANSUMBET14/weather-vision-app",
        filename="weather_saved_model.zip",
        repo_type="space"
    )

    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(tmp_dir)

    # 3. Load SavedModel computation endpoint
    loaded = tf.saved_model.load(tmp_dir)
    infer_fn = loaded.signatures["serving_default"]

    return infer_fn, idx_to_class

infer_fn, idx_to_class = load_all()

uploaded_file = st.file_uploader("Upload a weather image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_raw = Image.open(uploaded_file).convert("RGB")
    st.image(image_raw, caption="Uploaded Image", use_container_width=True)

    # ImageNet preprocessing
    resized = image_raw.resize((224, 224))
    img_array = np.array(resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Run inference through the exported graph
    input_tensor = tf.convert_to_tensor(img_array)
    outputs = infer_fn(input_tensor)
    
    # Extract prediction tensor
    output_key = list(outputs.keys())[0]
    preds = outputs[output_key].numpy()[0]

    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])

    st.subheader(f"Prediction: **{idx_to_class[top_idx]}** ({confidence * 100:.1f}%)")
