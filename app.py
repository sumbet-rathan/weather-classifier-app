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
    with open("weather_classes.json", "r") as f:
        labels = json.load(f)
    idx_to_class = {int(k): v for k, v in labels.items()}

    zip_path = hf_hub_download(
        repo_id="RATHANSUMBET14/weather-vision-app",
        filename="weather_saved_model.zip",
        repo_type="space"
    )

    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(tmp_dir)

    loaded = tf.saved_model.load(tmp_dir)
    return loaded, idx_to_class

model_obj, idx_to_class = load_all()
infer_fn = model_obj.signatures["serving_default"]

uploaded_file = st.file_uploader("Upload a weather image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_raw = Image.open(uploaded_file).convert("RGB")
    st.image(image_raw, caption="Uploaded Image", use_container_width=True)

    resized = image_raw.resize((224, 224))
    img_array = np.array(resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    input_tensor = tf.convert_to_tensor(img_array)
    outputs = infer_fn(input_tensor)
    output_key = list(outputs.keys())[0]
    preds = outputs[output_key].numpy()[0]

    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])

    st.subheader(f"Prediction: **{idx_to_class[top_idx]}** ({confidence * 100:.1f}%)")

    # Grad-CAM Visualization
    try:
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            out = infer_fn(input_tensor)[output_key]
            target_score = out[:, top_idx]

        input_grads = tape.gradient(target_score, input_tensor)
        heatmap = tf.reduce_mean(tf.abs(input_grads), axis=-1)[0]
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

        heatmap = np.uint8(255 * heatmap.numpy())
        jet = plt.get_cmap("jet")(np.arange(256))[:, :3]
        jet_heatmap = Image.fromarray(np.uint8(jet[heatmap] * 255)).resize(image_raw.size)
        overlay = Image.blend(image_raw, jet_heatmap, alpha=0.45)

        st.image(overlay, caption="Grad-CAM Attention Map", use_container_width=True)
    except Exception as e:
        st.caption(f"Grad-CAM visualizer unavailable for this sample: {e}")
