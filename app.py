import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Weather Classifier", layout="centered")
st.title("🌦️ Weather Classifier (DenseNet121 + Grad-CAM)")

@st.cache_resource
def load_all():
    model_path = hf_hub_download(
        repo_id="RATHANSUMBET14/weather-vision-app",
        filename="best_weather_model.keras",
        repo_type="space"
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    
    with open("weather_classes.json", "r") as f:
        labels = json.load(f)
    return model, {int(k): v for k, v in labels.items()}

model, idx_to_class = load_all()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_raw = Image.open(uploaded_file).convert("RGB")
    st.image(image_raw, caption="Uploaded Image", use_container_width=True)

    resized = image_raw.resize((224, 224))
    img_array = np.expand_dims(np.array(resized) / 255.0, axis=0)

    preds = model(img_array, training=False).numpy()[0]
    top_idx = np.argmax(preds)

    st.subheader(f"Prediction: **{idx_to_class[top_idx]}** ({preds[top_idx]*100:.1f}%)")

    try:
        backbone = model.layers[0]
        grad_model = tf.keras.models.Model(
            inputs=backbone.input,
            outputs=[backbone.get_layer("relu").output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, p = grad_model(img_array)
            loss = p[:, top_idx]
        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

        heatmap = np.uint8(255 * heatmap.numpy())
        jet = plt.get_cmap("jet")(np.arange(256))[:, :3]
        jet_heatmap = Image.fromarray(np.uint8(jet[heatmap] * 255)).resize(image_raw.size)
        overlay = Image.blend(image_raw, jet_heatmap, alpha=0.4)

        st.image(overlay, caption="Grad-CAM Attention Map", use_container_width=True)
    except Exception as e:
        st.warning(f"Grad-CAM could not be computed: {e}")
