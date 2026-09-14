import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import zipfile
import tempfile
import h5py
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
    num_classes = len(idx_to_class)

    # Download model from Hugging Face
    model_path = hf_hub_download(
        repo_id="RATHANSUMBET14/weather-vision-app",
        filename="best_weather_model.keras",
        repo_type="space"
    )

    # Extract contents of the .keras zip archive
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(model_path, "r") as archive:
        archive.extractall(tmp_dir)

    config_path = os.path.join(tmp_dir, "config.json")
    weights_path = os.path.join(tmp_dir, "model.weights.h5")

    # Read config to find top layer structure
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Build standard DenseNet121 transfer model
    base = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Check layer names from config if sequential or functional
    layers_config = cfg.get("config", {}).get("layers", [])
    dense_units = []
    for l in layers_config:
        if l.get("class_name") == "Dense":
            dense_units.append(l.get("config", {}).get("units"))

    # Reconstruct top
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    if len(dense_units) > 1:
        for units in dense_units[:-1]:
            x = tf.keras.layers.Dense(units, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=base.input, outputs=outputs)

    # Load weights with by_name=True to safely align with DenseNet layers
    try:
        model.load_weights(weights_path, by_name=True)
    except Exception:
        # If strict by_name fails, load with skip_mismatch
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model, idx_to_class

model, idx_to_class = load_all()

uploaded_file = st.file_uploader("Upload a weather image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_raw = Image.open(uploaded_file).convert("RGB")
    st.image(image_raw, caption="Uploaded Image", use_container_width=True)

    # DenseNet requires ImageNet preprocessing:
    resized = image_raw.resize((224, 224))
    img_array = np.array(resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Inference
    preds = model(img_array, training=False).numpy()[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])

    st.subheader(f"Prediction: **{idx_to_class[top_idx]}** ({confidence * 100:.1f}%)")

    # Grad-CAM
    try:
        target_layer = None
        for layer in reversed(model.layers):
            if "relu" in layer.name or "conv" in layer.name:
                target_layer = layer
                break

        if target_layer:
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[target_layer.output, model.output]
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
        st.caption(f"Grad-CAM visualizer unavailable for this sample: {e}")
