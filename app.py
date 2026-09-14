import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
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

def transfer_k3_weights_to_k2(model, h5_path):
    """Maps Keras 3 H5 weights directly into the matching Keras 2 model layers."""
    with h5py.File(h5_path, "r") as f:
        # Check if weights are grouped under 'layers'
        root = f["layers"] if "layers" in f else f
        
        # Build a lookup table of weight arrays by their base layer name
        weight_store = {}
        for layer_key in root.keys():
            grp = root[layer_key]
            # Handle Keras 3 'vars' sub-group if present
            if "vars" in grp:
                grp = grp["vars"]
            tensors = [np.array(grp[k]) for k in sorted(grp.keys(), key=lambda x: int(x) if x.isdigit() else x)]
            weight_store[layer_key] = tensors

        # Assign weights to model layers
        for layer in model.layers:
            # If base model is nested or has matching name
            matched_weights = weight_store.get(layer.name)
            
            # Fallback search if names have suffixes like '_1' or nested prefixes
            if matched_weights is None:
                for k in weight_store:
                    if layer.name == k.split("/")[-1]:
                        matched_weights = weight_store[k]
                        break

            if matched_weights:
                model_w = layer.get_weights()
                if len(model_w) == len(matched_weights):
                    try:
                        layer.set_weights(matched_weights)
                    except Exception:
                        pass

@st.cache_resource
def load_all():
    # 1. Load class names
    with open("weather_classes.json", "r") as f:
        labels = json.load(f)
    idx_to_class = {int(k): v for k, v in labels.items()}
    num_classes = len(idx_to_class)

    # 2. Reconstruct architecture
    base = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=base.input, outputs=outputs)

    # 3. Download weights file from Hugging Face
    weights_path = hf_hub_download(
        repo_id="RATHANSUMBET14/weather-vision-app",
        filename="weather_model.weights.h5",
        repo_type="space"
    )

    # 4. Safely load weights
    try:
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    except Exception:
        transfer_k3_weights_to_k2(model, weights_path)

    return model, idx_to_class

model, idx_to_class = load_all()

uploaded_file = st.file_uploader("Upload a weather image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_raw = Image.open(uploaded_file).convert("RGB")
    st.image(image_raw, caption="Uploaded Image", use_container_width=True)

    # Preprocess with proper DenseNet normalization
    resized = image_raw.resize((224, 224))
    img_array = np.array(resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    preds = model(img_array, training=False).numpy()[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])

    st.subheader(f"Prediction: **{idx_to_class[top_idx]}** ({confidence * 100:.1f}%)")

    # Grad-CAM Visualization
    try:
        last_conv = model.get_layer("relu")
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[last_conv.output, model.output]
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
        st.caption(f"Grad-CAM visualizer unavailable: {e}")
