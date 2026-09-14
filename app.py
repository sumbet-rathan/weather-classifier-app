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
    # 1. Load classes
    with open("weather_classes.json", "r") as f:
        labels = json.load(f)
    idx_to_class = {int(k): v for k, v in labels.items()}

    # 2. Download model archive
    model_path = hf_hub_download(
        repo_id="RATHANSUMBET14/weather-vision-app",
        filename="best_weather_model.keras",
        repo_type="space"
    )

    # 3. Unpack .keras archive and patch Keras 3 config for Keras 2
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(model_path, "r") as archive:
        archive.extractall(tmp_dir)

    config_path = os.path.join(tmp_dir, "config.json")
    weights_path = os.path.join(tmp_dir, "model.weights.h5")

    with open(config_path, "r") as f:
        config_str = f.read()

    # Convert Keras 3 "batch_shape" -> Keras 2 "batch_input_shape"
    config_str = config_str.replace('"batch_shape"', '"batch_input_shape"')
    model_config = json.loads(config_str)

    # Reconstruct the EXACT trained model graph
    model = tf.keras.models.model_from_json(json.dumps(model_config))
    model.load_weights(weights_path)

    return model, idx_to_class

model, idx_to_class = load_all()

uploaded_file = st.file_uploader("Upload a weather image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_raw = Image.open(uploaded_file).convert("RGB")
    st.image(image_raw, caption="Uploaded Image", use_container_width=True)

    # Preprocessing with DenseNet normalization
    resized = image_raw.resize((224, 224))
    img_array = np.array(resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Forward pass
    preds = model(img_array, training=False).numpy()[0]
    top_idx = int(np.argmax(preds))

    st.subheader(f"Prediction: **{idx_to_class[top_idx]}** ({preds[top_idx]*100:.1f}%)")

    # Grad-CAM Visualization
    try:
        # Find the last convolutional/relu layer automatically
        target_layer = None
        for layer in reversed(model.layers):
            if hasattr(layer, "layers"):  # Nested functional backbone
                for sub_layer in reversed(layer.layers):
                    if "relu" in sub_layer.name or "conv" in sub_layer.name:
                        target_layer = sub_layer
                        break
            elif "relu" in layer.name or "conv" in layer.name:
                target_layer = layer
                break
            if target_layer:
                break

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
