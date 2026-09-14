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

def load_weights_by_shape(model, h5_file_path):
    """Walks the h5 weights file and sets weights matching layer shapes."""
    with h5py.File(h5_file_path, "r") as f:
        weight_tensors = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                weight_tensors.append(np.array(obj))
        f.visititems(visitor)

    # Assign weights sequentially to layers that expect them
    idx = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        new_weights = []
        for w in weights:
            if idx < len(weight_tensors) and weight_tensors[idx].shape == w.shape:
                new_weights.append(weight_tensors[idx])
                idx += 1
            else:
                # If exact shape isn't at idx, search upcoming matches
                match_found = False
                for j in range(idx, len(weight_tensors)):
                    if weight_tensors[j].shape == w.shape:
                        new_weights.append(weight_tensors[j])
                        weight_tensors.pop(j)
                        match_found = True
                        break
                if not match_found:
                    new_weights.append(w)
        layer.set_weights(new_weights)

@st.cache_resource
def load_all():
    # 1. Load labels
    with open("weather_classes.json", "r") as f:
        labels = json.load(f)
    idx_to_class = {int(k): v for k, v in labels.items()}
    num_classes = len(idx_to_class)

    # 2. Download model archive
    model_path = hf_hub_download(
        repo_id="RATHANSUMBET14/weather-vision-app",
        filename="best_weather_model.keras",
        repo_type="space"
    )

    # 3. Extract weights
    tmp_dir = tempfile.mkdtemp()
    weights_path = os.path.join(tmp_dir, "model.weights.h5")
    with zipfile.ZipFile(model_path, "r") as archive:
        archive.extract("model.weights.h5", path=tmp_dir)

    # 4. Reconstruct DenseNet121 architecture
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

    # 5. Populate model weights
    load_weights_by_shape(model, weights_path)

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

    # Grad-CAM
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
        st.caption(f"Grad-CAM visualizer unavailable for this sample: {e}")
