import os
# Use PyTorch CPU backend for Keras 3 (no TensorFlow C-library issues)
os.environ["KERAS_BACKEND"] = "torch"

import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import keras
import torch
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Weather Classifier", layout="centered")
st.title("🌦️ Weather Classifier (DenseNet121 + Grad-CAM)")

@st.cache_resource
def load_all():
    # 1. Download model file from Hugging Face
    model_path = hf_hub_download(
        repo_id="RATHANSUMBET14/weather-vision-app",
        filename="best_weather_model.keras",
        repo_type="space"
    )
    # Native Keras 3 loader - loads cleanly without compile issues
    model = keras.models.load_model(model_path, compile=False)
    
    # 2. Load classes
    with open("weather_classes.json", "r") as f:
        labels = json.load(f)
    idx_to_class = {int(k): v for k, v in labels.items()}
    
    return model, idx_to_class

model, idx_to_class = load_all()

uploaded_file = st.file_uploader("Upload a weather image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_raw = Image.open(uploaded_file).convert("RGB")
    st.image(image_raw, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    resized = image_raw.resize((224, 224))
    img_array = np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, axis=0)

    # Inference using Keras 3
    preds = model(img_array, training=False)
    preds = np.array(preds)[0]
    top_idx = int(np.argmax(preds))

    st.subheader(f"Prediction: **{idx_to_class[top_idx]}** ({preds[top_idx]*100:.1f}%)")

    # Grad-CAM Implementation
    try:
        # Locate target convolution layer
        target_layer = None
        for layer in reversed(model.layers):
            if hasattr(layer, "layers"):
                for sub_layer in reversed(layer.layers):
                    if "relu" in sub_layer.name or "conv" in sub_layer.name:
                        target_layer = sub_layer
                        break
            elif "relu" in layer.name or "conv" in layer.name:
                target_layer = layer
                break
            if target_layer:
                break

        if target_layer is not None:
            # Build feature extraction model
            feature_model = keras.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
            
            # Forward pass with PyTorch autograd
            tensor_in = torch.tensor(img_array, requires_grad=True)
            features, outputs = feature_model(tensor_in)
            score = outputs[0, top_idx]
            score.backward()

            grads = tensor_in.grad
            pooled_grads = torch.mean(features, dim=(0, 1, 2)).detach().numpy()
            feature_np = features[0].detach().numpy()

            for i in range(feature_np.shape[-1]):
                feature_np[:, :, i] *= pooled_grads[i]

            heatmap = np.mean(feature_np, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= (np.max(heatmap) + 1e-10)

            heatmap = np.uint8(255 * heatmap)
            jet = plt.get_cmap("jet")(np.arange(256))[:, :3]
            jet_heatmap = Image.fromarray(np.uint8(jet[heatmap] * 255)).resize(image_raw.size)
            overlay = Image.blend(image_raw, jet_heatmap, alpha=0.4)

            st.image(overlay, caption="Grad-CAM Attention Map", use_container_width=True)
    except Exception as e:
        st.info("Grad-CAM visualization skipped for this prediction.")
