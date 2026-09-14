import os
# Set Keras backend to Torch CPU before importing keras
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
    # 1. Download trained Keras 3 model directly
    model_path = hf_hub_download(
        repo_id="RATHANSUMBET14/weather-vision-app",
        filename="best_weather_model.keras",
        repo_type="space"
    )
    # Native Keras 3 direct load (no compile needed for inference)
    model = keras.models.load_model(model_path, compile=False)

    # 2. Load class labels
    with open("weather_classes.json", "r") as f:
        labels = json.load(f)
    idx_to_class = {int(k): v for k, v in labels.items()}

    return model, idx_to_class

model, idx_to_class = load_all()

uploaded_file = st.file_uploader("Upload a weather image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_raw = Image.open(uploaded_file).convert("RGB")
    st.image(image_raw, caption="Uploaded Image", use_container_width=True)

    # Preprocessing: DenseNet ImageNet normalization: (x / 255 - mean) / std
    resized = image_raw.resize((224, 224))
    img_np = np.array(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_np - mean) / std
    img_tensor = np.expand_dims(img_norm, axis=0)

    # Forward pass
    preds = model(img_tensor, training=False)
    preds = np.array(preds)[0]

    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])

    st.subheader(f"Prediction: **{idx_to_class[top_idx]}** ({confidence * 100:.1f}%)")

    # Grad-CAM Visualization
    try:
        # Locate target convolution layer in DenseNet121
        target_layer = None
        for layer in reversed(model.layers):
            if hasattr(layer, "layers"):
                for sub in reversed(layer.layers):
                    if "relu" in sub.name or "conv" in sub.name:
                        target_layer = sub
                        break
            elif "relu" in layer.name or "conv" in layer.name:
                target_layer = layer
                break
            if target_layer:
                break

        if target_layer is not None:
            feature_extractor = keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output]
            )

            x_torch = torch.tensor(img_tensor, requires_grad=True)
            conv_out, out_logits = feature_extractor(x_torch)
            loss = out_logits[0, top_idx]
            loss.backward()

            grads = x_torch.grad
            pooled_grads = torch.mean(conv_out, dim=(0, 1, 2)).detach().numpy()
            conv_out_np = conv_out[0].detach().numpy()

            for i in range(conv_out_np.shape[-1]):
                conv_out_np[:, :, i] *= pooled_grads[i]

            heatmap = np.mean(conv_out_np, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= (np.max(heatmap) + 1e-10)

            heatmap = np.uint8(255 * heatmap)
            jet = plt.get_cmap("jet")(np.arange(256))[:, :3]
            jet_heatmap = Image.fromarray(np.uint8(jet[heatmap] * 255)).resize(image_raw.size)
            overlay = Image.blend(image_raw, jet_heatmap, alpha=0.4)

            st.image(overlay, caption="Grad-CAM Attention Map", use_container_width=True)
    except Exception as e:
        st.caption(f"Grad-CAM visualizer unavailable for this sample: {e}")
