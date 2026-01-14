import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import torch
from PIL import Image
import streamlit as st

from src.inference import SegmentationModel

MODEL_PATH = "models/segmentation_model.pth"

@st.cache_resource
def load_model():
    model = SegmentationModel(model_path=MODEL_PATH, device="cpu")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    metrics = {
        "epoch": checkpoint.get("epoch"),
        "val_dice": checkpoint.get("val_dice"),
        "val_iou": checkpoint.get("val_iou")
    }
    return model, metrics

st.set_page_config(page_title="ISIC Segmentation", layout="wide")
st.title("Сегментация кожных образований")

model, metrics = load_model()

st.sidebar.header("Метрики модели")
if metrics["epoch"] is not None:
    st.sidebar.metric("Epoch", metrics["epoch"])
if metrics["val_dice"] is not None:
    st.sidebar.metric("Validation Dice", f"{metrics['val_dice']:.4f}")
if metrics["val_iou"] is not None:
    st.sidebar.metric("Validation IoU", f"{metrics['val_iou']:.4f}")

uploaded = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])

if uploaded:
    image_bytes = uploaded.read()
    original = Image.open(io.BytesIO(image_bytes))
    original_size = original.size
    
    mask = model.predict_segmentation(image_bytes)
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.resize(original_size, Image.NEAREST)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Оригинал")
        st.image(original, use_container_width=True)
    with col2:
        st.subheader("Маска сегментации")
        st.image(mask_img, use_container_width=True)
