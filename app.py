"""Venue Crowd-Safety Dashboard (Streamlit)."""
import os
 
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
 
from utils import load_model, predict, safety_level
 
WEIGHTS = "weights/model.pth"
 
st.set_page_config(page_title="Crowd-Safety Dashboard", page_icon="👥", layout="wide")
st.title("👥 Venue Crowd-Safety Dashboard")
st.caption("CNN density-map regression (CSRNet) for crowd counting from surveillance images")
 
st.sidebar.header("Settings")
venue = st.sidebar.text_input("Venue name", "Main Hall")
threshold = st.sidebar.slider("Safe capacity (people)", 10, 5000, 500, step=10)
alpha = st.sidebar.slider("Heatmap overlay strength", 0.1, 0.9, 0.5, step=0.1)
 
 
@st.cache_resource
def get_model():
    return load_model(WEIGHTS)
 
 
if not os.path.exists(WEIGHTS):
    st.warning(f"Trained weights not found at `{WEIGHTS}`. Run train.py first and add the file to the repo.")
    st.stop()
 
file = st.file_uploader("Upload a crowd image", type=["jpg", "jpeg", "png"])
 
if file:
    image = Image.open(file).convert("RGB")
    count, density = predict(get_model(), image)
    level, advice = safety_level(count, threshold)
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated crowd count", f"{count:.0f}")
    c2.metric("Safe capacity", threshold)
    c3.metric("Occupancy", f"{100 * count / threshold:.0f}%")
 
    {"SAFE": st.success, "WARNING": st.warning, "DANGER": st.error}[level](f"**{venue}: {level}**. {advice}")
 
    # Density map resized to the image so it can be overlaid on top of it
    d = np.clip(density, 0, None)
    d = d / d.max() if d.max() > 0 else d
    heat = Image.fromarray((d * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)
    heat = np.asarray(heat, dtype=np.float32) / 255.0
 
    col1, col2, col3 = st.columns(3)
    col1.image(image, caption="Input image", use_container_width=True)
 
    fig, ax = plt.subplots()
    ax.imshow(heat, cmap="jet")
    ax.axis("off")
    col2.pyplot(fig)
    col2.caption("Predicted density map (regression output)")
 
    fig2, ax2 = plt.subplots()
    ax2.imshow(image)
    ax2.imshow(heat, cmap="jet", alpha=alpha)
    ax2.axis("off")
    col3.pyplot(fig2)
    col3.caption("Density map overlaid on image")
 
    st.caption(
        "The model regresses a density value for every region of the image. "
        "Summing the density map gives the estimated crowd count."
    )
else:
    st.info("Upload an image to estimate the crowd count.")
 
