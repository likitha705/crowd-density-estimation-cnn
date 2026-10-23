"""Venue Crowd-Safety Dashboard (Streamlit)."""
import os

import matplotlib.pyplot as plt
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

    left, right = st.columns(2)
    left.image(image, caption="Input image", use_container_width=True)
    fig, ax = plt.subplots()
    ax.imshow(density, cmap="jet")
    ax.axis("off")
    right.pyplot(fig)
    right.caption("Predicted density map")
else:
    st.info("Upload an image to estimate the crowd count.")
