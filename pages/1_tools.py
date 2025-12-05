import streamlit as st
import numpy as np
import cv2
from utils import (
    load_css,
    to_gray_if_needed,
    apply_affine_transform,
    manual_convolution_2d,
    get_blur_kernel,
    get_sharpen_kernel,
    hsv_background_removal,
    add_background_video,  # penting untuk background video
)

st.set_page_config(page_title="Tools - Matrix Vision Lab", page_icon="🛠️", layout="wide")
load_css()
add_background_video("bg.mp4")  # panggil video background

st.markdown("## 🛠️ Image Processing Tools")

tool = st.sidebar.radio(
    "Choose tool",
    [
        "Translation",
        "Scaling",
        "Rotation",
        "Shearing",
        "Reflection",
        "Blur (Convolution)",
        "Sharpen (Convolution)",
        "Background Removal (Optional)",
    ],
    index=0,
)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Please upload an image to start.")
    st.stop()

# Read image as BGR then convert to RGB
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Original")
    st.image(rgb, use_column_width=True)

processed = rgb.copy()

if tool == "Translation":
    st.sidebar.markdown("### Translation Parameters")
    tx = st.sidebar.slider("Shift X (pixels)", -300, 300, 50)
    ty = st.sidebar.slider("Shift Y (pixels)", -300, 300, 50)

    M = np.array([[1, 0, tx],
                  [0, 1, ty]], dtype=np.float32)
    processed = apply_affine_transform(rgb, M)

    st.sidebar.code(f"Translation matrix:\n{M}")

elif tool == "Scaling":
    st.sidebar.markdown("### Scaling Parameters")
    sx = st.sidebar.slider("Scale X", 0.1, 3.0, 1.2, 0.1)
    sy = st.sidebar.slider("Scale Y", 0.1, 3.0, 1.2, 0.1)

    M = np.array([[sx, 0, 0],
                  [0, sy, 0]], dtype=np.float32)
    processed = apply_affine_transform(rgb, M)

    st.sidebar.code(f"Scaling matrix:\n{M}")

elif tool == "Rotation":
    st.sidebar.markdown("### Rotation Parameters")
    angle = st.sidebar.slider("Angle (degrees)", -180, 180, 30)
    h, w = rgb.shape[:2]
    cx, cy = w / 2, h / 2

    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Translate to origin, rotate, translate back
    T1 = np.array([[1, 0, -cx],
                   [0, 1, -cy],
                   [0, 0, 1]])
    R = np.array([[cos_t, -sin_t, 0],
                  [sin_t,  cos_t, 0],
                  [0,      0,     1]])
    T2 = np.array([[1, 0, cx],
                   [0, 1, cy],
                   [0, 0, 1]])
    H = T2 @ R @ T1
    M = H[:2, :]
    processed = apply_affine_transform(rgb, M)

    st.sidebar.code(f"Rotation matrix (homogeneous):\n{H}")

elif tool == "Shearing":
    st.sidebar.markdown("### Shearing Parameters")
    shx = st.sidebar.slider("Shear X", -1.0, 1.0, 0.4, 0.05)
    shy = st.sidebar.slider("Shear Y", -1.0, 1.0, 0.0, 0.05)

    M = np.array([[1, shx, 0],
                  [shy, 1, 0]], dtype=np.float32)
    processed = apply_affine_transform(rgb, M)

    st.sidebar.code(f"Shear matrix:\n{M}")

elif tool == "Reflection":
    st.sidebar.markdown("### Reflection Parameters")
    axis = st.sidebar.selectbox("Reflection axis", ["Vertical (y-axis)", "Horizontal (x-axis)", "Both"])
    h, w = rgb.shape[:2]

    if axis == "Vertical (y-axis)":
        M = np.array([[-1, 0, w],
                      [0, 1, 0]], dtype=np.float32)
    elif axis == "Horizontal (x-axis)":
        M = np.array([[1, 0, 0],
                      [0, -1, h]], dtype=np.float32)
    else:
        M = np.array([[-1, 0, w],
                      [0, -1, h]], dtype=np.float32)

    processed = apply_affine_transform(rgb, M)
    st.sidebar.code(f"Reflection matrix:\n{M}")

elif tool == "Blur (Convolution)":
    st.sidebar.markdown("### Blur Parameters")
    ksize = st.sidebar.select_slider("Kernel size", options=[3, 5, 7])
    kernel = get_blur_kernel(ksize)

    gray = to_gray_if_needed(rgb)
    blurred = manual_convolution_2d(gray, kernel)
    processed = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)

    st.sidebar.code(f"Blur kernel ({ksize}x{ksize}):\n{kernel}")

elif tool == "Sharpen (Convolution)":
    st.sidebar.markdown("### Sharpen Parameters")
    strength = st.sidebar.slider("Sharpen strength", 0.0, 2.0, 1.0, 0.1)
    kernel = get_sharpen_kernel(strength)

    gray = to_gray_if_needed(rgb)
    sharp = manual_convolution_2d(gray, kernel)
    processed = cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)

    st.sidebar.code(f"Sharpen kernel:\n{kernel}")

elif tool == "Background Removal (Optional)":
    st.sidebar.markdown("### Background Removal (HSV Threshold)")
    st.sidebar.write("Tune HSV range to keep foreground and remove background.")
    h_min, h_max = st.sidebar.slider("Hue range", 0, 179, (20, 160))
    s_min, s_max = st.sidebar.slider("Saturation range", 0, 255, (40, 255))
    v_min, v_max = st.sidebar.slider("Value range", 0, 255, (40, 255))

    processed = hsv_background_removal(
        rgb,
        (h_min, s_min, v_min),
        (h_max, s_max, v_max),
    )

with col2:
    st.markdown("#### Transformed / Processed")
    st.image(processed, use_column_width=True)
