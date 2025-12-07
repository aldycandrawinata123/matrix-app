import streamlit as st
import numpy as np
import cv2
import base64


def load_css():
    css = """
    :root {
        --bg-gradient: radial-gradient(circle at top left, #1f2933, #020617);
        --accent: #38bdf8;
        --accent-soft: rgba(56,189,248,0.15);
        --accent-strong: #0ea5e9;
        --card-bg: rgba(15,23,42,0.30);
        --card-border: rgba(148,163,184,0.45);
        --text-main: #e5e7eb;
        --text-soft: #9ca3af;
        --danger: #f97373;
    }

    .stApp, .stMarkdown, .stText, .stRadio, .stSelectbox, .stSlider {
        color: var(--text-main) !important;
    }

    .stSidebar, .stSidebar * {
        color: var(--text-main) !important;
    }

    .main {
        background: transparent !important;
    }

    .stApp {
        background: transparent !important;
        color: var(--text-main);
    }

    /* Card kaca iOS */
    .hero-card,
    .card {
        padding: 2rem 2.4rem;
        border-radius: 1.7rem;
        background: radial-gradient(circle at 0% 0%, rgba(255,255,255,0.16), rgba(15,23,42,0.7));
        border: 1px solid rgba(148,163,184,0.5);
        box-shadow:
            0 0 0 1px rgba(15,23,42,0.65),
            0 26px 60px rgba(15,23,42,0.95);
        backdrop-filter: blur(26px);
        -webkit-backdrop-filter: blur(26px);
    }

    .hero-card.secondary {
        background: radial-gradient(circle at 0% 0%, rgba(191,219,254,0.22), rgba(15,23,42,0.9));
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        background: linear-gradient(to right, #e0f2fe, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: #ffffff !important;
        max-width: 34rem;
        margin-bottom: 1.1rem;
    }

    /* Badge hero */
    .hero-badges .badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.28rem 0.95rem;
        border-radius: 999px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #f9fafb;
        margin-right: 0.45rem;
        margin-bottom: 0.45rem;
        background: radial-gradient(circle at 30% 0%, rgba(255,255,255,0.04), rgba(15,23,42,0.10));
        box-shadow:
            0 4px 14px rgba(15,23,42,0.45),
            inset 0 0 6px rgba(15,23,42,0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .hero-badges .badge:hover {
        background: radial-gradient(circle at 20% 0%, rgba(191,219,254,0.30), rgba(15,23,42,0.35));
        border-color: rgba(226,232,240,0.95);
        box-shadow:
            0 0 0 1px rgba(59,130,246,0.85),
            0 14px 32px rgba(15,23,42,0.95);
    }

    /* Team section wrapper */
    .team-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.6rem;
        margin-top: 1.5rem;
        width: 100%;
    }

    /* Shell buat posisi avatar */
    .avatar-shell {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: -1.6rem;
        z-index: 2;
        width: 100%;
    }

    /* Container stImage untuk memastikan terpusat */
    .avatar-shell .stImage {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        margin: 0 auto !important;
    }

    /* st.image avatar -> bulat, ukuran fix */
    .stImage img {
        width: 130px !important;
        height: 130px !important;
        border-radius: 50% !important;
        object-fit: cover !important;
        object-position: center center !important;
        box-shadow: 0 18px 40px rgba(15,23,42,0.85);
        border: 1px solid rgba(148,163,184,0.6);
        margin: 0 auto !important;
        display: block !important;
    }

    .team-card {
        position: relative;
        top: 0.4rem;
        width: 100%;
        max-width: 360px;
        padding: 1.1rem 1.4rem;
        border-radius: 1.4rem;
        background: radial-gradient(circle at 0% 0%, rgba(255,255,255,0.12), rgba(15,23,42,0.80));
        border: 1px solid rgba(148,163,184,0.55);
        box-shadow:
            0 0 0 1px rgba(15,23,42,0.70),
            0 18px 36px rgba(15,23,42,0.95);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        text-align: left;
    }

    .team-name {
        margin: 0 0 0.2rem 0;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        color: #f9fafb;
    }

    .team-sid {
        margin: 0;
        font-size: 0.86rem;
        color: #a5b4fc;
        font-weight: 600;
    }

    .team-univ {
        margin: 0 0 0.3rem 0;
        font-size: 0.86rem;
        color: #e5e7eb;
    }

    .team-role {
        margin: 0 0 0.4rem 0;
        font-size: 0.9rem;
        color: #e5e7eb;
    }

    .team-role span {
        font-weight: 700;
    }

    .team-bio {
        margin: 0;
        font-size: 0.86rem;
        color: #d1d5db;
    }

    .stSidebar {
        background: linear-gradient(180deg, rgba(15,23,42,0.97), rgba(15,23,42,0.9));
        border-right: 1px solid rgba(31,41,55,1);
    }

    .stFileUploader label {
        color: #ffffff !important;
        font-weight: 600;
    }

    .stSlider > div > div > div[role="slider"] {
        box-shadow: 0 0 0 2px rgba(56,189,248,0.9);
        background: var(--accent);
    }

    .stRadio > label, .stSelectbox > label {
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    h2, h3, h4 {
        color: #e5e7eb;
    }

    .matrix-code {
        font-size: 0.8rem !important;
        color: var(--text-soft) !important;
    }

    .stCode, .stCode > div, .stCode pre {
        background: radial-gradient(circle at 0% 0%, rgba(255,255,255,0.14), rgba(15,23,42,0.35)) !important;
        border-radius: 1.2rem !important;
        border: 1px solid rgba(148,163,184,0.55) !important;
        box-shadow:
            0 0 0 1px rgba(15,23,42,0.65),
            0 18px 40px rgba(15,23,42,0.9) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        color: #e5e7eb !important;
    }

    /* Expander kaca */
    .st-expander {
        background: rgba(15, 23, 42, 0.24) !important;
        border-radius: 999px !important;
        border: 1px solid rgba(148, 163, 184, 0.4) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        padding: 0.15rem 0.6rem !important;
    }

    .st-expanderHeader {
        background: transparent !important;
        color: #e5e7eb !important;
        font-weight: 600 !important;
        border-radius: 999px !important;
        border: none !important;
    }

    .st-expanderHeader * {
        color: #e5e7eb !important;
    }

    .st-expanderHeader:focus,
    .st-expanderHeader:hover {
        outline: none !important;
        box-shadow: 0 0 0 1px rgba(148, 163, 184, 0.7) !important;
        background: rgba(15, 23, 42, 0.5) !important;
    }

    /* Label Section */
    .section-switch {
        margin-top: 1.5rem;
        margin-bottom: 0.4rem;
    }

    .section-switch .switch-label {
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-soft);
    }

    /* Tombol umum */
    .stButton > button {
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        border: 1px solid rgba(148,163,184,0.6);
        background: radial-gradient(circle at 30% 0%, rgba(255,255,255,0.10), rgba(15,23,42,0.40));
        color: #e5e7eb;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        box-shadow:
            0 4px 14px rgba(15,23,42,0.55),
            inset 0 0 8px rgba(15,23,42,0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }

    .stButton > button:hover {
        border-color: rgba(191,219,254,0.9);
        background: radial-gradient(circle at 20% 0%, rgba(191,219,254,0.45), rgba(15,23,42,0.55));
        box-shadow:
            0 0 0 1px rgba(59,130,246,0.85),
            0 16px 32px rgba(15,23,42,0.95);
    }

    /* Toggle App / Tools */
    .toggle-wrapper {
        margin-top: 0.2rem;
    }

    .toggle-pill {
        position: relative;
        width: 230px;
        height: 34px;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.7);
        background: radial-gradient(circle at 0% 0%, rgba(255,255,255,0.12), rgba(15,23,42,0.8));
        box-shadow:
            0 0 0 1px rgba(15,23,42,0.8),
            0 16px 32px rgba(15,23,42,0.9);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 6px;
        box-sizing: border-box;
    }

    .toggle-thumb {
        position: absolute;
        top: 3px;
        width: 50%;
        height: 28px;
        border-radius: 999px;
        background: radial-gradient(circle at 30% 0%, rgba(251,113,133,0.95), rgba(59,130,246,0.85));
        box-shadow:
            0 0 0 1px rgba(15,23,42,0.9),
            0 14px 30px rgba(15,23,42,0.95);
        transition: left 0.25s ease;
        z-index: 1;
    }

    .toggle-label {
        position: relative;
        z-index: 2;
        flex: 1;
        text-align: center;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #e5e7eb;
        user-select: none;
    }

    .toggle-left {
        text-align: left;
        padding-left: 6px;
    }

    .toggle-right {
        text-align: right;
        padding-right: 6px;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def add_background_video(video_file: str):
    with open(video_file, "rb") as f:
        video_bytes = f.read()
    encoded = base64.b64encode(video_bytes).decode()

    video_html = f"""
    <style>
    #bg-video {{
      position: fixed;
      right: 0;
      bottom: 0;
      min-width: 100%;
      min-height: 100%;
      z-index: -1;
      object-fit: cover;
    }}
    </style>

    <video autoplay muted loop id="bg-video">
      <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)


def show_example_matrices():
    st.markdown("##### Example 2D homogeneous transformation matrices")
    tx, ty = 30, 10
    translation = np.array([[1, 0, tx],
                            [0, 1, ty],
                            [0, 0, 1]], dtype=float)
    theta = np.deg2rad(30)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta),  np.cos(theta), 0],
                         [0, 0, 1]], dtype=float)
    scaling = np.array([[1.2, 0, 0],
                        [0, 0.8, 0],
                        [0, 0, 1]], dtype=float)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Translation")
        st.code(translation, language="text")
    with col2:
        st.caption("Rotation")
        st.code(rotation, language="text")
    with col3:
        st.caption("Scaling")
        st.code(scaling, language="text")


def show_convolution_illustration():
    st.markdown(
        """
        A convolution kernel slides over the image, multiplies each neighbor pixel
        by a weight, and sums them to produce the new pixel value. Small kernels
        such as 3×3 are commonly used for simple blur or sharpen filters.
        """
    )


def apply_affine_transform(img_rgb, M):
    h, w = img_rgb.shape[:2]
    warped = cv2.warpAffine(
        img_rgb, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


def to_gray_if_needed(rgb):
    if len(rgb.shape) == 3:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb


def manual_convolution_2d(img, kernel):
    if img.ndim != 2:
        raise ValueError("manual_convolution_2d expects a 2D (grayscale) image.")

    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    padded = np.pad(
        img,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0,
    )
    out = np.zeros_like(img, dtype=np.float32)

    kernel_flipped = np.flipud(np.fliplr(kernel))

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            region = padded[y:y + k_h, x:x + k_w]
            out[y, x] = np.sum(region * kernel_flipped)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def get_blur_kernel(ksize=3):
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    kernel /= kernel.size
    return kernel


def get_sharpen_kernel(strength=1.0):
    identity = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=np.float32)
    neighbors = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]], dtype=np.float32)
    kernel = identity * (1 + 4 * strength) - strength * neighbors
    return kernel


def hsv_background_removal(rgb_img, lower_hsv, upper_hsv):
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    fg = cv2.bitwise_and(bgr, mask_3c)
    out_rgb = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    return out_rgb
