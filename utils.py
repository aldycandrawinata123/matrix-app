import streamlit as st
import numpy as np
import cv2


def load_css():
    css = """
    :root {
        --bg-gradient: radial-gradient(circle at top left, #1f2933, #020617);
        --accent: #38bdf8;
        --accent-soft: rgba(56,189,248,0.15);
        --accent-strong: #0ea5e9;
        --card-bg: rgba(15,23,42,0.86);
        --card-border: rgba(148,163,184,0.35);
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


    .hero-card {
        padding: 2rem 2.4rem;
        border-radius: 1.5rem;
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(15,23,42,0.75));
        border: 1px solid var(--card-border);
        box-shadow: 0 18px 55px rgba(15,23,42,0.9);
        backdrop-filter: blur(28px);
    }

    .hero-card.secondary {
        background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(8,47,73,0.8));
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
        color: var(--text-soft);
        max-width: 34rem;
        margin-bottom: 1.1rem;
    }

    .hero-badges .badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(56,189,248,0.9);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #e0f2fe;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
    background: radial-gradient(circle at top left, rgba(56,189,248,0.18), rgba(15,23,42,0.95));
}


    .card {
        padding: 1.4rem 1.6rem;
        border-radius: 1.2rem;
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        box-shadow: 0 14px 35px rgba(15,23,42,0.9);
        backdrop-filter: blur(22px);
        margin-bottom: 1rem;
    }

    .team-card {
        text-align: center;
    }

    .avatar-placeholder {
        width: 120px;
        height: 120px;
        border-radius: 999px;
        background: radial-gradient(circle at 30% 0%, #38bdf8, #0f172a);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 0.8rem auto;
        font-size: 2.4rem;
        border: 1px solid rgba(148,163,184,0.6);
        box-shadow: 0 18px 40px rgba(15,23,42,0.85);
    }

    .stSidebar {
        background: linear-gradient(180deg, rgba(15,23,42,0.97), rgba(15,23,42,0.9));
        border-right: 1px solid rgba(31,41,55,1);
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
        color: var(--text-soft);
    }

    /* Expander: gelap, teks putih */
    /* Ubah style file uploader dan expander jadi biru gelap */
.stFileUploader, .st-expander {
    background-color: #020617 !important;      /* biru sangat gelap */
    border-radius: 0.9rem !important;
    border: 1px solid #0f172a !important;      /* border biru gelap */
}

/* Header expander */
.st-expanderHeader {
    background-color: #020617 !important;
    color: #f9fafb !important;
    font-weight: 700 !important;
    border-radius: 0.9rem !important;
    border: 1px solid #0f172a !important;
}

/* Teks di header expander tetap putih */
.st-expanderHeader * {
    color: #f9fafb !important;
}


    .st-expanderHeader:focus,
    .st-expanderHeader:hover {
        outline: none !important;
        box-shadow: none !important;
    }
        /* Bikin label di atas uploader (Upload an image) jadi putih */
    .stFileUploader label {
        color: #ffffff !important;
        font-weight: 600;
    }

    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


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
    """
    Manual 2D convolution for a single-channel image (grayscale).
    Uses zero padding and 'same' output size.
    """
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
    """
    Simple background removal using HSV in-range masking.
    Keeps pixels within the specified HSV range and makes others black.
    """
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    fg = cv2.bitwise_and(bgr, mask_3c)
    out_rgb = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    return out_rgb
