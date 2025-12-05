import streamlit as st
from utils import load_css, add_background_video

st.set_page_config(page_title="Team - Matrix Vision Lab", page_icon="👥", layout="wide")
load_css()
add_background_video("bg.mp4")  # panggil background video di page ini juga

st.markdown("## 👥 Team Members")

cols = st.columns(3)

members = [
    {
        "name": "Nama Anggota 1",
        "role": "Project Lead & Backend",
        "bio": "Responsible for overall architecture, matrix operations, and convolution logic.",
        "image": None,
    },
    {
        "name": "Nama Anggota 2",
        "role": "Frontend & UI/UX",
        "bio": "Designs the futuristic theme and user interaction flow.",
        "image": None,
    },
    {
        "name": "Nama Anggota 3",
        "role": "Image Processing",
        "bio": "Focuses on tuning transformations, filters, and background removal.",
        "image": None,
    },
]

for col, m in zip(cols, members):
    with col:
        st.markdown('<div class="card team-card">', unsafe_allow_html=True)
        if m["image"]:
            st.image(m["image"], use_column_width=True)
        else:
            st.markdown(
                '<div class="avatar-placeholder">👤</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f"### {m['name']}")
        st.markdown(f"**Role:** {m['role']}")
        st.write(m["bio"])
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### 🧩 How the app works")
st.write(
    """
    - Images are loaded and processed as NumPy arrays.  
    - Geometric transformations are implemented via 2×3 affine matrices derived from 3×3 homogeneous coordinates.  
    - Blur and sharpen filters are computed with manual 2D convolution over gray-scale images.  
    - Optional background removal uses HSV color thresholding to separate foreground from background.
    """
)
