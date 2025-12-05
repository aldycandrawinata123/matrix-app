import streamlit as st
import base64
from utils import load_css, show_example_matrices, show_convolution_illustration

st.set_page_config(
    page_title="Matrix Vision Lab",
    page_icon="🛰️",
    layout="wide",
)

load_css()
def add_background_video(video_file):
    video_bytes = open(video_file, "rb").read()
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
      Your browser does not support HTML5 video.
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)

# panggil di awal halaman
add_background_video("bg.mp4")


def main():
    with st.container():
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown(
                """
                <div class="hero-card">
                    <h1 class="hero-title">Matrix Vision Lab</h1>
                    <p class="hero-subtitle">
                        Explore 2D matrix transformations & convolution-based image processing
                        in an interactive, futuristic Streamlit environment.
                    </p>
                    <div class="hero-badges">
                        <span class="badge">Translation</span>
                        <span class="badge">Scaling</span>
                        <span class="badge">Rotation</span>
                        <span class="badge">Shearing</span>
                        <span class="badge">Reflection</span>
                        <span class="badge badge-accent">Blur & Sharpen</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            st.markdown(
                """
                <div class="hero-card secondary">
                    <h3>Project Overview</h3>
                    <p>
                        This application demonstrates how matrix operations and convolution
                        can be used for basic image processing and 2D graphics.
                    </p>
                    <ul>
                        <li>Matrix-based geometric transformations in homogeneous coordinates.</li>
                        <li>Manual convolution kernels for blur and sharpen filters.</li>
                        <li>Optional background removal using color-based segmentation.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 🔢 Matrix Transformations in 2D Graphics")
    st.write(
        "Geometric transformations can be represented as matrix multiplications in homogeneous coordinates, allowing multiple transformations to be composed efficiently."
    )
    show_example_matrices()

    st.markdown("### 🧮 Convolution for Image Filtering")
    st.write(
        "Convolution applies a small kernel matrix across an image to compute a weighted sum of neighborhood pixels, enabling blur, sharpen, edge detection, and more."
    )
    show_convolution_illustration()

    with st.expander("📚 How to use this app"):
        st.markdown(
            """
            - Go to **Image Processing Tools** page to upload an image and apply transformations or filters.  
            - Use the **sidebar** to switch between tools (translation, scaling, rotation, etc.).  
            - Compare **original vs transformed** images side-by-side.  
            - Visit **Team Members** page for project info and group details.
            """
        )

if __name__ == "__main__":
    main()
