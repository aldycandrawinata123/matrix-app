import streamlit as st
import numpy as np
import cv2

from utils import (
    load_css,
    show_example_matrices,
    show_convolution_illustration,
    add_background_video,
    to_gray_if_needed,
    apply_affine_transform,
    manual_convolution_2d,
    get_blur_kernel,
    get_sharpen_kernel,
    hsv_background_removal,
)

# ---------- GLOBAL SETUP ----------
st.set_page_config(
    page_title="Matrix Vision Lab",
    page_icon="🛰️",
    layout="wide",
)

load_css()
add_background_video("bg.mp4")


# ---------- HOME / EXPLANATION SECTION ----------
def render_home():
    st.markdown("### Matrix Transformations in 2D Graphics")
    st.write(
        "Geometric transformations can be represented as matrix multiplications "
        "in homogeneous coordinates, allowing multiple transformations to be "
        "composed efficiently."
    )
    show_example_matrices()

    st.markdown("### Convolution for Image Filtering")
    st.write(
        "Convolution applies a small kernel matrix across an image to compute "
        "a weighted sum of neighborhood pixels, enabling blur, sharpen, edge "
        "detection, and more."
    )
    show_convolution_illustration()

    with st.expander("How to use this app"):
        st.markdown(
            """
            - Gunakan **Tools** di bawah untuk upload image dan menerapkan transformasi atau filter.  
            - Atur parameter transformasi (translation, scaling, rotation, dll.) dari panel samping.  
            - Bandingkan **original vs transformed** image secara berdampingan.  
            """
        )


# ---------- TOOLS SECTION ----------
def render_tools():
    st.markdown("### Image Processing Tools")

    tool_options = [
        "Translation",
        "Scaling",
        "Rotation",
        "Shearing",
        "Reflection",
        "Blur",
        "Sharpen",
        "Background Removal",
    ]

    if "selected_tool" not in st.session_state:
        st.session_state.selected_tool = tool_options[0]

    cols = st.columns(len(tool_options))
    for col, name in zip(cols, tool_options):
        with col:
            is_active = st.session_state.selected_tool == name
            label = f"● {name}" if is_active else name
            if st.button(label, key=f"tool-{name}"):
                st.session_state.selected_tool = name

    tool = st.session_state.selected_tool
    st.write("")

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Please upload an image to start.")
        return

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
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        processed = apply_affine_transform(rgb, M)
        st.sidebar.code(f"Translation matrix:\n{M}")

    elif tool == "Scaling":
        st.sidebar.markdown("### Scaling Parameters")
        sx = st.sidebar.slider("Scale X", 0.1, 3.0, 1.2, 0.1)
        sy = st.sidebar.slider("Scale Y", 0.1, 3.0, 1.2, 0.1)
        M = np.array([[sx, 0, 0], [0, sy, 0]], dtype=np.float32)
        processed = apply_affine_transform(rgb, M)
        st.sidebar.code(f"Scaling matrix:\n{M}")

    elif tool == "Rotation":
        st.sidebar.markdown("### Rotation Parameters")
        angle = st.sidebar.slider("Angle (degrees)", -180, 180, 30)
        h, w = rgb.shape[:2]
        cx, cy = w / 2, h / 2
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
        T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        H = T2 @ R @ T1
        M = H[:2, :]
        processed = apply_affine_transform(rgb, M)
        st.sidebar.code(f"Rotation matrix (homogeneous):\n{H}")

    elif tool == "Shearing":
        st.sidebar.markdown("### Shearing Parameters")
        shx = st.sidebar.slider("Shear X", -1.0, 1.0, 0.4, 0.05)
        shy = st.sidebar.slider("Shear Y", -1.0, 1.0, 0.0, 0.05)
        M = np.array([[1, shx, 0], [shy, 1, 0]], dtype=np.float32)
        processed = apply_affine_transform(rgb, M)
        st.sidebar.code(f"Shear matrix:\n{M}")

    elif tool == "Reflection":
        st.sidebar.markdown("### Reflection Parameters")
        axis = st.sidebar.selectbox(
            "Reflection axis",
            ["Vertical (y-axis)", "Horizontal (x-axis)", "Both"],
        )
        h, w = rgb.shape[:2]
        if axis == "Vertical (y-axis)":
            M = np.array([[-1, 0, w], [0, 1, 0]], dtype=np.float32)
        elif axis == "Horizontal (x-axis)":
            M = np.array([[1, 0, 0], [0, -1, h]], dtype=np.float32)
        else:
            M = np.array([[-1, 0, w], [0, -1, h]], dtype=np.float32)
        processed = apply_affine_transform(rgb, M)
        st.sidebar.code(f"Reflection matrix:\n{M}")

    elif tool == "Blur":
        st.sidebar.markdown("### Blur Parameters")
        ksize = st.sidebar.select_slider("Kernel size", options=[3, 5, 7])
        kernel = get_blur_kernel(ksize)
        gray = to_gray_if_needed(rgb)
        blurred = manual_convolution_2d(gray, kernel)
        processed = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
        st.sidebar.code(f"Blur kernel ({ksize}x{ksize}):\n{kernel}")

    elif tool == "Sharpen":
        st.sidebar.markdown("### Sharpen Parameters")
        strength = st.sidebar.slider("Sharpen strength", 0.0, 2.0, 1.0, 0.1)
        kernel = get_sharpen_kernel(strength)
        gray = to_gray_if_needed(rgb)
        sharp = manual_convolution_2d(gray, kernel)
        processed = cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)
        st.sidebar.code(f"Sharpen kernel:\n{kernel}")

    elif tool == "Background Removal":
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


# ---------- TEAM MEMBERS SECTION ----------
def render_team_members():
    st.markdown("### Team Members")

    cols = st.columns(3)

    members = [
        {
            "name": "Aldy Candra Winata",
            "sid": "SID: 004202400230",
            "university": "Universitas Presiden",
            "role": "Project Lead & Backend (All Role)",
            "bio": "Responsible for overall architecture, matrix operations, and convolution logic.",
            "photo": "aldycandrawinata.jpg",
        },
        {
            "name": "Miftahul Khaerunnisa",
            "sid": "SID: 004202400057",
            "university": "Universitas Presiden",
            "role": "Frontend & UI/UX",
            "bio": "Designs the futuristic theme and user interaction flow.",
            "photo": "miftahul.jpg",
        },
        {
            "name": "Fauziah Fithriyani Pamuji",
            "sid": "SID: 004202400007",
            "university": "Universitas Presiden",
            "role": "Image Processing",
            "bio": "Focuses on tuning transformations, filters, and background removal.",
            "photo": "fauziah.jpg",
        },
    ]

    for col, m in zip(cols, members):
        with col:
            st.markdown('<div class="team-wrapper">', unsafe_allow_html=True)

            st.markdown('<div class="avatar-shell">', unsafe_allow_html=True)
            st.image(m["photo"], width=130)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="team-card">
                    <h4 class="team-name">{m['name']}</h4>
                    <p class="team-sid">{m['sid']}</p>
                    <p class="team-univ">{m['university']}</p>
                    <p class="team-role"><span>Role:</span> {m['role']}</p>
                    <p class="team-bio">{m['bio']}</p>
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------- MAIN LAYOUT ----------
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
                        <span class="badge">Blur & Sharpen</span>
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

    st.markdown(
        """
        <div class="section-switch">
            <span class="switch-label">Section</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "section" not in st.session_state:
        st.session_state.section = "App"

    badge_col, _ = st.columns([1, 4])
    with badge_col:
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("APP", key="section-app"):
                st.session_state.section = "App"
        with b2:
            if st.button("TOOLS", key="section-tools"):
                st.session_state.section = "Tools"

    active = st.session_state.section
    thumb_left = "0%" if active == "App" else "50%"

    st.markdown(
        f"""
        <div class="toggle-wrapper">
          <div class="toggle-pill">
            <div class="toggle-thumb" style="left:{thumb_left};"></div>
            <span class="toggle-label toggle-left">App</span>
            <span class="toggle-label toggle-right">Tools</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    choice = st.session_state.section

    st.write("")

    if choice == "App":
        render_home()
    else:
        render_tools()

    st.markdown("---")
    render_team_members()


if __name__ == "__main__":
    main()
