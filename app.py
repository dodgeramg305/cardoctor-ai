import streamlit as st
import numpy as np
import cv2
from io import BytesIO

st.set_page_config(page_title="CarDoctor-AI", layout="wide")

# ---------------------------
# Custom Page Header
# ---------------------------
st.markdown("""
<style>
.title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    margin-bottom: -10px;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}
.score-badge {
    display: inline-block;
    padding: 8px 18px;
    background: #2E86C1;
    color: white;
    border-radius: 10px;
    font-size: 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">CarDoctor-AI — Vehicle Damage Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a car image and detect possible damage areas using OpenCV</div>', unsafe_allow_html=True)

# ----------------------------------------
# Helper: Convert BGR (OpenCV) to RGB
# ----------------------------------------
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ----------------------------------------
# Image Processing Function
# ----------------------------------------
def process_image(image_bgr):
    output = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    image_area = h * w
    min_area = image_area * 0.005   # requires larger contour (reduces box clutter)

    total_damage_area = 0
    damage_regions = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x+cw, y+ch), (255, 0, 0), 3)
            total_damage_area += area
            damage_regions += 1

    damage_ratio = total_damage_area / image_area
    raw_score = damage_ratio * 25  
    damage_score = int(np.clip(np.round(raw_score), 1, 10)) if damage_regions > 0 else 1

    return output, edges, damage_score, damage_regions


uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.divider()
    st.subheader("1. Preview of Uploaded Image")

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(bgr_to_rgb(image_bgr), caption="Original Image", use_container_width=True)

    if st.button("Analyze Image"):
        processed_bgr, edges, score, regions = process_image(image_bgr)

        st.divider()
        st.subheader("2. Detection Results")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(bgr_to_rgb(processed_bgr), caption="Damage Highlighted", use_container_width=True)
        with col2:
            st.image(edges, caption="Edge Map", use_container_width=True, clamp=True)

        st.markdown(f"<br><span class='score-badge'>Damage Score: {score} / 10</span>", unsafe_allow_html=True)

        st.write(f"**Regions Detected:** {regions}")
        st.write("Higher damage scores indicate more detected impact areas.")

        ok, buffer = cv2.imencode(".png", processed_bgr)
        if ok:
            st.download_button(
                "Download Processed Image",
                data=BytesIO(buffer),
                file_name="cardoctor_ai_processed.png",
                mime="image/png"
            )

else:
    st.info("Upload an image to get started.")
