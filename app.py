import streamlit as st
import numpy as np
import cv2
from io import BytesIO

st.set_page_config(page_title="CarDoctor-AI", layout="wide")

# ---------------------------
# Page Header Styling
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
    padding: 10px 20px;
    background: #2E86C1;
    color: white;
    border-radius: 12px;
    font-size: 22px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">CarDoctor-AI — Vehicle Damage Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a car image and detect major damage regions using OpenCV</div>', unsafe_allow_html=True)


# ---------------------------
# Convert BGR → RGB
# ---------------------------
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------
# Main Damage Processing
# ---------------------------
def process_image(image_bgr):
    output = image_bgr.copy()

    # Grayscale + blur
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Step 1: Canny edges
    edges = cv2.Canny(blur, 60, 160)

    # Step 2: Dilate edges to MERGE small contours
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Step 3: Find contours on dilated mask
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    image_area = float(h * w)

    # Require each damage region to be at least 1% of image
    min_area = image_area * 0.01

    total_damage_area = 0
    damage_regions = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + cw, y + ch), (255, 0, 0), 4)
            total_damage_area += area
            damage_regions += 1

    # ---------------------------
    # Improved Damage Score
    # ---------------------------
    damage_ratio = total_damage_area / image_area if image_area > 0 else 0

    # Area-based score (major damage becomes 7–10)
    area_score = np.clip(damage_ratio / 0.08, 0, 1)  # 8% area = max severity

    # Region count score (up to 10 regions)
    region_score = np.clip(damage_regions / 6, 0, 1)

    # Weighted combination: area more important than count
    combined = (0.75 * area_score) + (0.25 * region_score)

    damage_score = int(np.clip(round(combined * 9) + 1, 1, 10))

    return output, damage_score, damage_regions, damage_ratio


# ---------------------------
# Streamlit UI
# ---------------------------
uploaded_file = st.file_uploader("Upload a car image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.divider()
    st.subheader("1. Uploaded Image")

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(bgr_to_rgb(image_bgr), caption="Original Image", use_container_width=True)

    # Analyze button
    if st.button("Analyze Damage"):
        processed_bgr, score, regions, ratio = process_image(image_bgr)

        st.divider()
        st.subheader("2. Detected Damage")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(bgr_to_rgb(processed_bgr), caption="Damage Highlighted", use_container_width=True)

        with col2:
            st.markdown(f"<br><span class='score-badge'>Damage Score: {score} / 10</span>", unsafe_allow_html=True)
            st.write(f"**Damage Regions Detected:** {regions}")
            st.write(f"**Approx. Damage Coverage:** {ratio * 100:.1f}% of image")
            st.write("Higher scores indicate major damaged areas (large dents, smashed panels, etc.).")

        # Download button
        ok, buffer = cv2.imencode(".png", processed_bgr)
        if ok:
            st.download_button(
                "Download Highlighted Image",
                data=BytesIO(buffer),
                file_name="cardoctor_ai_damage.png",
                mime="image/png"
            )

else:
    st.info("Upload an image to get started.")
