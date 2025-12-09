import streamlit as st
import numpy as np
import cv2
from io import BytesIO

st.set_page_config(page_title="CarDoctor-AI", layout="wide")

# ---------------------------
# Custom Page Header Styling
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
    """
    Returns:
      processed_bgr  - image with damage boxes drawn
      damage_score   - 1–10 score based on area + regions
      damage_regions - number of detected regions
      damage_ratio   - fraction of image area covered by damage (0–1)
    """
    output = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edges (internal, not displayed)
    edges = cv2.Canny(blur, 80, 180)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    image_area = float(h * w)

    # Require a contour to be at least 0.25% of image area (filters tiny noise)
    min_area = image_area * 0.0025

    total_damage_area = 0.0
    damage_regions = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Thicker red box for clarity
            cv2.rectangle(output, (x, y), (x + cw, y + ch), (255, 0, 0), 3)
            total_damage_area += area
            damage_regions += 1

    # Fraction of the image covered by detected damage
    damage_ratio = total_damage_area / image_area if image_area > 0 else 0.0

    # --------- Damage Score Logic (more realistic) ---------
    # Score based on coverage: 
    # ~2% area -> low score, ~20%+ area -> near max
    area_component = np.clip(damage_ratio / 0.02, 0, 1)  # 0 to 1

    # Score based on count of regions (up to 10 regions)
    region_component = np.clip(damage_regions / 8.0, 0, 1)  # 0 to 1

    # Combine them: area is more important than count
    combined = 0.7 * area_component + 0.3 * region_component

    # Map 0–1 combined to 1–10, clamp
    damage_score = int(np.clip(np.round(combined * 9) + 1, 1, 10)) if total_damage_area > 0 else 1

    return output, damage_score, damage_regions, damage_ratio


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
        processed_bgr, score, regions, ratio = process_image(image_bgr)

        st.divider()
        st.subheader("2. Detection Results")

        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(bgr_to_rgb(processed_bgr), caption="Detected Damage (Highlighted)", use_container_width=True)

        with col2:
            st.markdown("<br><span class='score-badge'>Damage Score: "
                        f"{score} / 10</span>", unsafe_allow_html=True)
            st.write(f"**Regions Detected:** {regions}")
            st.write(f"**Approx. Area Affected:** {ratio * 100:.1f}% of image")

            st.markdown(
                "- Scores closer to **1** indicate minor or limited detected damage.\n"
                "- Scores closer to **10** indicate larger or multiple impacted areas."
            )

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
