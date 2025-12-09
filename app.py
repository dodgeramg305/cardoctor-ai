import streamlit as st
import numpy as np
import cv2
from io import BytesIO

st.set_page_config(page_title="CarDoctor-AI", layout="wide")

# ---------------------------
# Header Styling
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
    padding: 10px 24px;
    background: #1F77B4;
    color: white;
    border-radius: 12px;
    font-size: 22px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">CarDoctor-AI — Vehicle Damage Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a car image to detect scratched, dented, or smashed areas</div>', unsafe_allow_html=True)


# ---------------------------
# Helper: Convert BGR → RGB
# ---------------------------
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------
# MAIN DAMAGE DETECTION LOGIC
# ---------------------------
def process_image(image_bgr):
    output = image_bgr.copy()

    # Convert → grayscale + reduce noise
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 80, 200)

    # Find contours (NO grouping – keeps small boxes you like)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    image_area = float(h * w)

    # You want small boxes → keep contours larger than tiny noise
    min_area = image_area * 0.0005   # 0.05% of image
    max_area = image_area * 0.20     # ignore GIANT shadows/panels

    total_damage_area = 0.0
    detected_regions = 0

    region_sizes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + cw, y + ch), (255, 0, 0), 2)
            total_damage_area += area
            detected_regions += 1
            region_sizes.append(area)

    # ---------------------------
    # IMPROVED DAMAGE SCORING
    # ---------------------------
    damage_ratio = total_damage_area / image_area if image_area > 0 else 0

    avg_region_size = (total_damage_area / detected_regions) if detected_regions > 0 else 0
    region_size_score = np.clip(avg_region_size / (0.02 * image_area), 0, 1)

    # Weighted scoring model
    area_component = np.clip(damage_ratio / 0.04, 0, 1)      # 4% image area = heavy damage
    region_component = np.clip(detected_regions / 20, 0, 1)  # saturates at 20 areas
    size_component  = np.clip(region_size_score, 0, 1)

    combined_score = (
        0.60 * area_component +
        0.25 * region_component +
        0.15 * size_component
    )

    damage_score = int(np.clip(round(combined_score * 9) + 1, 1, 10))

    return output, damage_score, detected_regions, damage_ratio


# ---------------------------
# STREAMLIT USER INTERFACE
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

    if st.button("Analyze Damage"):
        processed, score, regions, ratio = process_image(image_bgr)

        st.divider()
        st.subheader("2. Damage Detection Results")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(bgr_to_rgb(processed), caption="Detected Damage Regions", use_container_width=True)

        with col2:
            st.markdown(f"<br><span class='score-badge'>Damage Score: {score} / 10</span>", unsafe_allow_html=True)
            st.write(f"**Damage Regions Detected:** {regions}")
            st.write(f"**Approx. Damaged Area:** {ratio * 100:.2f}% of image")

            st.write("Score is based on total area, number of regions, and average region size.")

        # Download result
        ok, buffer = cv2.imencode(".png", processed)
        if ok:
            st.download_button(
                "Download Processed Image",
                data=BytesIO(buffer),
                file_name="cardoctor_ai_damage.png",
                mime="image/png"
            )

else:
    st.info("Upload an image to get started.")
