import streamlit as st
import numpy as np
import cv2
from io import BytesIO

st.set_page_config(page_title="CarDoctor-AI", layout="wide")

# ---------------------------
# CSS STYLING (with !important so Streamlit doesn't override it)
# ---------------------------
st.markdown("""
<style>

.score-badge {
    display: inline-block;
    padding: 12px 26px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: 700;
    color: white !important;
}

.score-green {
    background-color: #28A745 !important;  /* GREEN */
}

.score-yellow {
    background-color: #FFC107 !important;  /* YELLOW */
    color: black !important;
}

.score-red {
    background-color: #DC3545 !important;  /* RED */
}

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

</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">CarDoctor-AI — Vehicle Damage Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a car image to detect scratches, dents, and smashed areas</div>', unsafe_allow_html=True)


# ---------------------------
# CONVERT BGR → RGB
# ---------------------------
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------
# DAMAGE DETECTION FUNCTION
# ---------------------------
def process_image(image_bgr):
    output = image_bgr.copy()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edges
    edges = cv2.Canny(blur, 80, 200)

    # No merging; you want small boxes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    image_area = float(h * w)

    min_area = image_area * 0.0005   # keep small boxes
    max_area = image_area * 0.20     # ignore giant areas

    total_damage_area = 0.0
    detected_regions = 0
    region_sizes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
            total_damage_area += area
            detected_regions += 1
            region_sizes.append(area)

    # ---------------------------
    # UPDATED DAMAGE RATING SYSTEM
    # ---------------------------
    damage_ratio = total_damage_area / image_area if image_area > 0 else 0
    avg_region_size = (total_damage_area / detected_regions) if detected_regions > 0 else 0

    region_size_score = np.clip(avg_region_size / (0.02 * image_area), 0, 1)

    area_component = np.clip(damage_ratio / 0.04, 0, 1)
    region_component = np.clip(detected_regions / 20, 0, 1)
    size_component = np.clip(region_size_score, 0, 1)

    combined_score = (
        0.60 * area_component +
        0.25 * region_component +
        0.15 * size_component
    )

    damage_score = int(np.clip(round(combined_score * 9) + 1, 1, 10))

    return output, damage_score, detected_regions, damage_ratio


# ---------------------------
# GET SEVERITY COLOR CLASS
# ---------------------------
def get_severity_class(score):
    if score <= 3:
        return "score-green", "Minor Damage"
    elif 4 <= score <= 6:
        return "score-yellow", "Moderate Damage"
    else:
        return "score-red", "Severe Damage"


# ---------------------------
# STREAMLIT UI
# ---------------------------
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

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

        severity_class, severity_label = get_severity_class(score)

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(bgr_to_rgb(processed), caption="Detected Damage Regions", use_container_width=True)

        with col2:
            st.markdown(
                f"<br><span class='score-badge {severity_class}'>Damage Score: {score} / 10</span>",
                unsafe_allow_html=True
            )
            st.write(f"**Severity:** {severity_label}")
            st.write(f"**Damage Regions Detected:** {regions}")
            st.write(f"**Approx. Damaged Area:** {ratio * 100:.2f}%")

        ok, buffer = cv2.imencode(".png", processed)
        if ok:
            st.download_button(
                "Download Processed Image",
                data=BytesIO(buffer),
                file_name="cardoctor_ai_damage.png",
                mime="image/png"
            )

else:
    st.info("Upload an image to begin.")
