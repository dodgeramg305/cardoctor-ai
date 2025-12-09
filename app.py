import streamlit as st
import numpy as np
import cv2
from io import BytesIO

st.set_page_config(page_title="CarDoctor-AI — Damage Detection", layout="centered")

# ------------------------------
# SIMPLE UTILS
# ------------------------------
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def calculate_severity(area_ratio):
    """
    area_ratio: percent of image covered by detected damage
    """
    if area_ratio < 0.01:
        return 1, "Minor"
    elif area_ratio < 0.03:
        return 3, "Low"
    elif area_ratio < 0.06:
        return 5, "Moderate"
    elif area_ratio < 0.10:
        return 7, "High"
    else:
        return 9, "Severe"

def draw_boxes_and_detect_damage(img):
    """
    Traditional OpenCV damage detection using:
    - grayscale
    - blur
    - edge detection
    - contour detection
    """
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 60, 160)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    image_area = h * w

    total_damage_area = 0
    damage_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:   # ignore tiny contours
            continue

        total_damage_area += area
        damage_count += 1

        x, y, cw, ch = cv2.boundingRect(cnt)
        cv2.rectangle(img_copy, (x, y), (x+cw, y+ch), (0, 0, 255), 2)

    area_ratio = total_damage_area / image_area if image_area > 0 else 0
    severity_score, severity_label = calculate_severity(area_ratio)

    return img_copy, severity_score, severity_label, damage_count


# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("CarDoctor-AI — Damage Detection (Simple, Stable Version)")
st.write("Upload a car image and detect possible damage using clean OpenCV methods.")

uploaded = st.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read the image.")
    else:
        st.subheader("Original Image")
        st.image(bgr_to_rgb(img_bgr), use_container_width=True)

        if st.button("Analyze Damage"):
            processed, severity_score, severity_label, count = draw_boxes_and_detect_damage(img_bgr)

            st.subheader("Detected Damage")
            st.image(bgr_to_rgb(processed), use_container_width=True)

            st.markdown(f"### Damage Score: **{severity_score}/10**")
            st.markdown(f"### Severity: **{severity_label}**")
            st.markdown(f"### Regions Detected: `{count}`")

            # simple repair estimate
            estimated_cost = severity_score * 50
            st.markdown(f"### Estimated Repair Cost: **${estimated_cost} – ${estimated_cost+250}**")

            # download processed image
            ok, buffer = cv2.imencode(".png", processed)
            if ok:
                st.download_button(
                    "Download Processed Image",
                    buffer.tobytes(),
                    file_name="car_damage_result.png",
                    mime="image/png"
                )
