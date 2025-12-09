import streamlit as st
import numpy as np
import cv2
from io import BytesIO

# ----------------------------
# Streamlit App Title & Setup
# ----------------------------
st.set_page_config(page_title="CarDoctor-AI — Damage Detection", layout="centered")
st.title("CarDoctor-AI — Vehicle Damage Detection")
st.write("Upload a car image and I’ll highlight possible damage using simple OpenCV techniques.")

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
    Steps:
    1. Convert to grayscale
    2. Gaussian blur
    3. Canny edge detection
    4. Contour extraction
    5. Highlight large contours as 'damage'
    6. Compute simple damage score
    """

    output = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    image_area = h * w

    min_area = image_area * 0.001  # Contours must be at least 0.1% of image area
    total_damage_area = 0
    damage_regions = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
            total_damage_area += area
            damage_regions += 1

    # Compute damage score
    damage_ratio = total_damage_area / image_area
    raw_score = damage_ratio * 50  # Scale 0–50
    damage_score = int(np.clip(np.round(raw_score), 1, 10)) if damage_regions > 0 else 1

    return output, edges, damage_score, damage_regions

# ----------------------------------------
# File Upload Section
# ----------------------------------------
uploaded_file = st.file_uploader("Upload a car image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Error reading image. Please try another file.")
    else:
        st.subheader("Original Image")
        st.image(bgr_to_rgb(image_bgr), use_container_width=True)

        if st.button("Analyze Image"):
            processed_bgr, edges, score, regions = process_image(image_bgr)

            st.subheader("Detected Damage (Highlighted)")
            st.image(bgr_to_rgb(processed_bgr), use_container_width=True)

            st.subheader("Edge Map")
            st.image(edges, use_container_width=True, clamp=True)

            st.markdown(f"### **Damage Score:** `{score}` / 10")
            st.markdown(f"### **Regions Detected:** `{regions}`")

            # Prepare image for download
            ok, buffer = cv2.imencode(".png", processed_bgr)
            if ok:
                st.download_button(
                    label="Download Processed Image",
                    data=BytesIO(buffer),
                    file_name="cardoctor_ai_processed.png",
                    mime="image/png",
                )
else:
    st.info("Please upload an image to start.")
