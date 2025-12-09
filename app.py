import streamlit as st
import numpy as np
import cv2
from io import BytesIO

st.set_page_config(page_title="Car Doctor AI — Damage Detection", layout="centered")

st.title("Car Doctor AI — Damage Detection")
st.write("Upload a car image and I'll highlight possible damage using simple OpenCV techniques.")

uploaded_file = st.file_uploader("Upload a car image (JPG or PNG)", type=["jpg", "jpeg", "png"])

def process_image(image_bgr):
    """
    Process the image using basic OpenCV:
    - grayscale
    - blur
    - Canny edges
    - contours
    - bounding boxes on larger contours (possible damage)
    """
    # Keep a copy for drawing
    output = image_bgr.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blur, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    image_area = float(h * w)

    damage_contours = []
    total_damage_area = 0.0

    # Very simple rule: consider only contours above a minimum area
    min_contour_area = image_area * 0.001  # 0.1% of image area

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_contour_area:
            damage_contours.append(cnt)
            total_damage_area += area
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Draw rectangle in red
            cv2.rectangle(output, (x, y), (x + cw, y + ch), (0, 0, 255), 2)

    # Compute a very rough "damage score" from 1–10 based on fraction of area
    if image_area > 0:
        damage_ratio = total_damage_area / image_area
    else:
        damage_ratio = 0.0

    # Scale ratio to 1–10, but clamp
    # e.g., 0–5% = small, 5–20% = medium, >20% = high
    raw_score = damage_ratio * 50  # just a simple scaling factor
    damage_score = int(np.clip(np.round(raw_score), 1, 10)) if total_damage_area > 0 else 1

    return output, edges, damage_score, len(damage_contours)


def convert_bgr_to_rgb(image_bgr):
    """Streamlit expects RGB, OpenCV uses BGR."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


if uploaded_file is not None:
    # Read image file into OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not read the image. Please try another file.")
    else:
        st.subheader("Original Image")
        st.image(convert_bgr_to_rgb(image_bgr), use_container_width=True)

        if st.button("Analyze Image"):
            processed_bgr, edges, damage_score, num_regions = process_image(image_bgr)

            st.subheader("Processed Image (Possible Damage Highlighted)")
            st.image(convert_bgr_to_rgb(processed_bgr), use_container_width=True)

            st.subheader("Edge Map (for reference)")
            st.image(edges, use_container_width=True, clamp=True)

            st.markdown(f"**Estimated Damage Score:** `{damage_score}` / 10")
            st.markdown(f"**Number of highlighted regions:** `{num_regions}`")

            # Prepare image for download
            success, buffer = cv2.imencode(".png", processed_bgr)
            if success:
                byte_io = BytesIO(buffer.tobytes())
                st.download_button(
                    label="Download Processed Image",
                    data=byte_io,
                    file_name="car_damage_processed.png",
                    mime="image/png"
                )
            else:
                st.warning("Could not prepare image for download.")
else:
    st.info("Please upload an image to begin.")
