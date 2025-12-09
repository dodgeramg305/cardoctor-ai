import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from datetime import datetime

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="CarDoctor-AI", layout="wide")

# =========================
# GLOBAL STYLES
# =========================
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
.footer {
    text-align: center;
    font-size: 12px;
    color: #999;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR BRANDING
# =========================
with st.sidebar:
    st.markdown("### 🚗 CarDoctor-AI")
    st.write("AI-powered visual pre-check for vehicle damage.")
    st.markdown("---")
    st.markdown("**Use Cases**")
    st.write("- Body shops\n- Used car dealers\n- Private buyers\n- Insurance pre-checks")
    st.markdown("---")
    st.markdown("**How it works**")
    st.write(
        "1. Upload a photo of the vehicle front, rear, or side.\n"
        "2. CarDoctor-AI analyzes edges and contours.\n"
        "3. It highlights potential damage and estimates severity."
    )
    st.markdown("---")
    st.write("v1.0 • © 2025 CarDoctor-AI")

# =========================
# HEADER
# =========================
st.markdown('<div class="title">CarDoctor-AI — Vehicle Damage Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a car image to detect scratches, dents, and smashed areas</div>', unsafe_allow_html=True)


# =========================
# HELPERS
# =========================
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_severity_class(score: int):
    """Return CSS class + human label based on damage score."""
    if score <= 3:
        return "score-green", "Minor Damage"
    elif 4 <= score <= 6:
        return "score-yellow", "Moderate Damage"
    else:
        return "score-red", "Severe Damage"


# =========================
# CORE DAMAGE DETECTION
# =========================
def process_image(image_bgr):
    output = image_bgr.copy()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edges
    edges = cv2.Canny(blur, 80, 200)

    # Contours (no grouping so you still get multiple boxes)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    image_area = float(h * w)

    # Keep small but meaningful boxes, ignore tiny noise and giant spans
    min_area = image_area * 0.0005   # 0.05% of image
    max_area = image_area * 0.20     # ignore very large sheets / shadows

    total_damage_area = 0.0
    detected_regions = 0
    region_sizes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Blue boxes look clean and professional
            cv2.rectangle(output, (x, y), (x + cw, y + ch), (255, 0, 0), 2)
            total_damage_area += area
            detected_regions += 1
            region_sizes.append(area)

    damage_ratio = total_damage_area / image_area if image_area > 0 else 0.0
    avg_region_size = (total_damage_area / detected_regions) if detected_regions > 0 else 0.0

    # --- Scoring model (area + count + average size) ---
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


# =========================
# PDF REPORT GENERATOR
# =========================
def generate_pdf_report(
    original_rgb: np.ndarray,
    processed_rgb: np.ndarray,
    score: int,
    severity_label: str,
    regions: int,
    ratio: float,
):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Title & header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, height - 72, "CarDoctor-AI Damage Report")

    c.setFont("Helvetica", 11)
    c.drawString(72, height - 95, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(72, height - 110, "Powered by CarDoctor-AI — Visual Damage Detection")

    # Metrics block
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 140, "Summary")

    c.setFont("Helvetica", 11)
    c.drawString(90, height - 160, f"Damage Score: {score} / 10")
    c.drawString(90, height - 175, f"Severity: {severity_label}")
    c.drawString(90, height - 190, f"Detected Regions: {regions}")
    c.drawString(90, height - 205, f"Approx. Damaged Area: {ratio * 100:.2f}% of image")

    # Images (scaled down)
    try:
        orig_pil = Image.fromarray(original_rgb)
        proc_pil = Image.fromarray(processed_rgb)

        orig_reader = ImageReader(orig_pil)
        proc_reader = ImageReader(proc_pil)

        img_w, img_h = 220, 150

        c.drawString(72, height - 235, "Original:")
        c.drawImage(orig_reader, 72, height - 235 - img_h, width=img_w, height=img_h)

        c.drawString(310, height - 235, "Detected Damage:")
        c.drawImage(proc_reader, 310, height - 235 - img_h, width=img_w, height=img_h)
    except Exception:
        # If image embed fails, still return PDF with text
        pass

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(72, 40, "This is an automated visual estimation and does not replace a professional mechanical inspection.")
    c.drawString(72, 28, "© 2025 CarDoctor-AI")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf


# =========================
# MAIN APP UI
# =========================
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not read the image. Please try another file.")
    else:
        original_rgb = bgr_to_rgb(image_bgr)

        st.divider()
        st.subheader("1. Uploaded Image")
        st.image(original_rgb, caption="Original View", use_container_width=True)

        if st.button("Analyze Damage"):
            processed_bgr, score, regions, ratio = process_image(image_bgr)
            processed_rgb = bgr_to_rgb(processed_bgr)

            st.divider()
            st.subheader("2. Damage Detection Results")

            severity_class, severity_label = get_severity_class(score)

            col1, col2 = st.columns([2, 1])

            with col1:
                # Before/After slider using alpha blend
                st.write("**Before / After Viewer**")
                alpha = st.slider(
                    "Slide to blend between original (left) and detected damage (right):",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.05,
                )
                blended = cv2.addWeighted(original_rgb, 1 - alpha, processed_rgb, alpha, 0)
                st.image(blended, caption="Drag slider above to compare", use_container_width=True)

                st.write("**Detected Damage Overlay**")
                st.image(processed_rgb, caption="Detected Damage Regions", use_container_width=True)

            with col2:
                st.markdown(
                    f"<br><span class='score-badge {severity_class}'>Damage Score: {score} / 10</span>",
                    unsafe_allow_html=True,
                )
                st.write(f"**Severity:** {severity_label}")
                st.write(f"**Regions Detected:** {regions}")
                st.write(f"**Approx. Damaged Area:** {ratio * 100:.2f}% of image")

                st.write("**How to interpret this:**")
                st.write(
                    "- 1–3: Cosmetic or light impact.\n"
                    "- 4–6: Clear visible damage across multiple areas.\n"
                    "- 7–10: Major impact or structural damage likely."
                )

                # PDF report button
                pdf_buffer = generate_pdf_report(
                    original_rgb, processed_rgb, score, severity_label, regions, ratio
                )
                st.download_button(
                    "Download PDF Damage Report",
                    data=pdf_buffer,
                    file_name="CarDoctorAI_Damage_Report.pdf",
                    mime="application/pdf",
                )

            st.markdown('<div class="footer">© 2025 CarDoctor-AI • Visual Damage Pre-Check</div>', unsafe_allow_html=True)

else:
    st.info("Upload a clear photo of the vehicle (front, rear, or side) to get started.")
