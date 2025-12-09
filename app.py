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
.score-green { background-color: #28A745 !important; }
.score-yellow { background-color: #FFC107 !important; color: black !important; }
.score-red { background-color: #DC3545 !important; }
.title { font-size: 38px; font-weight: 700; text-align: center; margin-bottom: -10px; }
.subtitle { text-align: center; font-size: 18px; color: #666; margin-bottom: 25px; }
.footer { text-align: center; font-size: 12px; color: #999; margin-top: 40px; }
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
        "1. Upload a vehicle photo.\n"
        "2. CarDoctor-AI analyzes edges and contours.\n"
        "3. It highlights potential damage and estimates severity + repair cost."
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
    if score <= 3:
        return "score-green", "Minor Damage"
    elif 4 <= score <= 6:
        return "score-yellow", "Moderate Damage"
    else:
        return "score-red", "Severe Damage"


# =========================
# COST ESTIMATOR
# =========================
def estimate_repair_cost(score, regions, ratio):
    """
    Estimate repair cost using severity score, region count, and damaged area ratio.
    """

    # Base ranges depending on severity
    if score <= 3:
        base_min, base_max = 100, 350
    elif score <= 6:
        base_min, base_max = 350, 1200
    else:
        base_min, base_max = 1200, 3000

    # Each detected region adds cost
    region_factor = 1 + (regions * 0.10)

    # Damaged area ratio also increases cost
    area_factor = 1 + min(ratio * 8, 1.5)  # cap multiplier at 1.5

    est_min = int(base_min * region_factor * area_factor)
    est_max = int(base_max * region_factor * area_factor)

    return est_min, est_max


# =========================
# CORE DAMAGE DETECTION
# =========================
def process_image(image_bgr):
    output = image_bgr.copy()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 80, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    image_area = float(h * w)

    min_area = image_area * 0.0005
    max_area = image_area * 0.20

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


# =========================
# PDF REPORT GENERATOR
# =========================
def generate_pdf_report(original_rgb, processed_rgb, score, severity_label, regions, ratio, cost_min, cost_max):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, height - 72, "CarDoctor-AI Damage Report")

    c.setFont("Helvetica", 11)
    c.drawString(72, height - 95, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 140, "Summary")

    c.setFont("Helvetica", 11)
    c.drawString(90, height - 160, f"Damage Score: {score} / 10")
    c.drawString(90, height - 175, f"Severity Level: {severity_label}")
    c.drawString(90, height - 190, f"Detected Regions: {regions}")
    c.drawString(90, height - 205, f"Approx. Damage Area: {ratio * 100:.2f}%")
    c.drawString(90, height - 220, f"Estimated Repair Cost: ${cost_min:,} - ${cost_max:,}")

    try:
        orig = ImageReader(Image.fromarray(original_rgb))
        proc = ImageReader(Image.fromarray(processed_rgb))

        c.drawString(72, height - 250, "Original Image:")
        c.drawImage(orig, 72, 350, width=220, height=150)

        c.drawString(320, height - 250, "Detected Damage:")
        c.drawImage(proc, 320, 350, width=220, height=150)
    except:
        pass

    c.showPage()
    c.save()
    buf.seek(0)
    return buf


# =========================
# MAIN UI
# =========================
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not process image.")
    else:
        original_rgb = bgr_to_rgb(image_bgr)

        st.subheader("1. Uploaded Image")
        st.image(original_rgb, use_container_width=True)

        if st.button("Analyze Damage"):
            processed_bgr, score, regions, ratio = process_image(image_bgr)
            processed_rgb = bgr_to_rgb(processed_bgr)

            severity_class, severity_label = get_severity_class(score)

            st.subheader("2. Detection Results")

            col1, col2 = st.columns([2, 1])

            with col1:
                alpha = st.slider("Before / After Blend", 0.0, 1.0, 1.0, 0.05)
                blended = cv2.addWeighted(original_rgb, 1 - alpha, processed_rgb, alpha, 0)
                st.image(blended, use_container_width=True)
                st.image(processed_rgb, caption="Detected Damage Regions", use_container_width=True)

            with col2:
                st.markdown(
                    f"<span class='score-badge {severity_class}'>Damage Score: {score}/10</span>",
                    unsafe_allow_html=True
                )
                st.write(f"**Severity:** {severity_label}")
                st.write(f"**Regions Detected:** {regions}")
                st.write(f"**Damage Area:** {ratio * 100:.2f}%")

                # COST ESTIMATION
                cost_min, cost_max = estimate_repair_cost(score, regions, ratio)
                st.markdown("### 💵 Estimated Repair Cost")
                st.write(f"**Range:** ${cost_min:,} – ${cost_max:,}")
                st.write("*Estimate depends on severity, region count, and surface area.*")

                # PDF DOWNLOAD (updated to include cost estimate)
                pdf = generate_pdf_report(
                    original_rgb, processed_rgb, score, severity_label, regions, ratio,
                    cost_min, cost_max
                )
                st.download_button(
                    "Download PDF Damage Report",
                    data=pdf,
                    file_name="CarDoctorAI_Damage_Report.pdf",
                    mime="application/pdf",
                )

        st.markdown('<div class="footer">© 2025 CarDoctor-AI • Visual Damage Pre-Check</div>', unsafe_allow_html=True)

else:
    st.info("Upload a clear photo of the vehicle to begin.")
