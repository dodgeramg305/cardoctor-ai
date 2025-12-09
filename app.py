import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from datetime import datetime

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

from ultralytics import YOLO

st.set_page_config(page_title="CarDoctor-AI", layout="wide")

# =========================
# CSS STYLING
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
    background-color: #28A745 !important;
}
.score-yellow {
    background-color: #FFC107 !important;
    color: black !important;
}
.score-red {
    background-color: #DC3545 !important;
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
        "1. Upload a photo of the vehicle.\n"
        "2. CarDoctor-AI detects damaged parts.\n"
        "3. It estimates severity and approximate repair cost."
    )
    st.markdown("---")
    st.write("v2.0 • © 2025 CarDoctor-AI")

# =========================
# HEADER
# =========================
st.markdown('<div class="title">CarDoctor-AI — Vehicle Damage Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect damaged parts, estimate severity, and approximate repair costs</div>', unsafe_allow_html=True)


# =========================
# HELPERS
# =========================
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_severity_class(score: int):
    """Return CSS class + label based on overall damage score."""
    if score <= 3:
        return "score-green", "Minor Damage"
    elif 4 <= score <= 6:
        return "score-yellow", "Moderate Damage"
    else:
        return "score-red", "Severe Damage"


# ---- COST & LABEL MAPPINGS ----
# Adapt these to match your YOLO class names AND your pricing assumptions.
CLASS_INFO = {
    "front_bumper_damage": {
        "part": "Front bumper",
        "damage_type": "Impact / crack",
        "cost_range": (700, 1500),
    },
    "rear_bumper_damage": {
        "part": "Rear bumper",
        "damage_type": "Impact / crack",
        "cost_range": (500, 1200),
    },
    "door_dent": {
        "part": "Door",
        "damage_type": "Dent / panel deformation",
        "cost_range": (350, 900),
    },
    "headlight_broken": {
        "part": "Headlight",
        "damage_type": "Broken / missing",
        "cost_range": (300, 700),
    },
    "windshield_crack": {
        "part": "Windshield",
        "damage_type": "Crack / chip / shatter",
        "cost_range": (200, 500),
    },
    "scratch_panel": {
        "part": "Body panel",
        "damage_type": "Paint scratch",
        "cost_range": (120, 400),
    },
    "hood_damage": {
        "part": "Hood",
        "damage_type": "Dent / bend",
        "cost_range": (400, 1400),
    },
    "mirror_broken": {
        "part": "Side mirror",
        "damage_type": "Broken / hanging",
        "cost_range": (150, 350),
    },
}

DEFAULT_CLASS_INFO = {
    "part": "Vehicle body",
    "damage_type": "General damage",
    "cost_range": (200, 800),
}


# =========================
# LOAD YOLO MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    # TODO: Adjust this path to your actual trained model file
    # For example: "models/cardoctor_damage.pt"
    model_path = "models/cardoctor_damage.pt"
    return YOLO(model_path)


model = load_model()


# =========================
# YOLO DAMAGE DETECTION
# =========================
def detect_damage_yolo(image_rgb: np.ndarray):
    """
    Runs the YOLO model on an RGB image and returns:
    - annotated image (RGB)
    - list of detection dicts: {label, conf, part, damage_type, cost_min, cost_max, bbox, area_ratio}
    - overall damage score
    """

    h, w, _ = image_rgb.shape
    image_area = float(h * w)

    # Run YOLO
    results = model(image_rgb, verbose=False)[0]

    detections = []
    total_area = 0.0
    confs = []

    # Draw on a copy
    annotated = image_rgb.copy()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cls_name = results.names.get(cls_id, "damage")
        info = CLASS_INFO.get(cls_name, DEFAULT_CLASS_INFO)

        part = info["part"]
        damage_type = info["damage_type"]
        cmin, cmax = info["cost_range"]

        # Box area & area ratio
        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)
        area = float(box_w * box_h)
        area_ratio = area / image_area if image_area > 0 else 0.0

        total_area += area
        confs.append(conf)

        label_text = f"{part} ({damage_type}) {conf*100:.0f}%"

        # Draw box
        cv2.rectangle(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            2,
        )
        # Label background
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            annotated,
            (int(x1), int(y1) - th - 4),
            (int(x1) + tw + 4, int(y1)),
            (255, 0, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label_text,
            (int(x1) + 2, int(y1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        detections.append(
            {
                "cls_name": cls_name,
                "confidence": conf,
                "part": part,
                "damage_type": damage_type,
                "cost_min": cmin,
                "cost_max": cmax,
                "area_ratio": area_ratio,
            }
        )

    # Compute overall metrics
    total_area_ratio = total_area / image_area if image_area > 0 else 0.0
    avg_conf = float(np.mean(confs)) if confs else 0.0
    num_detections = len(detections)

    # Overall severity score (1–10)
    # based on area coverage, detection count, and avg confidence
    area_component = np.clip(total_area_ratio / 0.06, 0, 1)      # 6% area = high damage
    count_component = np.clip(num_detections / 6, 0, 1)          # 6+ detections saturate
    conf_component = np.clip((avg_conf - 0.5) / 0.4, 0, 1)       # confidences around 0.9 -> high

    combined = 0.5 * area_component + 0.3 * count_component + 0.2 * conf_component
    damage_score = int(np.clip(round(combined * 9) + 1, 1, 10))

    return annotated, detections, damage_score, total_area_ratio, avg_conf


# =========================
# PDF REPORT GENERATION
# =========================
def generate_pdf_report(
    original_rgb: np.ndarray,
    annotated_rgb: np.ndarray,
    detections: list,
    score: int,
    severity_label: str,
    total_area_ratio: float,
    avg_conf: float,
    total_min_cost: float,
    total_max_cost: float,
):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, height - 72, "CarDoctor-AI Damage Report")

    c.setFont("Helvetica", 11)
    c.drawString(72, height - 95, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(72, height - 110, "Powered by CarDoctor-AI — Vehicle Damage Detection & Estimation")

    # Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 140, "Summary")

    c.setFont("Helvetica", 11)
    c.drawString(90, height - 160, f"Overall Damage Score: {score} / 10 ({severity_label})")
    c.drawString(90, height - 175, f"Estimated Damaged Area: {total_area_ratio * 100:.2f}% of visible vehicle")
    c.drawString(90, height - 190, f"Average Detection Confidence: {avg_conf * 100:.1f}%")
    c.drawString(90, height - 205, f"Estimated Repair Cost Range: ${total_min_cost:,.0f} – ${total_max_cost:,.0f} USD")

    # Per-damage breakdown
    y = height - 230
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "Detected Damage")
    y -= 18
    c.setFont("Helvetica", 10)
    for det in detections:
        line = (
            f"- {det['part']}: {det['damage_type']} "
            f"(Conf: {det['confidence']*100:.0f}%, "
            f"Est: ${det['cost_min']}-{det['cost_max']})"
        )
        c.drawString(90, y, line)
        y -= 14
        if y < 90:
            c.showPage()
            y = height - 72

    # Images
    try:
        orig_pil = Image.fromarray(original_rgb)
        ann_pil = Image.fromarray(annotated_rgb)

        orig_reader = ImageReader(orig_pil)
        ann_reader = ImageReader(ann_pil)

        img_w, img_h = 220, 150
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, height - 72, "Visuals")

        c.drawString(72, height - 100, "Original Image:")
        c.drawImage(orig_reader, 72, height - 100 - img_h, width=img_w, height=img_h)

        c.drawString(320, height - 100, "Detected Damage:")
        c.drawImage(ann_reader, 320, height - 100 - img_h, width=img_w, height=img_h)
    except Exception:
        pass

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(72, 40, "This automated estimate is for informational purposes only and does not replace a professional inspection.")
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
            annotated_rgb, detections, score, total_area_ratio, avg_conf = detect_damage_yolo(original_rgb)

            st.divider()
            st.subheader("2. Detected Damage & Estimates")

            severity_class, severity_label = get_severity_class(score)

            # Cost summary
            total_min = sum(d["cost_min"] for d in detections)
            total_max = sum(d["cost_max"] for d in detections)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(annotated_rgb, caption="Detected Damage (YOLO)", use_container_width=True)

            with col2:
                st.markdown(
                    f"<br><span class='score-badge {severity_class}'>Damage Score: {score} / 10</span>",
                    unsafe_allow_html=True,
                )
                st.write(f"**Severity:** {severity_label}")
                st.write(f"**Damaged Area (approx):** {total_area_ratio * 100:.2f}% of image")
                st.write(f"**Average AI Confidence:** {avg_conf * 100:.1f}%")

                st.markdown("### Estimated Repair Cost")
                if detections:
                    st.write(f"**Estimated Total Range:** ${total_min:,.0f} – ${total_max:,.0f} USD")
                else:
                    st.write("No clear damage detected above the current confidence threshold.")

            st.markdown("### Detailed Findings")
            if detections:
                for det in detections:
                    st.write(
                        f"- **{det['part']}** — {det['damage_type']}  "
                        f"(Conf: {det['confidence']*100:.0f}%, "
                        f"Est: ${det['cost_min']}-{det['cost_max']})"
                    )
            else:
                st.write("No significant damage regions detected.")

            # PDF download
            pdf_buf = generate_pdf_report(
                original_rgb,
                annotated_rgb,
                detections,
                score,
                severity_label,
                total_area_ratio,
                avg_conf,
                total_min,
                total_max,
            )
            st.download_button(
                "Download PDF Damage Report",
                data=pdf_buf,
                file_name="CarDoctorAI_Damage_Report.pdf",
                mime="application/pdf",
            )

            st.markdown('<div class="footer">© 2025 CarDoctor-AI • Visual Damage Pre-Check & Estimate</div>', unsafe_allow_html=True)

else:
    st.info("Upload a clear photo of the vehicle (front, rear, or side) to get started.")
