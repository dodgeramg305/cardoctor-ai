# Car Doctor AI — Vehicle Damage Detection (Simple Version)

This is a simple final project for a Computer Vision class.

It uses **Streamlit + OpenCV + NumPy** to:
- Upload a car image
- Convert to grayscale
- Apply Gaussian blur
- Run Canny edge detection
- Find contours
- Draw bounding boxes around larger contours (possible damage)
- Compute a very rough "Damage Score" from 1–10 based on contour area

## How to Run (Bolt.new or local)

1. Install the dependencies:

```bash
pip install -r requirements.txt
