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
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. In the browser:
   - Upload a JPG or PNG of a car
   - Click "Analyze Image"
   - See the original and processed images
   - Download the processed image with highlighted regions

---

## How to Use This Inside Bolt.new (Step-by-step)

In **Bolt.new**:

1. Create a **New Project** (Python / blank is fine).
2. In the **file explorer** on the left:
   - Click **+ File** → name it `app.py` → paste the `app.py` code.
   - Click **+ File** → name it `requirements.txt` → paste those 3 lines.
   - Click **+ File** → name it `README.md` → paste the README.

3. Open a terminal in Bolt and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

4. Access the app in your browser (usually opens automatically at `http://localhost:8501`)

---

## Project Features

- **Simple UI**: Clean Streamlit interface with no fancy styling
- **Classical OpenCV Only**: No ML models, just edge detection and contour analysis
- **Local Processing**: Everything runs locally, no external APIs
- **Damage Detection**: Highlights areas with significant contours as potential damage
- **Score System**: Provides a 1-10 damage severity score based on contour area
- **Download Results**: Save the processed image with highlighted regions

---

## Technical Details

The app uses:
- **Grayscale conversion** to simplify image data
- **Gaussian blur** to reduce noise
- **Canny edge detection** to find edges in the image
- **Contour detection** to identify closed shapes
- **Area thresholding** to filter significant contours (>0.1% of image area)
- **Bounding boxes** drawn in red around detected regions
- **Damage score** calculated from total contour area relative to image size

---

## Requirements

- Python 3.7+
- streamlit
- opencv-python
- numpy

---

## License

MIT License - Free to use for educational purposes
