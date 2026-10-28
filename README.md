# CamBoard

Interactive, in-browser digital whiteboard with zero-latency hand perception, mid-air drawing, and stroke handwriting recognition.

---

## 🚀 Running the Frontend Web App

To launch and test the interactive whiteboard in your browser:

```bash
# From the project root (/CamBoard):
make serve
# Or:
npm start
# Or directly:
python3 -m http.server 8000 --directory web
```

Then open **[http://localhost:8000](http://localhost:8000)** in Chrome, Edge, or Safari.

---

## 🎨 Frontend Features & Interactive Testing

### 1. 🖐️ Mid-Air Drawing & Live Hand Perception
- Click **"Start Camera"** to activate your webcam.
- **MediaPipe Hands** extracts 21 3D landmarks in real-time.
- **`gesture.onnx`** runs directly in your browser using **ONNX Runtime WebAssembly**.
- Detects core whiteboard gestures:
  - `pinch`: Activates pen-down drawing mid-air at your fingertip!
  - `point`: Hover & point drawing cursor.
  - `palm` / `palm_out`: Hover navigation without drawing.
  - `fist`, `thumbs_up`, `swipe_start`, `none`.

### 2. 📐 Stroke & Shape Recognition ($Q & 1D-CNN)
- Draw any geometric shape (circle, square, triangle, star, arrow, line, diamond, hexagon) using either your mouse or mid-air finger gestures.
- The browser normalizes the stroke to 64 points and classifies it with the `$Q` algorithm and `stroke_shapes.onnx`.

### 3. 📹 In-App Data Recorder
- Switch to the **"In-App Recorder"** tab.
- Set a **Signer ID** and **Class Label** (e.g. `pinch`, `circle`, `A`).
- Press the **Spacebar** or click **"Capture Sample"** to save landmark arrays.
- Click **"Export Collected Dataset"** to download the session JSON ready for `/ml` training.

### 4. ⚡ In-Browser ONNX Latency Benchmarks
- Switch to the **"Model Benchmarks"** tab.
- Click **"Run Full Browser Benchmark"** to execute 50 warmup runs and 200 timed inferences directly on your device's WebAssembly backend, reporting P50 and P95 latencies.

---

## 🔬 ML Training Infrastructure (`/ml`)

To train or verify any of the underlying machine learning models:

```bash
cd ml
pip install -r requirements.txt
make test                 # Run all 16 ONNX parity unit tests
make train-gesture        # Train Gesture MLP
make train-fingerspelling # Train static ASL alphabet
make train-stroke_shapes  # Train 1D-CNN shape recognizer
make train-stroke_chars   # Train BiGRU character recognizer
make train-motion_letters # Train motion letters J & Z
make train-math           # Integrate pretrained CROHME math model
make train-sign_vocab     # Train isolated sign vocabulary Transformer
```

