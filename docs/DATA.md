# CamBoard Data Architecture & Dataset Licenses

This document details the datasets, licensing constraints, collection protocols, and preprocessing pipelines for all machine learning models in **CamBoard**.

---

## 1. Zero-Leakage Splitting Guarantee
To ensure production generalization and avoid test-set data leakage:
- **Never split by random frame or stroke:** All splits (`train`, `val`, `test`) are strictly partitioned by **Signer ID**, **User/Session ID**, or **Writer ID**.
- **Cross-Signer Validation:** Evaluation metrics are computed strictly **once** on the final held-out signers/writers who were never seen during training or early stopping validation.

---

## 2. Dataset Registry & Licensing

| Dataset | Modality | Primary Use Case | License | Commercial Use | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CamBoard In-App Recorder** | 3D Hand Landmarks & Air Strokes | Gesture, Fingerspelling, Stroke Chars | Proprietary / Apache 2.0 (Internal) | Yes | Collected via in-app webcam recorder across diverse lighting, hand poses, distances. |
| **ASL Alphabet (Kaggle / Grassknoted)** | Static RGB Images -> Landmarks | Static ASL Letters (24 static) | CC BY-SA 4.0 | Yes (with attribution) | Landmarks extracted via MediaPipe Hands; failed detections discarded. |
| **Google QuickDraw** | Vector Stroke Sequences | Stroke Shapes (circle, square, arrow, etc.) | CC BY 4.0 | Yes | Filtered to 10 whiteboard shape classes; 5,000–10,000 samples per class. |
| **Hershey Vector Fonts** | Single-stroke Vector Glyphs | Stroke Characters (A-Z, a-z, 0-9) | Public Domain (US Gov / Dr. A. V. Hershey) | Yes | Serves as canonical baseline stroke templates with heavy synthetic distortion. |
| **IAM On-Line Handwriting** | Online Digital Pen Strokes | Stroke Characters & Words | Non-Commercial Research (FKI-IAM) | No (Research Only / Pretraining) | Used for pretraining; production air-drawn models are fine-tuned on in-app air-drawn data. |
| **UNIPEN (ICROW-03)** | Online Digital Pen Trajectories | Stroke Characters | UNIPEN Association License | Restricted / Research | Evaluated for handwriting baseline dynamics. |
| **WLASL (Word-Level ASL)** | Video -> 2D/3D Landmarks | Isolated Sign Vocabulary (50 signs) | Academic Research License | Non-Commercial | Landmarks extracted from video frames; split by signer ID. |
| **ASL Citizen** | Video -> 3D Landmarks | Isolated Sign Vocabulary Scaling | CC BY 4.0 / Microsoft Research | Yes (with terms) | High-quality diverse signer contributions; used alongside WLASL. |
| **CROHME** | Online Ink & Stroke Math Expressions | Math Recognition (Strokes -> LaTeX) | Research License | Non-Commercial Benchmark | Used to benchmark BTTR/TAMER open-source checkpoints. |

---

## 3. Data Collection Protocol (In-App Recorder)
CamBoard includes an in-app recorder tool designed for zero-friction data capture:
- **Target:** 500+ samples per class.
- **Coverage:**
  - Multiple signers / writers.
  - Variable room illumination (natural day, dim indoor, backlit).
  - Variable camera-to-hand distances (0.4m to 1.5m).
  - Both left and right hands.
- **File Format:**
  - Landmark sessions stored in `data/processed/<task>/<signer_id>/<class_name>_<idx>.npy` containing `(21, 3)` arrays.
  - Stroke sessions stored in `(n_points, 2)` or `(n_points, 3)` with coordinates and pen-lift markers.

---

## 4. Preprocessing & Normalization
- **Hand Landmarks:**
  1. Wrist (landmark 0) translated to origin `(0, 0, 0)`.
  2. Palm scale calculated as mean distance from wrist to MCP knuckles (landmarks 5, 9, 13, 17).
  3. All coordinates divided by palm scale to achieve camera distance invariance.
  4. Yields 63 static features + optional 63 inter-frame velocity features.
- **Strokes:**
  1. Arc-length equidistant resampling to exactly 64 points.
  2. Aspect-ratio preserving bounding box normalization into `[0, 1] x [0, 1]`.
  3. Features represented as `(dx, dy, pen_state)`.

