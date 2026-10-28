# CamBoard ML Infrastructure (`/ml`)

Production machine learning training and ONNX deployment pipeline for **CamBoard** — interactive, camera-driven digital whiteboard with zero-latency browser perception.

---

## 🚀 Quick Start

### 1. Requirements & Setup
```bash
cd ml
pip install -r requirements.txt
make dirs
```

### 2. Available Training Targets
Execute the pipeline in strict training order:
```bash
make train-gesture          # (1) Gesture MLP (pinch, palm, fist, point, swipe, etc.)
make train-fingerspelling   # (2) Static ASL fingerspelling (24 letters)
make train-stroke_shapes    # (4) Shapes ($Q evaluation + 1D-CNN)
make train-stroke_chars     # (5) Stroke characters A-Z, 0-9 (BiGRU + orientation)
make train-motion_letters   # (3) Motion letters J, Z & sequences (2-layer GRU)
make train-math             # (7) Pretrained CROHME math integration (BTTR/TAMER)
make train-sign_vocab       # (6) Isolated ASL vocabulary (Temporal Transformer)
```

### 3. Verification & Parity Testing
Validate that exported ONNX models achieve exact parity (`max_diff < 1e-4`) against PyTorch:
```bash
make test
```

---

## 📐 General Training Rules (CamBoard Standards)
1. **Zero-Leakage Splitting:**
   - Partitioning is strictly performed by `signer_id`, `session_id`, or `writer_id`.
   - Never randomly split frames or sample points.
2. **Early Stopping & Best Checkpoint:**
   - Validated on held-out sessions.
   - Saves only the best validation checkpoint (`checkpoints/<model>_best.pt`), discarding suboptimal trailing epochs.
   - Evaluates test set **strictly once** at conclusion.
3. **Reproducibility:**
   - Unified YAML configuration per model in `configs/`.
   - Seeds fixed across Python `random`, `numpy`, and `torch` (`camboard.utils.seed.fix_seed`).
   - Pinned dependency lockfile: `requirements-lock.txt`.
4. **Export & Quantization:**
   - Standard ONNX export (`opset=17`).
   - Dynamic INT8 quantization only accepted if accuracy drop `< 0.5%`.
5. **Model Deliverables:**
   - Checkpoints saved in `checkpoints/`.
   - Optimized ONNX graphs saved in `onnx/`.
   - GitHub-flavored Model Cards and Confusion Matrices saved in `/docs/models/`.

---

## 📂 Directory Layout
```
/ml
├── Makefile                    # Unified commands (train-*, export-*, test-*)
├── requirements.txt            # Dependency definitions
├── requirements-lock.txt       # Pinned versions
├── configs/                    # One YAML configuration per model
│   ├── gesture.yaml
│   ├── fingerspelling.yaml
│   ├── motion_letters.yaml
│   ├── stroke_shapes.yaml
│   ├── stroke_chars.yaml
│   ├── sign_vocab.yaml
│   └── math_integration.yaml
├── camboard/                   # CamBoard ML Python package
│   ├── data/                   # Data loaders, augmentations, and processors
│   ├── models/                 # PyTorch architectures (MLP, GRU, 1D-CNN, BiGRU, Transformer, Math)
│   ├── training/               # Early-stop trainer, metrics, and ONNX parity export
│   └── utils/                  # Seed, config loading, and model card renderer
├── scripts/                    # Standalone training and integration entrypoints
├── tests/                      # Automated ONNX loading and numerical parity tests
├── benchmarks/                 # Browser WebAssembly latency benchmark script
└── checkpoints/                # Best model checkpoints (.pt)
```

