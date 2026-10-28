# CamBoard ML Makefile
PYTHON ?= python3
ML_DIR := $(shell pwd)
export PYTHONPATH := $(ML_DIR):$(PYTHONPATH)
CONFIG_DIR := $(ML_DIR)/configs
CKPT_DIR := $(ML_DIR)/checkpoints
ONNX_DIR := $(ML_DIR)/onnx
DOCS_DIR := $(ML_DIR)/../docs/models

MODELS := gesture fingerspelling motion_letters stroke_shapes stroke_chars sign_vocab math

.PHONY: all help dirs install test clean all-cards $(addprefix train-,$(MODELS)) $(addprefix export-,$(MODELS)) $(addprefix test-,$(MODELS))

help:
	@echo "======================================================================"
	@echo "                     CamBoard ML Automation Tool                      "
	@echo "======================================================================"
	@echo "Training commands:"
	@echo "  make train-gesture          Train Gesture MLP (pinch, palm, fist, etc.)"
	@echo "  make train-fingerspelling   Train ASL static fingerspelling (24 letters)"
	@echo "  make train-motion_letters   Train Motion letters J, Z & sequences (GRU)"
	@echo "  make train-stroke_shapes    Train Stroke shapes (1D-CNN + $$Q baseline)"
	@echo "  make train-stroke_chars     Train Stroke characters A-Z, 0-9 (BiGRU)"
	@echo "  make train-sign_vocab       Train Isolated sign vocabulary (Transformer)"
	@echo "  make train-math             Integrate pretrained CROHME math model"
	@echo ""
	@echo "Export commands:"
	@echo "  make export-<model>         Export PyTorch checkpoint to ONNX + INT8"
	@echo ""
	@echo "Verification:"
	@echo "  make test                   Run all ONNX loading and parity tests"
	@echo "  make test-<model>           Run specific ONNX parity test"
	@echo "  make benchmark              Run in-browser latency benchmark harness"
	@echo "======================================================================"

dirs:
	mkdir -p $(CKPT_DIR) $(ONNX_DIR) $(DOCS_DIR) data/raw data/processed

install:
	$(PYTHON) -m pip install -r requirements.txt

# --- Training Targets ---
train-gesture: dirs
	$(PYTHON) scripts/train_gesture.py --config configs/gesture.yaml

train-fingerspelling: dirs
	$(PYTHON) scripts/train_fingerspelling.py --config configs/fingerspelling.yaml

train-motion_letters: dirs
	$(PYTHON) scripts/train_motion_letters.py --config configs/motion_letters.yaml

train-stroke_shapes: dirs
	$(PYTHON) scripts/train_stroke_shapes.py --config configs/stroke_shapes.yaml

train-stroke_chars: dirs
	$(PYTHON) scripts/train_stroke_chars.py --config configs/stroke_chars.yaml

train-sign_vocab: dirs
	$(PYTHON) scripts/train_sign_vocab.py --config configs/sign_vocab.yaml

train-math: dirs
	$(PYTHON) scripts/integrate_math.py --config configs/math_integration.yaml

# --- Export Targets ---
export-gesture: dirs
	$(PYTHON) -c "from camboard.training.onnx_export import export_and_test; export_and_test('$(CKPT_DIR)/gesture_best.pt', '$(ONNX_DIR)/gesture.onnx', 'gesture')"

export-fingerspelling: dirs
	$(PYTHON) -c "from camboard.training.onnx_export import export_and_test; export_and_test('$(CKPT_DIR)/fingerspelling_best.pt', '$(ONNX_DIR)/fingerspelling.onnx', 'fingerspelling')"

export-motion_letters: dirs
	$(PYTHON) -c "from camboard.training.onnx_export import export_and_test; export_and_test('$(CKPT_DIR)/motion_letters_best.pt', '$(ONNX_DIR)/motion_letters.onnx', 'motion_letters')"

export-stroke_shapes: dirs
	$(PYTHON) -c "from camboard.training.onnx_export import export_and_test; export_and_test('$(CKPT_DIR)/stroke_shapes_best.pt', '$(ONNX_DIR)/stroke_shapes.onnx', 'stroke_shapes')"

export-stroke_chars: dirs
	$(PYTHON) -c "from camboard.training.onnx_export import export_and_test; export_and_test('$(CKPT_DIR)/stroke_chars_best.pt', '$(ONNX_DIR)/stroke_chars.onnx', 'stroke_chars')"

export-sign_vocab: dirs
	$(PYTHON) -c "from camboard.training.onnx_export import export_and_test; export_and_test('$(CKPT_DIR)/sign_vocab_best.pt', '$(ONNX_DIR)/sign_vocab.onnx', 'sign_vocab')"

export-math: dirs
	$(PYTHON) scripts/integrate_math.py --config configs/math_integration.yaml --export-only

# --- Testing Targets ---
test:
	pytest tests/ -v

test-gesture:
	pytest tests/test_gesture_onnx.py -v

test-fingerspelling:
	pytest tests/test_fingerspelling_onnx.py -v

test-motion_letters:
	pytest tests/test_motion_letters_onnx.py -v

test-stroke_shapes:
	pytest tests/test_stroke_shapes_onnx.py -v

test-stroke_chars:
	pytest tests/test_stroke_chars_onnx.py -v

test-sign_vocab:
	pytest tests/test_sign_vocab_onnx.py -v

test-math:
	pytest tests/test_math_onnx.py -v

benchmark:
	node benchmarks/browser_latency.js

all-cards:
	@echo "All model cards are generated during training and stored in docs/models/."

clean:
	rm -rf __pycache__ camboard/**/__pycache__ .pytest_cache
