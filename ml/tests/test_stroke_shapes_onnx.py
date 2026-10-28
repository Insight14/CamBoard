"""Unit test verifying Stroke Shape Recognizer (1D-CNN) ONNX model loading and parity."""

from pathlib import Path
import numpy as np
import onnxruntime as ort
import pytest
import torch
from camboard.models.cnn1d import CNN1DClassifier

ONNX_PATH = Path("onnx/stroke_shapes.onnx")
CKPT_PATH = Path("checkpoints/stroke_shapes_best.pt")


@pytest.fixture
def dummy_input():
    # 1D-CNN takes channels-first (batch, 3, 64)
    return np.random.RandomState(42).randn(1, 3, 64).astype(np.float32)


def test_stroke_shapes_onnx_loads(dummy_input):
    if not ONNX_PATH.exists():
        pytest.skip("Run 'make train-stroke_shapes' before testing.")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    assert sess is not None
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: dummy_input})
    assert outputs[0].shape == (1, 10), f"Expected (1, 10), got {outputs[0].shape}"


def test_stroke_shapes_parity(dummy_input):
    if not ONNX_PATH.exists() or not CKPT_PATH.exists():
        pytest.skip("Missing checkpoint or ONNX model.")
    model = CNN1DClassifier(input_dim=3, num_classes=10, channels=[64, 128, 256])
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pt_out = model(torch.from_numpy(dummy_input)).numpy()

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {input_name: dummy_input})[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    assert max_diff < 1e-4, f"Parity diff {max_diff} exceeded 1e-4"

