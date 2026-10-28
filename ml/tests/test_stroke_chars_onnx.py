"""Unit test verifying Stroke Character Recognizer (BiGRU) ONNX model loading and parity."""

from pathlib import Path
import numpy as np
import onnxruntime as ort
import pytest
import torch
from camboard.models.bigru import BiGRUClassifier

ONNX_PATH = Path("onnx/stroke_chars.onnx")
CKPT_PATH = Path("checkpoints/stroke_chars_best.pt")


@pytest.fixture
def dummy_input():
    # BiGRU takes (batch, 64, 3)
    return np.random.RandomState(42).randn(1, 64, 3).astype(np.float32)


def test_stroke_chars_onnx_loads(dummy_input):
    if not ONNX_PATH.exists():
        pytest.skip("Run 'make train-stroke_chars' before testing.")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    assert sess is not None
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: dummy_input})
    assert outputs[0].shape == (1, 62), f"Expected (1, 62), got {outputs[0].shape}"


def test_stroke_chars_parity(dummy_input):
    if not ONNX_PATH.exists() or not CKPT_PATH.exists():
        pytest.skip("Missing checkpoint or ONNX model.")
    model = BiGRUClassifier(input_dim=3, hidden_dim=128, num_layers=2, num_classes=62, orientation_head=True)
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pt_out = model(torch.from_numpy(dummy_input))
        if isinstance(pt_out, tuple):
            pt_out = pt_out[0]
        pt_np = pt_out.numpy()

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {input_name: dummy_input})[0]

    max_diff = float(np.max(np.abs(pt_np - ort_out)))
    assert max_diff < 1e-4, f"Parity diff {max_diff} exceeded 1e-4"

