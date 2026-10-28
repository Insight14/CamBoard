"""Unit test verifying Motion Letters (J/Z) ONNX model loading and parity."""

from pathlib import Path
import numpy as np
import onnxruntime as ort
import pytest
import torch
from camboard.models.gru_seq import GRUSequenceClassifier

ONNX_PATH = Path("onnx/motion_letters.onnx")
CKPT_PATH = Path("checkpoints/motion_letters_best.pt")


@pytest.fixture
def dummy_input():
    return np.random.RandomState(42).randn(1, 30, 126).astype(np.float32)


def test_motion_letters_onnx_loads(dummy_input):
    if not ONNX_PATH.exists():
        pytest.skip("Run 'make train-motion_letters' before testing.")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    assert sess is not None
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: dummy_input})
    assert outputs[0].shape == (1, 26), f"Expected (1, 26), got {outputs[0].shape}"


def test_motion_letters_parity(dummy_input):
    if not ONNX_PATH.exists() or not CKPT_PATH.exists():
        pytest.skip("Missing checkpoint or ONNX model.")
    model = GRUSequenceClassifier(input_dim=126, hidden_dim=64, num_layers=2, num_classes=26)
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

