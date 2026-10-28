"""Unit test verifying Sign Vocabulary (Temporal Transformer) ONNX model loading and parity."""

from pathlib import Path
import numpy as np
import onnxruntime as ort
import pytest
import torch
from camboard.models.temporal_transformer import SignVocabTransformer

ONNX_PATH = Path("onnx/sign_vocab.onnx")
CKPT_PATH = Path("checkpoints/sign_vocab_best.pt")


@pytest.fixture
def dummy_input():
    # Transformer takes (batch, 64, 132)
    return np.random.RandomState(42).randn(1, 64, 132).astype(np.float32)


def test_sign_vocab_onnx_loads(dummy_input):
    if not ONNX_PATH.exists():
        pytest.skip("Run 'make train-sign_vocab' before testing.")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    assert sess is not None
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: dummy_input})
    # Returns (logits, confidence)
    assert len(outputs) >= 1
    assert outputs[0].shape == (1, 50), f"Expected (1, 50), got {outputs[0].shape}"


def test_sign_vocab_parity(dummy_input):
    if not ONNX_PATH.exists() or not CKPT_PATH.exists():
        pytest.skip("Missing checkpoint or ONNX model.")
    model = SignVocabTransformer(input_dim=132, d_model=128, num_heads=4, num_layers=2, num_classes=50)
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pt_out = model(torch.from_numpy(dummy_input))
        pt_logits = pt_out[0].numpy() if isinstance(pt_out, tuple) else pt_out.numpy()

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {input_name: dummy_input})[0]

    max_diff = float(np.max(np.abs(pt_logits - ort_out)))
    assert max_diff < 1e-4, f"Parity diff {max_diff} exceeded 1e-4"

