"""Unit test verifying Math Recognition (TAMER/BTTR encoder & decoder) ONNX loading."""

from pathlib import Path
import numpy as np
import onnxruntime as ort
import pytest
from camboard.data.stroke_utils import render_strokes_to_canvas

ENCODER_ONNX = Path("onnx/math_encoder.onnx")
DECODER_ONNX = Path("onnx/math_decoder.onnx")
FULL_MATH_ONNX = Path("onnx/math.onnx")


def test_math_stroke_rendering():
    """Verify stroke rendering generates standard (1, 512, 512) tensor canvas."""
    sample_strokes = [
        np.array([[50, 100], [120, 100]], dtype=np.float32),  # minus sign
        np.array([[85, 70], [85, 130]], dtype=np.float32),   # plus cross
    ]
    canvas = render_strokes_to_canvas(sample_strokes, width=512, height=512)
    assert canvas.shape == (1, 512, 512)
    assert canvas.max() <= 1.0 and canvas.min() >= 0.0


def test_math_encoder_onnx_loads():
    if not ENCODER_ONNX.exists():
        pytest.skip("Run 'make train-math' to export math encoder.")
    sess = ort.InferenceSession(str(ENCODER_ONNX), providers=["CPUExecutionProvider"])
    assert sess is not None
    dummy = np.random.randn(1, 1, 512, 512).astype(np.float32)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: dummy})
    assert outputs[0].shape == (1, 64, 256)


def test_math_decoder_onnx_loads():
    if not DECODER_ONNX.exists():
        pytest.skip("Run 'make train-math' to export math decoder.")
    sess = ort.InferenceSession(str(DECODER_ONNX), providers=["CPUExecutionProvider"])
    assert sess is not None
    dummy_tgt = np.zeros((1, 10), dtype=np.int64)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: dummy_tgt})
    assert outputs[0].shape == (1, 10, 110)
