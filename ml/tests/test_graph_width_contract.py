"""
test_graph_width_contract.py — the fixed width is a contract, not a local detail.

The recogniser's LSTM cannot export a dynamic width, so the graph is fixed-width
and the app pads/truncates crops to it. Three places have to agree, and they were
allowed to drift:

  realdata.MAX_W            the clamp training used (eval_ocr still uses) = 320
  export_crnn(width=...)    the width baked into models/onnx/crnn.onnx = 256
  recognizer.js EXPORT_W    the width the app builds its tensor at = 256

256 is the served width ON PURPOSE (docs/12 publishes [1,1,32,256], and the CTC
argument for a fixed width is in export_crnn's docstring); it sits below the
training clamp so a crop never exceeds what training allowed. What must NEVER
drift is the app's tensor width versus the graph's, because buildInput writes rows
at EXPORT_W stride — a mismatch corrupts the tensor silently instead of failing.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import realdata as R  # noqa: E402

GRAPH = ROOT / "app/models/crnn.onnx"
RECOGNIZER_JS = ROOT / "app/js/recognizer.js"


def graph_width():
    import onnxruntime as ort
    s = ort.InferenceSession(str(GRAPH), providers=["CPUExecutionProvider"])
    shape = s.get_inputs()[0].shape
    return int(shape[3])


def js_export_w():
    src = RECOGNIZER_JS.read_text()
    m = re.search(r"export const EXPORT_W = (\d+)", src)
    assert m, "recognizer.js no longer declares EXPORT_W"
    return int(m.group(1))


def test_the_graph_is_not_wider_than_the_training_clamp():
    """A graph wider than MAX_W would invite crops training never allowed."""
    w = graph_width()
    assert w <= R.MAX_W, (
        f"the graph takes {w}px, above the training clamp MAX_W={R.MAX_W}: crops "
        "that wide were truncated before the model ever saw them")
    assert w >= R.MIN_W, f"graph width {w} is below the training minimum {R.MIN_W}"


def test_the_app_pads_to_the_graph_width():
    assert js_export_w() == graph_width(), (
        "recognizer.js EXPORT_W and the ONNX graph disagree about the input "
        "width: buildInput writes rows at EXPORT_W stride, so a mismatch "
        "corrupts the tensor instead of failing loudly")
