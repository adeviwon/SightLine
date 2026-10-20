"""
test_graph_width_contract.py — the fixed width is a contract, not a local detail.

The recogniser's LSTM cannot export a dynamic width, so the graph is fixed-width
and the app pads/truncates crops to it. Three places have to agree, and they were
allowed to drift:

  realdata.MAX_W            the width training built crops at (crop_to_array
                            clamps natural width to this) = 320
  export_crnn(width=...)    the width baked into models/onnx/crnn.onnx
  recognizer.js EXPORT_W    the width the app builds its tensor at

THE SERVED WIDTH MUST EQUAL THE TRAINING CLAMP, and this test used to only assert
`graph width <= MAX_W`, which is how 256 shipped for so long: it satisfied the
inequality while sitting BELOW the width the training crops actually are. The
corpus says what that costs — in crops_train.npz only 40 of 3,270 test crops (1.2%)
are <= 256px and the median and max are both 320, so a 256px graph truncated the
tail off ~99% of lines. Measured on 200 held-out crops: exact line match 44.7% at
natural width against 0.0% cut to 256, the read changed on 200 of 200, e.g.
'PATIENT JANE DOE DOB 04021979' -> 'PERIENT JANE DOE 2O08010'.

The other invariant is that the app's tensor width and the graph's must match,
because buildInput writes rows at EXPORT_W stride — a mismatch corrupts the tensor
silently instead of failing.
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


def test_the_graph_width_is_the_training_clamp():
    """The served width must EQUAL realdata.MAX_W, not merely sit under it.

    `<= MAX_W` is the assertion that let 256 ship: it is trivially satisfied by any
    narrower graph, and a narrower graph truncates the tail off the lines the
    training distribution is made of (median crop width == the clamp; 98.8% of
    held-out crops exceed 256). Equality is the only width that neither truncates
    nor invites widths training never allowed.
    """
    w = graph_width()
    assert w == R.MAX_W, (
        f"the graph takes {w}px but training built crops up to MAX_W={R.MAX_W}: "
        f"crops between {w} and {R.MAX_W}px lose their tail before the model sees "
        "them. Re-export with `bash run.sh export` (width defaults to the clamp) "
        "and set recognizer.js EXPORT_W to match.")
    assert w >= R.MIN_W, f"graph width {w} is below the training minimum {R.MIN_W}"


def test_the_app_pads_to_the_graph_width():
    assert js_export_w() == graph_width(), (
        "recognizer.js EXPORT_W and the ONNX graph disagree about the input "
        "width: buildInput writes rows at EXPORT_W stride, so a mismatch "
        "corrupts the tensor instead of failing loudly")
