"""
test_graph_width_contract.py — the app must not disagree with the graph about width.

THE CONTRACT, AND WHY IT CHANGED
--------------------------------
The graph's input width used to be a single number that three places had to
agree on, and they were allowed to drift:

  realdata.MAX_W            the width training built crops at (crop_to_array
                            clamps natural width to this) = 320
  export_crnn(width=...)    the width baked into app/models/crnn.onnx
  recognizer.js EXPORT_W    the width the app built its tensor at

Equality was the right contract while the graph was fixed, and it is the WRONG
one now: app/js/recognizer.js is width-adaptive — it reads the width axis off the
session it just loaded (engineWidth()) and serves what the graph declares. So the
assertions below are the invariants that survive the dynamic-width swap:

  1. THE GRAPH'S WIDTH AXIS IS EITHER THE TRAINING CLAMP OR SYMBOLIC. A fixed
     graph is correct only at R.MAX_W, never below it: 98.8% of held-out crops
     exceed 256px, so the old 256px graph truncated the tail off nearly every
     line before the model saw it ('PATIENT JANE DOE DOB 04021979' came back as
     'PERIENT JANE DOE 2O08010' on 200 of 200 held-out crops). A symbolic axis
     needs no clamp at all, because nothing is ever truncated and nothing is
     padded — the CRNN's backward LSTM pass starts at the RIGHT EDGE, so padding
     to a fixed width moves the hidden states across the VALID frames (measured
     max |delta| 9.10 at content 64, 15.28 at 200).
  2. THE CLAMP IS STILL THE CROP CAP. recognizer.js's TRAINING_CLAMP must equal
     R.MAX_W, because app/js/ocr_engine.js passes it as cropForRecognizer's
     maxW — a crop wider than the clamp is text the model was never trained on.
  3. THE APP CANNOT HOLD A WIDTH OF ITS OWN. No tensor dim and no row stride in
     recognizer.js may be a constant: the tensor is built at the session's width
     (fixed graph) or at contentWidth (symbolic graph), and the stride IS that
     same value. A hardcoded shape is silent corruption rather than an error —
     buildInput writes rows at the stride, so a stride that disagrees with the
     tensor interleaves the crop's rows into each other.

Each assertion is a regex or a graph read that can genuinely fail; the failure
mode was reproduced by pinning the tensor back to EXPORT_W (see the report).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import realdata as R  # noqa: E402

GRAPH = ROOT / "app/models/crnn.onnx"
RECOGNIZER_JS = ROOT / "app/js/recognizer.js"


def graph_width_axis():
    """The graph's input width axis: an int (fixed) or a str/None (symbolic)."""
    import onnxruntime as ort
    s = ort.InferenceSession(str(GRAPH), providers=["CPUExecutionProvider"])
    return s.get_inputs()[0].shape[3]


def js_const(name):
    src = RECOGNIZER_JS.read_text()
    m = re.search(r"export const %s = (\d+)" % re.escape(name), src)
    return int(m.group(1)) if m else None


def test_the_graph_width_axis_is_the_clamp_or_symbolic():
    """Fixed is only allowed AT the clamp; symbolic is allowed outright.

    `<= MAX_W` is the assertion that let 256 ship: it is trivially satisfied by
    any narrower graph, and a narrower graph cuts the tail off the lines the
    training distribution is made of (median crop width == the clamp; 98.8% of
    held-out crops exceed 256). So a fixed width must EQUAL the clamp, and any
    non-integer axis must be a symbolic dimension rather than, say, a bad parse.
    """
    ax = graph_width_axis()
    if isinstance(ax, int):
        assert ax == R.MAX_W, (
            f"the graph takes a FIXED {ax}px but training built crops up to "
            f"MAX_W={R.MAX_W}: crops between {ax} and {R.MAX_W}px lose their tail "
            "before the model sees them. Re-export with `bash run.sh export` "
            "(width defaults to the clamp), or export the dynamic graph "
            "(ml/src/export_dynamic.py) and swap the artifact.")
        assert ax >= R.MIN_W, f"graph width {ax} is below the training minimum {R.MIN_W}"
    else:
        assert ax is None or isinstance(ax, str), (
            f"unreadable width axis {ax!r}: it must be an int (fixed) or a "
            "symbolic dimension name (dynamic)")


def test_the_app_serves_the_width_the_session_declares():
    """recognizer.js must not be able to hold a width the graph disagrees with.

    Three source-level facts, each of which is the difference between "adaptive"
    and "adapts in the comment": the served width is chosen from the
    session-derived `graphWidth`, every tensor dim is that variable, and the row
    stride is the same variable.
    """
    src = RECOGNIZER_JS.read_text()

    assert re.search(r"const w = graphWidth === null \? contentWidth : graphWidth;", src), (
        "recognizer.js no longer derives the served width from the session "
        "(graphWidth): it must build its tensor at the graph's own width when "
        "that width is a number, and at contentWidth when the axis is symbolic")

    dims = re.findall(r"\[1,\s*1,\s*CROP_H,\s*([\w.]+)\]", src)
    assert dims, "recognizer.js no longer builds a [1, 1, CROP_H, <width>] tensor"
    for d in dims:
        assert d == "w", (
            f"a tensor is built with width '{d}'. It must be the runtime `w` — a "
            "constant (EXPORT_W/TRAINING_CLAMP or a literal) is a width the app "
            "can disagree with the graph about, which corrupts the tensor "
            "silently instead of failing")

    assert re.search(r"function buildInput\(gray, contentWidth, w\)", src), (
        "buildInput must take the tensor width as a parameter; it used to close "
        "over EXPORT_W")
    assert re.search(r"const dstRow = y \* w;", src), (
        "buildInput's row stride is not the tensor width: mismatched, it "
        "interleaves the crop's rows into each other")
    assert re.search(r"new Float32Array\(CROP_H \* w\)", src), (
        "buildInput's buffer is not sized at the tensor width")


def test_the_clamp_is_the_crop_cap_not_the_tensor_width():
    """TRAINING_CLAMP == R.MAX_W, and EXPORT_W is only its legacy alias.

    The clamp is still load-bearing — app/js/ocr_engine.js imports EXPORT_W and
    passes it to cropForRecognizer as maxW, and a crop wider than the clamp is
    text training never saw — so it must stay equal to the training clamp. What
    it must NOT do is reappear in the tensor shape, which the test above pins.
    """
    clamp = js_const("TRAINING_CLAMP")
    assert clamp == R.MAX_W, (
        f"recognizer.TRAINING_CLAMP is {clamp} but training clamps crops to "
        f"realdata.MAX_W={R.MAX_W}: ocr_engine passes it as cropForRecognizer's "
        "maxW, so a smaller value truncates served crops")
    src = RECOGNIZER_JS.read_text()
    assert re.search(r"export const EXPORT_W = TRAINING_CLAMP;", src), (
        "EXPORT_W must be an alias of TRAINING_CLAMP (one number, one meaning), "
        "not a second literal that can drift — ocr_engine.js still imports it as "
        "the crop cap")
    assert not re.search(r"export const EXPORT_W = \d+", src), (
        "EXPORT_W is a literal again: two width constants that can drift from "
        "each other and from the training clamp")
