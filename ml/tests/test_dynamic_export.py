"""
Tests for the dynamic-width ONNX export (ml/src/export_dynamic.py).

What is pinned here:

  1. THE FRAME MAP at widths that are NOT multiples of 8, where T(W) != W//8
     (W=47 -> 6, W=63 -> 8). These are the widths a decoder that assumes W//8
     silently mis-trims, and they are exactly the widths the dynamic graph must
     still handle. Checked against the model's own conv-stack timesteps().

  2. THE DECODE LIMIT on synthetic logits of known length: the padded path's
     limit must drop only the padding tail and nothing else.

  3. THE BUCKET CHOOSER (the fallback rung): the chosen bucket is the SMALLEST
     one >= content width and is NEVER smaller. A bucket below content truncates
     real ink, which is worse than the padding it is meant to bound.

  4. THE DYNAMIC EXPORT ITSELF: the ONNX input width is a symbolic axis, and
     Python-ORT parity against PyTorch holds at several widths including the
     non-multiple-of-8 ones. Skipped (with a reason) when the artifact has not
     been produced -- run `bash run.sh py ml/src/export_dynamic.py`.

Run:  bash run.sh py -m pytest ml/tests/test_dynamic_export.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import export_dynamic as ED              # noqa: E402
import fix_padding as FP                 # noqa: E402
import realdata as R                     # noqa: E402
from ocr_model import build as build_model   # noqa: E402

DYNAMIC = ED.DYNAMIC_ONNX
CKPT = ROOT / "models" / "ocr" / "crnn.pt"


# ── the frame map at non-multiples of 8 ──────────────────────────────────────

@pytest.mark.parametrize("width,t,not_w8", [(47, 6, 5), (63, 8, 7)])
def test_timesteps_at_non_multiples_of_eight(width, t, not_w8):
    """
    T(W) is NOT W//8 off the multiples of 8, and the dynamic graph emits exactly
    T(W) frames. A decoder that assumes W//8 trims one frame too many or too few
    on these widths. Cross-check the closed form against the running conv stack.
    """
    assert FP.timesteps(width) == t
    assert FP.timesteps(width) != not_w8, "the W//8 shortcut is wrong here"
    assert build_model(R.NUM_CLASSES).timesteps(width) == t


def test_dynamic_frame_limit_equals_the_model_timesteps():
    """With no padding, the decode limit IS the content's timestep count."""
    for w in (32, 47, 63, 64, 128, 200, 256):
        assert FP.frame_limit(w, None) == FP.timesteps(w)


# ── the decode limit, on synthetic logits with a known tail ──────────────────

def test_decode_limit_drops_only_the_padding_tail():
    from realdata import decode_greedy
    content = [1, 0, 2, 0, 3, 0, 0, 0]        # '012'
    tail = [4, 5, 6, 0, 7, 0, 0, 0]           # would read '3456' if not limited
    seq = np.array(content + tail)
    # A dynamic graph has no tail at all, but the same limit rule is shared:
    assert decode_greedy(seq, limit=FP.timesteps(64)) == "012"
    assert decode_greedy(seq) == "0123456"


# ── the fallback: width buckets ──────────────────────────────────────────────

def test_choose_bucket_is_never_below_the_content():
    for w in range(R.MIN_W, R.MAX_W + 1):
        b = ED.choose_bucket(w)
        assert b >= w, f"bucket {b} below content {w} truncates ink"


def test_choose_bucket_is_the_smallest_such_bucket():
    assert ED.choose_bucket(1) == 64
    assert ED.choose_bucket(64) == 64
    assert ED.choose_bucket(65) == 96
    assert ED.choose_bucket(96) == 96
    assert ED.choose_bucket(128) == 128
    assert ED.choose_bucket(193) == 256
    assert ED.choose_bucket(256) == 256


def test_choose_bucket_clamps_an_oversized_crop_to_the_widest():
    assert ED.choose_bucket(9999) == max(ED.BUCKETS)


# ── the dynamic export itself ────────────────────────────────────────────────

@pytest.mark.skipif(not DYNAMIC.exists(),
                    reason="run `bash run.sh py ml/src/export_dynamic.py` first")
def test_graph_input_width_is_a_symbolic_axis():
    """
    The shipped graph's width is a fixed int; the dynamic one must be a symbol.
    This is the contract the whole fix rests on -- if width is baked, the graph
    is fixed-width again and padding comes back.
    """
    import onnx
    m = onnx.load(str(DYNAMIC))
    dims = [d.dim_param or d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    assert dims[0] == "batch", f"batch axis is not dynamic: {dims}"
    assert dims[3] == "width", f"width axis is fixed at {dims[3]!r}, not dynamic"
    assert dims[2] == 32, f"height must stay a fixed 32, got {dims[2]!r}"


@pytest.mark.skipif(not DYNAMIC.exists(),
                    reason="run `bash run.sh py ml/src/export_dynamic.py` first")
@pytest.mark.skipif(not CKPT.exists(), reason="no trained checkpoint")
def test_ort_parity_with_pytorch_at_every_width():
    """
    The dynamic ONNX must compute what eager PyTorch computes, at every width --
    including 47 and 63, where the frame map is not exactly w//8.
    """
    rows, ok = ED.verify_parity(widths=(32, 47, 63, 64, 128, 200, 256))
    bad = [r for r in rows if not r[6]]
    assert not bad, f"argmax mismatch at {[(r[0], r[4], r[5]) for r in bad]}"
    worst = max(r[3] for r in rows)
    assert worst <= 1e-3, f"max |Δ| {worst:.2e} exceeds tolerance"
    assert ok


@pytest.mark.skipif(not DYNAMIC.exists(), reason="no dynamic artifact")
def test_dynamic_graph_feeds_a_natural_width_and_a_padded_one_alike():
    """
    A width that is a multiple of 8 and one that is not both run, and the
    output frame count equals T(W) -- so the graph is genuinely width-agnostic
    rather than merely accepting one shape.
    """
    import onnxruntime as ort
    s = ort.InferenceSession(str(DYNAMIC), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for w in (47, 64, 200):
        y = s.run(None, {"input": rng.uniform(-1, 1, (1, 1, 32, w)).astype(np.float32)})[0]
        assert y.shape == (1, FP.timesteps(w), R.NUM_CLASSES), (w, y.shape)


# ── the browser fixture must describe the graph the browser actually runs ────

FIXTURE = ROOT / "artifacts" / "dynamic_width_fixture.json"


@pytest.mark.skipif(not (DYNAMIC.exists() and FIXTURE.exists()),
                    reason="run `bash run.sh py ml/src/export_dynamic.py` first")
def test_browser_fixture_matches_the_dynamic_graph_it_claims():
    """
    app/tools/dynamic_width_check.js loads the fixture's crop bytes and requires
    its decodes to equal the fixture's Python reference. If the fixture drifts
    from the artifact, the browser check passes or fails for the wrong reason —
    so re-derive both arms here from the artifact itself.

      py_natural[W]  the dynamic graph at the crop's own width
      py_shipped[W]  the SHIPPED path: right-pad to 256, decode TimestepsFor(W)
                     frames (floor(W/8), which is what recognizer.js computes)

    A fixture written for a stale checkpoint is exactly the failure this
    catches, which is how the artifact on disk was found to be stale.
    """
    import base64
    import json
    import onnxruntime as ort
    from realdata import decode_greedy

    fx = json.loads(FIXTURE.read_text())
    s = ort.InferenceSession(str(DYNAMIC), providers=["CPUExecutionProvider"])
    for W in fx["widths"]:
        k = str(W)
        u8 = np.frombuffer(base64.b64decode(fx["crops"][k]), np.uint8)
        assert u8.size == 32 * W, (k, u8.size)
        x = ((u8.reshape(32, W).astype(np.float32) / 255.0) - 0.5) / 0.5
        nat = s.run(None, {"input": x[None, None]})[0][0].argmax(-1)
        assert nat.shape[0] == FX_TIMESTEPS(W), (W, nat.shape)
        assert decode_greedy(nat) == fx["py_natural"][k], (
            f"width {W}: fixture natural decode {fx['py_natural'][k]!r} is not "
            f"what crnn_dynamic.onnx produces ({decode_greedy(nat)!r}) — the "
            f"fixture was written against a different export")
        pad = np.zeros((1, 1, 32, 256), np.float32)
        pad[0, 0, :, :W] = x
        sh = s.run(None, {"input": pad})[0][0].argmax(-1)
        lim = min(max(1, W // 8), sh.shape[0])
        assert decode_greedy(sh, limit=lim) == fx["py_shipped"][k], (
            f"width {W}: fixture shipped decode {fx['py_shipped'][k]!r} is not "
            f"what the padded graph produces ({decode_greedy(sh, limit=lim)!r})")


def FX_TIMESTEPS(w: int) -> int:
    """The dynamic graph's frame count at a natural width (== the model's)."""
    return build_model(R.NUM_CLASSES).timesteps(w)
