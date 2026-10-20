"""
Tests for the serve-time width fix (ml/src/fix_padding.py).

Three things are pinned here:

  1. THE FRAME MAP. T_out(W) is the contract everything else depends on —
     how many timesteps correspond to content. The closed form in
     fix_padding.timesteps() is checked against SightLineCRNN.timesteps(),
     which RUNS the conv stack, over every width 16..320, plus the exact
     values for the widths the padding invariant is parametrised on.

  2. THE TRUNCATE RULE on synthetic logits of known length, so a decoder that
     truncates at the wrong frame is caught without a checkpoint.

  3. THE INVERSE: the DYNAMIC-WIDTH export must accept any width and compute
     what eager PyTorch computes, and the OLD fixed-width path must still be
     measurably wrong — otherwise the regression test above is vacuous.

Run:  bash run.sh py -m pytest ml/tests/test_fix_padding.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fix_padding as FP                      # noqa: E402
import realdata as R                          # noqa: E402
from ocr_model import build as build_model    # noqa: E402

CKPT = ROOT / "models" / "ocr" / "crnn.pt"
CACHE = ROOT / "artifacts" / "ocr_train.npz"


# ── the frame map ────────────────────────────────────────────────────────────

def test_timesteps_matches_the_model_at_every_width():
    """
    The closed form must equal the value produced by running the conv stack.

    If they disagree the whole fix is off by a frame: the decoder would either
    drop a real character or decode into padding. The model needs no trained
    weights for this — T is a property of the conv geometry.
    """
    m = build_model(R.NUM_CLASSES)
    bad = [(w, FP.timesteps(w), m.timesteps(w))
           for w in range(R.MIN_W, R.MAX_W + 1) if FP.timesteps(w) != m.timesteps(w)]
    assert not bad, f"frame map disagrees with the model at {bad[:8]}"


@pytest.mark.parametrize("width,t", [(32, 4), (64, 8), (128, 16), (200, 25),
                                     (256, 32), (320, 40)])
def test_frame_map_table(width, t):
    assert FP.timesteps(width) == t


def test_multiples_of_eight_are_exactly_width_over_eight():
    """
    T(W) == W/8 for W % 8 == 0 — the widths the render corpus emits.

    Worth pinning separately: the general formula is NOT W//8 (W=47 -> 6,
    W=63 -> 8), so a caller that assumes W//8 silently mis-trims odd widths.
    """
    for w in range(8, R.MAX_W + 1, 8):
        assert FP.timesteps(w) == w // 8, w


# ── the truncate rule, on synthetic logits with known lengths ────────────────

def test_frame_limit_for_a_padded_graph_is_the_content_share():
    assert FP.frame_limit(64, 256) == 8
    assert FP.frame_limit(128, 256) == 16
    assert FP.frame_limit(200, 256) == 25
    # dynamic: no padding, so the limit IS the content's timestep count
    assert FP.frame_limit(200, None) == FP.timesteps(200) == 25


def test_truncation_drops_only_the_tail():
    """
    Synthetic argmax of known length: a decoder limited to T frames must see
    exactly those frames. Built so the padding tail carries a DIFFERENT
    character at every frame — the shape a leaked padding region has — so
    decoding past the limit is visible as extra characters.
    """
    from realdata import decode_greedy
    content = [1, 0, 2, 0, 3, 0, 0, 0]            # '012' in 8 frames
    pad_tail = [4, 5, 6, 0, 7, 0, 0, 0]           # '3456' if not truncated
    seq = np.array(content + pad_tail)
    assert decode_greedy(seq, limit=8) == "012"
    assert decode_greedy(seq) == "0123456"        # the leak, un-truncated


def test_one_timestep_too_many_or_too_few_changes_the_decode():
    """A wrong frame limit must be DETECTABLE — these are the failure modes."""
    from realdata import decode_greedy
    seq = np.array([1, 0, 2, 0, 3, 0, 0, 0] + [4] * 4)
    assert decode_greedy(seq, limit=7) == "012"
    assert decode_greedy(seq, limit=9) == "0123"  # one frame into padding


# ── the selected fix: a dynamic-width graph needs no padding ─────────────────

def test_content_width_equal_to_graph_width_is_a_noop():
    """
    content_w == graph_w must be byte-identical before and after the fix.

    The widest crop is the one case where the old fixed-width path was already
    correct, so the fix must not change it at all — not even by re-normalising.
    """
    rng = np.random.default_rng(0)
    crop = rng.uniform(-1, 1, (R.CROP_H, 256)).astype(np.float32)
    served, limit = FP.pad_for_export(crop, 256, graph_w=None)
    legacy, llimit = FP.pad_for_export(crop, 256, graph_w=256)
    assert np.array_equal(served, crop)
    assert np.array_equal(legacy, crop)
    assert served.shape == legacy.shape == (R.CROP_H, 256)
    assert limit == llimit == 32

    from realdata import decode_greedy
    m = build_model(R.NUM_CLASSES)
    # eval(): the model carries dropout=0.1, so two forward passes over the
    # SAME tensor differ in training mode and the comparison would fail for a
    # reason that has nothing to do with padding.
    m.eval()
    def dec(a):
        with torch.no_grad():
            t = torch.from_numpy(a)[None, None]
            return decode_greedy(m(t)[0].argmax(-1).numpy(), limit=limit)
    assert dec(served) == dec(legacy)


def test_serve_path_never_pads():
    """The whole fix in one assertion: served width == content width."""
    rng = np.random.default_rng(1)
    for cw in (32, 64, 128, 200, 255):
        crop = rng.uniform(-1, 1, (R.CROP_H, cw)).astype(np.float32)
        served, limit = FP.pad_for_export(crop, cw, graph_w=None)
        assert served.shape == (R.CROP_H, cw)
        assert limit == FP.timesteps(cw)
        assert np.array_equal(served, crop)


def test_legacy_path_still_pads_and_is_limited():
    """The removed behaviour, kept measurable so the fix cannot be vacuous."""
    rng = np.random.default_rng(2)
    crop = rng.uniform(-1, 1, (R.CROP_H, 64)).astype(np.float32)
    padded, limit = FP.pad_for_export(crop, 64, graph_w=256)
    assert padded.shape == (R.CROP_H, 256)
    assert np.array_equal(padded[:, :64], crop)
    assert not padded[:, 64:].any(), "the padding tail must be zeros"
    assert limit == 8


# ── the defect the fix removes, measured on the real checkpoint ─────────────

@pytest.mark.skipif(not CKPT.exists(), reason="no trained checkpoint")
@pytest.mark.skipif(not CACHE.exists(), reason="no training cache")
@pytest.mark.parametrize("content_w", [64, 128, 200])
def test_fixed_width_padding_changes_the_decode(content_w):
    """
    The RED GATE, restated as a negative control: on a fixed 256px graph, a
    real crop padded to 256 decodes DIFFERENTLY from the same crop at its own
    width, even after truncating to the valid frames.

    Without this test the invariant in test_padding_invariant.py could pass for
    a reason that has nothing to do with the fix.

    content_w=32 is deliberately excluded: there the argmax sequence diverges
    at frame 2 (23 vs 28) but CTC's repeat-collapse hides it and the strings
    agree by luck, so it cannot serve as a control.
    """
    import export_onnx as X
    from realdata import decode_greedy

    crop = X._sample_real_crop(content_w, 256)
    if crop is None:
        pytest.skip(f"no real crop of width {content_w} in the cache")
    crop = np.asarray(crop, np.float32)
    if crop.ndim == 3:
        crop = crop[0]

    m = build_model(R.NUM_CLASSES)
    m.load_state_dict(torch.load(str(CKPT), map_location="cpu",
                                 weights_only=False)["model"])
    m.eval()

    def run(a):
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(a, np.float32))
            return m(t[None, None])[0].numpy()

    natural = run(crop)
    padded, limit = FP.pad_for_export(crop, content_w, graph_w=FP.LEGACY_EXPORT_W)
    pad = run(padded)

    assert limit == natural.shape[0] == FP.timesteps(content_w)
    a = decode_greedy(natural.argmax(-1))
    b = decode_greedy(pad.argmax(-1), limit=limit)
    assert b != a, (
        f"fixed-width padding no longer changes the {content_w}px decode "
        f"({a!r} vs {b!r}) — re-measure before trusting the invariant test")

    # Not a trailing-character artifact: the logits over the VALID frames are
    # perturbed, which is why truncation alone cannot fix this.
    valid = float(np.abs(natural[:limit] - pad[:limit]).max())
    assert valid > 1e-3, (
        f"valid-region logits agree to {valid:.2e}; if the padding really were "
        f"inert, truncating the decode would have been enough")


# ── the fix's artifact: a dynamic-width ONNX graph ─────────────────────────

@pytest.fixture(scope="module")
def dynamic_onnx(tmp_path_factory):
    """
    Export the recogniser with a dynamic width axis, once per session.

    Uses the FUSED nn.LSTM (export_safe_lstm False). The explicit per-timestep
    loop the old exporter enables unrolls the recurrence at trace time, which
    bakes T into the graph; the fused LSTM lowers to a real ONNX LSTM op whose
    sequence axis is dynamic.
    """
    if not CKPT.exists():
        pytest.skip("no trained checkpoint")
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    m = build_model(R.NUM_CLASSES)
    m.load_state_dict(torch.load(str(CKPT), map_location="cpu",
                                 weights_only=False)["model"])
    m.eval()
    m.set_export_mode(False)
    path = tmp_path_factory.mktemp("onnx") / "crnn_dynamic.onnx"
    # dynamo=False IS LOAD-BEARING, not a style choice. torch 2.14 defaults to
    # dynamo=True (torch.export), whose decomposition of aten::lstm at a
    # symbolic sequence length fails with "stack expects each tensor to be
    # equal size ... torch.Size([Min(((((((s0 - 1)//4)) - 1)//2)) + 1, ...]"
    # — the second half of the same root cause export_crnn's docstring blamed
    # on the fused LSTM. The legacy JIT tracer maps the fused LSTM to a real
    # ONNX LSTM op whose sequence axis stays dynamic, which is the whole fix.
    torch.onnx.export(
        m, torch.zeros(1, 1, FP.CROP_H, FP.LEGACY_EXPORT_W), str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "b", 3: "w"}, "logits": {0: "b", 1: "t"}},
        opset_version=17, do_constant_folding=True, dynamo=False)
    return path, m, ort


@pytest.mark.parametrize("width", [32, 64, 128, 200, 256])
def test_dynamic_graph_has_no_fixed_width_and_matches_pytorch(dynamic_onnx, width):
    """
    The fix end to end: one graph, every width, no padding, PyTorch-equal.

    A graph with a baked width would either refuse the input or return the
    wrong number of timesteps; both are asserted.
    """
    path, m, ort = dynamic_onnx
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    decl = s.get_inputs()[0].shape
    assert not isinstance(decl[3], int), (
        f"the graph still declares a fixed width {decl}. Dynamic export bakes "
        f"nothing only if the width axis stays symbolic.")

    x = torch.rand(1, 1, FP.CROP_H, width)
    with torch.no_grad():
        yt = m(x).numpy()
    yo = s.run(None, {"input": x.numpy()})[0]
    assert yo.shape == yt.shape, (width, yt.shape, yo.shape)
    assert float(np.abs(yt - yo).max()) < 1e-3
    assert (yt.argmax(-1) == yo.argmax(-1)).all(), f"argmax differs at W={width}"
    assert yo.shape[1] == FP.timesteps(width)
