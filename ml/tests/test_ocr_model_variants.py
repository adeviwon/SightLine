"""
Capacity knobs on the CRNN — WITHOUT changing the shipped default.

The app ships models/ocr/crnn.onnx with a fixed [1,1,32,256] input, and a
training run is writing a checkpoint that must load into the SAME architecture.
So the whole point of this file is the contradiction it resolves: knobs for
capacity experiments (build(channels=48), build(lstm_layers=3),
build(input_height=48)) that leave build() with no arguments byte-identical to
the frozen backup models/ocr/crnn_before_render.pt.

If a refactor changes any state_dict key or shape for the DEFAULT arguments,
the first test below fails and says which keys diverged — that is the gate
that protects both the running training and the exported ONNX graph.

Run:  bash run.sh test
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_model import build  # noqa: E402

# The frozen backup. models/ocr/crnn.pt is being REWRITTEN by a live training
# run, so it must never be used as a reference here.
CKPT = (Path(__file__).resolve().parents[2] / "models" / "ocr"
        / "crnn_before_render.pt")


def _frozen_state():
    """The state_dict of the frozen backup, or skip if it is not on disk."""
    if not CKPT.exists():
        pytest.skip(f"frozen checkpoint missing: {CKPT}")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck


# ── the hard requirement: the default IS the shipped architecture ────────────

def test_default_is_byte_compatible_with_frozen_checkpoint():
    """
    Same keys, same shapes, loadable. This is what keeps the exported ONNX
    graph and the in-flight training run valid.

    Reports the offending keys rather than a bare `assert False`: "keys differ"
    on a 38-key dict is unactionable, and the likely cause (a renamed conv, a
    dropped norm) is only obvious when named.
    """
    want = _frozen_state()
    got = build().state_dict()

    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    assert not missing and not extra, (
        f"state_dict KEYS changed for default build().\n"
        f"  missing: {missing}\n  extra:   {extra}")

    bad = {k: (tuple(got[k].shape), tuple(want[k].shape))
           for k in want if tuple(got[k].shape) != tuple(want[k].shape)}
    assert not bad, f"state_dict SHAPES changed for default build(): {bad}"

    # Strongest form: the frozen weights actually load, with no missing/unexpected.
    build().load_state_dict(want)


def test_default_param_count_is_the_shipped_one():
    """The frozen run recorded its own parameter count; match it exactly."""
    if not CKPT.exists():
        pytest.skip(f"frozen checkpoint missing: {CKPT}")
    recorded = torch.load(CKPT, map_location="cpu",
                          weights_only=False).get("params")
    n = build().n_params()
    print(f"  default params: {n:,}  (checkpoint recorded {recorded:,})")
    assert recorded is None or n == recorded, f"{n:,} != {recorded:,}"


# ── the knobs add capacity ───────────────────────────────────────────────────

def test_variants_have_strictly_more_parameters(capsys):
    """
    A width knob and a depth knob that do not add capacity are silently
    no-ops. Print the counts so a human can quote them in the write-up.
    """
    base = build().n_params()
    wide = build(channels=48).n_params()
    deep = build(lstm_layers=3).n_params()
    tall = build(input_height=48).n_params()

    out = (f"  params  default={base:,}  channels=48={wide:,}  "
           f"lstm_layers=3={deep:,}  input_height=48={tall:,}")
    print(out)

    assert wide > base, f"channels=48 must add capacity: {wide} vs {base}"
    assert deep > base, f"lstm_layers=3 must add capacity: {deep} vs {base}"
    # Height is a geometry knob, not a capacity knob: the conv stack sees the
    # same number of weights either way. Assert that, so nobody "fixes" it.
    assert tall == base, f"input_height must not change params: {tall} vs {base}"


# ── the timesteps helper agrees with a REAL forward pass ─────────────────────

@pytest.mark.parametrize("kwargs", [{}, {"channels": 48}, {"input_height": 48}])
def test_timesteps_matches_a_real_forward(kwargs):
    """
    timesteps() is what train_ocr uses to size the CTC input_lengths and to
    drop infeasible crops. Checked against the graph's actual output, not
    against a formula written here — a formula would agree with itself while
    both disagreed with the model.
    """
    m = build(**kwargs).eval()
    with torch.no_grad():
        y = m(torch.randn(1, 1, m.input_height, 256))
    assert y.shape[0] == 1 and y.shape[2] == m.num_classes
    assert y.shape[1] == m.timesteps(256), (
        f"{kwargs}: forward gives {y.shape[1]} timesteps, helper says "
        f"{m.timesteps(256)}")


# ── input height 48 ──────────────────────────────────────────────────────────

def test_height_48_forwards_and_keeps_class_count():
    """A 48px-tall crop must collapse to 1 row and emit the same classes."""
    m = build(input_height=48).eval()
    with torch.no_grad():
        y = m(torch.randn(1, 1, 48, 256))
    assert y.shape[2] == 38, y.shape
    assert y.shape[1] == m.timesteps(256)
    assert m.cnn(torch.zeros(1, 1, 48, 256)).shape[2] == 1


# ── CTC feasibility ──────────────────────────────────────────────────────────

def test_ctc_feasibility_helper_agrees_with_a_real_forward():
    """
    A label of length L is trainable only when timesteps(width) >= L. The
    filter in train_ocr decides that from timesteps(); if the helper ever
    over-counts, crops are kept whose CTC loss is infinite (target longer than
    the input), which shows up as a NaN epoch rather than an error here.
    """
    L = 12
    for kwargs in ({}, {"channels": 48}, {"input_height": 48}):
        m = build(**kwargs).eval()
        for w in (64, 256):
            with torch.no_grad():
                T = int(m(torch.randn(1, 1, m.input_height, w)).shape[1])
            assert m.timesteps(w) == T, (kwargs, w, m.timesteps(w), T)
            # The filter's decision, made two ways, must be the same decision.
            assert (m.timesteps(w) >= L) == (T >= L), (kwargs, w, L)
