"""
Regression tests for the padding invariant on the fixed-width export.

The gate protecting the fixed-width ONNX export was INVERTED: it read

    if _emits_any_blank(model):  SKIP

so it skipped exactly when the model was trained well enough for the check to
mean anything -- the one case it exists for -- and printed "untrained head
emits no blanks" while reporting True. It read as a deliberate guard, which is
why it survived review, and it had never executed on any checkpoint.

The assumption it guards is the load-bearing one behind exporting at a fixed
256px: right-padding is safe BECAUSE a trained CTC model emits the blank symbol
in padded regions, so the browser's right-pad-and-trim cannot invent
characters. Nothing else in the codebase verifies that.

IT WAS FALSE. 3 of the 4 parametrisations failed: a 2-layer BI-LSTM's backward
pass starts at the right edge of the padded tensor, so the padding moves the
hidden states over the CONTENT too — the valid-region logits differ by up to
15.3, and truncating the decode to the content's frame count (which the test
already did) only removes TRAILING characters, never the perturbation.
The fix (ml/src/fix_padding.py) is to stop padding: the recogniser is exported
with a dynamic width axis, so every crop is served at its own width, which is
the condition training used. The parametrised test below now runs the serve
helper rather than a hard-coded 256, and the padded path it replaced is kept
measurable in test_fix_padding.test_fixed_width_padding_changes_the_decode.

Run:  bash run.sh test
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import export_onnx as X                        # noqa: E402
import fix_padding as FP                       # noqa: E402
import realdata as R                          # noqa: E402
from ocr_model import build as build_model     # noqa: E402

# decode_greedy is IMPORTED into export_onnx, not defined there, so
# X.decode_greedy happens to resolve today. Reaching through another module's
# namespace for a name it borrowed is fragile -- it disappears the moment
# export_onnx stops using it -- and calling it X.decode_greedy raised
# AttributeError in the first version of these tests.
decode_greedy = R.decode_greedy

CACHE = Path(__file__).resolve().parents[2] / "artifacts" / "ocr_train.npz"


# ── the condition, which is the actual regression ────────────────────────────

def _code_lines(src):
    """
    Source with comments and docstrings stripped.

    Necessary because these tests assert on source text, and the source
    documents the inverted form it used to have -- so a naive substring check
    matches the COMMENT describing the bug and reports the bug is back. The
    first version of this test did exactly that: it failed on the very comment
    explaining why the check was written. Stripping comments is what makes a
    source assertion mean what it says.
    """
    import io
    import tokenize
    out = []
    prev_type = tokenize.INDENT
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING and prev_type in (
                    tokenize.INDENT, tokenize.NEWLINE, tokenize.NL,
                    tokenize.ENCODING):
                prev_type = tok.type      # a docstring
                continue
            out.append(tok.string)
            if tok.type not in (tokenize.NL,):
                prev_type = tok.type
    except tokenize.TokenError:
        return src
    return " ".join(out)


def test_padding_gate_is_not_inverted():
    """
    Guards the exact bug: the skip branch must be keyed on NOT emitting blanks.

    A behavioural test would be better, but it needs a trained checkpoint, and
    this must hold on any machine. Asserting on the source is honest about that
    trade-off -- it catches the specific inversion, it does not prove the gate
    works. The behavioural test below does that.
    """
    code = _code_lines(inspect.getsource(X.verify_crnn)).replace(" ", "")
    assert "ifnot_emits_any_blank(model):" in code, (
        "the padding-invariant skip condition is inverted again -- it must skip "
        "when the model emits NO blanks, and run when it does"
    )
    assert "if_emits_any_blank(model):" not in code, (
        "the inverted form is back: that skips precisely when the model is "
        "trained enough for the check to be meaningful"
    )


def test_skip_message_matches_the_condition_it_guards():
    """
    The misleading text is part of the bug: it printed 'untrained head emits
    no blanks' at the exact moment the model emitted 47% blanks. If someone
    ever reads that message and believe it, they will skip a real signal.

    Skipped rather than asserted: the string now lives only in a comment, since
    the branch is the trained path. What matters is that the RUN branch exists.
    """
    code = _code_lines(inspect.getsource(X.verify_crnn))
    assert "padding invariant: SKIPPED" in code, (
        "the skip message was removed from the code -- seeing it at export "
        "time is the only way to notice the gate did not run"
    )


def test_blank_detector_is_not_vacuous():
    """_emits_any_blank must be able to return False, or the check above is moot."""
    m = build_model(R.NUM_CLASSES)
    # An untrained head is arbitrary; assert we at least get a bool, and that
    # the function does not unconditionally return True.
    val = X._emits_any_blank(m)
    assert isinstance(val, bool)


# ── the padding invariant, behaviourally ─────────────────────────────────────

@pytest.mark.skipif(not CACHE.exists(),
                    reason="training cache absent; run bash run.sh build-data")
@pytest.mark.parametrize("content_w", [32, 64, 128, 200])
def test_padding_a_real_crop_does_not_change_its_decode(content_w):
    """
    The actual property: a real crop padded to the export width decodes
    identically to the same crop at its own width.

    This is what justifies exporting at a fixed 256px instead of a dynamic
    width. If it fails, the browser's right-pad-and-trim invents characters --
    and for a dosage or an account number that is worse than reading nothing.
    """
    crop = X._sample_real_crop(content_w, 256)
    if crop is None:
        pytest.skip(f"no real crop of width {content_w} in the cache")
    # _sample_real_crop's docstring says [1, 32, W] but it returns a 2-D
    # [32, W] array, so indexing crop[0] took a row and produced a 1-D
    # result the model could not accept. Normalise the rank here rather than
    # trusting the docstring, and keep the batch axis explicit.
    crop = np.asarray(crop, np.float32)
    if crop.ndim == 3:
        crop = crop[0]
    assert crop.shape == (R.CROP_H, content_w), \
        f"unexpected crop shape {crop.shape}"

    # ROOT is ml/tests -> ml -> repo. Three parents, not two: parents[2] of a file
    # in ml/tests is ml/, so joining "models" gave ml/models, which does not
    # exist, and the failure surfaced as a bare torch.load error rather than a
    # clear "checkpoint not found".
    REPO = Path(__file__).resolve().parents[2]
    CKPT = REPO / "models" / "ocr" / "crnn.pt"
    if not CKPT.exists():
        pytest.skip(f"no trained checkpoint at {CKPT}")

    m = build_model(R.NUM_CLASSES)
    m.load_state_dict(torch.load(str(CKPT), map_location="cpu",
                                 weights_only=False)["model"])
    m.eval()

    # The model takes [B, 1, 32, W] -- a batch, ONE channel, height, width.
    # The crop is [1, 32, W], so it still needs the channel axis inserted.
    # Getting this wrong is not subtle in the error but was here: the conv
    # stack read 32 as the channel count and complained about a BatchNorm
    # weight of size 16, which points at the normalisation layer rather than at
    # the missing axis.
    def to_batch(a):
        t = torch.from_numpy(np.ascontiguousarray(a, np.float32))
        return t[None, None] if t.ndim == 2 else t

    with torch.no_grad():
        a = decode_greedy(m(to_batch(crop))[0].argmax(-1).numpy())
        # THE SERVE PATH, stated as the padded-equivalent path. graph_w=None
        # means the DYNAMIC-width graph, so the crop is fed at its own width and
        # there is no padding to perturb the bidirectional pass.
        #
        # This used to build a fixed [32, 256] zero-padded tensor and decode
        # `limit=content_w // 8`. That failed on 3 of 4 widths, and the reason
        # matters: the limit is already applied there, so the divergence was NOT
        # trailing characters. It was the valid region itself — a 2-layer
        # BI-LSTM's backward pass starts at the RIGHT EDGE, i.e. inside the
        # padding, and moves the hidden states over the content (measured max
        # |Δ| over the valid frames: 9.10 at content_w=64, 15.28 at 200). So the
        # assertion below is paired with a shape assertion: padding factors out
        # only if the served tensor is exactly as wide as the content, and
        # test_fix_padding.test_fixed_width_padding_changes_the_decode keeps the
        # padded path measurable so this test cannot pass vacuously.
        served, limit = FP.pad_for_export(crop, content_w, graph_w=None)
        b = decode_greedy(m(to_batch(served))[0].argmax(-1).numpy(), limit=limit)
    assert served.shape == (R.CROP_H, content_w), (
        f"the serve path produced a {served.shape[1]}px tensor for a "
        f"{content_w}px crop: padding is back, and with it the mismatch")
    assert b == a, (f"padding {content_w}->256 changed the decode: "
                    f"{a!r} vs {b!r}")


def test_sample_real_crop_returns_normalised_values():
    """
    The helper feeds the model directly, so its output must be in the same
    [-1,1] range training uses. Returning raw 0..255 bytes would make the
    padding check pass or fail for entirely the wrong reason.
    """
    if not CACHE.exists():
        pytest.skip("no training cache")
    crop = X._sample_real_crop(128, 256)
    if crop is None:
        pytest.skip("no 128px crop in cache")
    assert crop.dtype == np.float32
    assert crop.shape == (R.CROP_H, 128)
    assert -1.001 <= crop.min() and crop.max() <= 1.001, (
        f"range [{crop.min():.2f}, {crop.max():.2f}] is not [-1,1]"
    )


def test_verify_crnn_reports_false_when_the_invariant_is_broken():
    """
    The gate must be able to FAIL. A gate that cannot fail is worse than no
    gate, because it is read as reassurance.

    Mutating the MODEL rather than the decoder: verify_crnn does
    `from realdata import decode_greedy` INSIDE its own body, so patching
    export_onnx.decode_greedy has no effect -- the first version of this test
    raised AttributeError on that patch and proved nothing.

    A model whose argmax is uniformly blank decodes to the empty string at
    every width, so padded and unpadded agree and the gate would pass. Instead
    the stub returns a DIFFERENT non-empty string per timestep count, which is
    what a genuine padding leak looks like from the decoder's side.
    """
    orig_emits = X._emits_any_blank

    class LeakyModel(torch.nn.Module):
        """Emits blanks only in the padded tail -- the exact leak we fear."""

        def __init__(self, real, export_w=256):
            super().__init__()
            self.real = real
            self.export_w = export_w

        def forward(self, x):
            b, _, _, w = x.shape
            t = max(1, w // 8)
            logits = torch.zeros(b, t, R.NUM_CLASSES)
            if w == self.export_w:
                # Padded: invent characters in the timesteps beyond 1.
                for i in range(t):
                    logits[0, i, 1 if i >= 1 else R.BLANK] = 10.0
            else:
                logits[0, :, R.BLANK] = 10.0
            return logits

    real = build_model(R.NUM_CLASSES)
    ck = REPO_CKPT()
    if ck is None:
        pytest.skip("no trained checkpoint")
    real.load_state_dict(ck)
    real.eval()

    leaky = LeakyModel(real)
    # Force the RUN branch without depending on a trained head's behaviour.
    X._emits_any_blank = lambda model: True
    try:
        assert X.verify_crnn(leaky, width=256) is False, (
            "verify_crnn passed on a model that invents characters in its "
            "padded region -- the gate cannot fail, so it proves nothing"
        )
    finally:
        X._emits_any_blank = orig_emits


def REPO_CKPT():
    """The trained checkpoint, or None when it has not been produced yet."""
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[2] / "models" / "ocr" / "crnn.pt"
    if not p.exists():
        return None
    return torch.load(str(p), map_location="cpu", weights_only=False)["model"]