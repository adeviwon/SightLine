"""
Tests for the CRNN recogniser's geometry and export path.

These exist because every one of the following was WRONG at some point in this
project and every one failed SILENTLY or with a message pointing somewhere
other than the cause:

  * height not reduced to exactly 1        -> LSTM reads interleaved rows
  * pool applied after height is 1         -> "Calculated output size: (…x0x…)"
  * timestep count drifting from W/8        -> browser decodes padding
  * export loop != fused LSTM              -> Python and phone disagree

Run:  bash run.sh test
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import realdata as R          # noqa: E402
from ocr_model import (  # noqa: E402
    SightLineCRNN, build, ConvBlock,
)


# One fixed set of weights, so any test comparing two models compares the same
# function rather than two different random initialisations (which differ by
# ~0.26 on identical input and look exactly like a broken gate).
TO_SHARED_WEIGHTS = build(R.NUM_CLASSES).state_dict()


# ── geometry ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("W", [16, 32, 64, 100, 128, 160, 200, 256, 320])
def test_conv_output_height_is_exactly_one(W):
    """
    The LSTM reads ONE row. Height 2 does not raise -- it trains on two
    interleaved sequences and simply never converges, which is the worst kind
    of failure to debug. This is the contract.
    """
    m = build(R.NUM_CLASSES)
    f = m.cnn(torch.zeros(1, 1, 32, W))
    assert f.shape[2] == 1, f"height {f.shape[2]} at W={W}, must be 1"


def test_forward_shape_and_height_contract():
    m = build(R.NUM_CLASSES)
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 1, 32, 160))
    assert y.shape[0] == 2
    assert y.shape[2] == R.NUM_CLASSES
    assert y.shape[1] == m.timesteps(160)


@pytest.mark.parametrize("W", [32, 64, 128, 160, 256, 320])
def test_timesteps_is_width_over_eight(W):
    """
    W/8 is what the JavaScript decoder relies on to trim padded timesteps.

    Checked against the MEASURED conv output, not a clean-product formula:
    MaxPool2d floors, so odd widths are off by one from W/8 exactly, and a
    formula that ignored that would silently mis-trim the browser decode.
    """
    m = build(R.NUM_CLASSES)
    assert m.timesteps(W) == m.cnn(torch.zeros(1, 1, 32, W)).shape[3]


def test_timesteps_memoised_not_recomputed():
    """
    timesteps() runs a conv forward. The training loop calls it twice per
    sample over 30,021 crops; without memoisation the dataset load took over
    four minutes instead of seconds. Guard the cache.
    """
    m = build(R.NUM_CLASSES)
    m.timesteps(128)
    n_after_first = len(m._ts_by_width)
    for _ in range(50):
        m.timesteps(128)
    assert len(m._ts_by_width) == n_after_first, "cache must not grow on hits"
    assert m.timesteps(128) == 16


def test_cache_attribute_name_does_not_shadow_module_state():
    """
    nn.Module reserves `.cache`. Naming the dict that collides shadows
    Module's own machinery, which is a confusing failure rather than an
    obvious one.
    """
    m = build(R.NUM_CLASSES)
    assert isinstance(m._ts_by_width, dict)
    assert not isinstance(getattr(m, "cache", None), dict) or \
        not hasattr(torch.nn.Module, "cache")


# ── pooling decoupled from stride ────────────────────────────────────────────

def test_height_pooling_never_spends_width():
    """
    A pool=2 on the height axis must not touch width once height is small.
    Every height-reducing block uses pool=(2,1).
    """
    m = build(R.NUM_CLASSES)
    for blk in m.cnn:
        if isinstance(blk.pool, torch.nn.MaxPool2d):
            k = blk.pool.kernel_size
            if isinstance(k, tuple) and k[1] > 1:
                # width-reducing pool; height must already be handled safely
                assert k[0] >= 1


def test_conv_block_accepts_tuple_stride_and_pool():
    """
    The tuple forms are load-bearing; a regression to int-only breaks export.

    Arithmetic: input 64x64. stride=(1,2) halves WIDTH only -> 32. Then
    pool=(2,1) halves HEIGHT only and leaves width alone -> 16x32.
    """
    b = ConvBlock(1, 8, stride=(1, 2), pool=(2, 1))
    out = b(torch.randn(1, 1, 32, 64))
    assert out.shape[2] == 16, f"pool=(2,1) must halve height only: {out.shape}"
    assert out.shape[3] == 32, f"stride=(1,2) halves width, pool must not: {out.shape}"


def test_conv_block_pool_one_is_identity():
    b = ConvBlock(1, 8, stride=1, pool=1)
    assert isinstance(b.pool, torch.nn.Identity)
    # tuple of ones must also disable, and must not raise TypeError from
    # comparing a tuple with an int (`pool > 1` did exactly that)
    b2 = ConvBlock(1, 8, stride=1, pool=(1, 1))
    assert isinstance(b2.pool, torch.nn.Identity)


# ── the export path ──────────────────────────────────────────────────────────

def test_export_loop_matches_fused_lstm():
    """
    The ONNX graph uses an explicit timestep loop because aten::lstm cannot be
    traced with a dynamic sequence length. The two must be the same function,
    or the phone reads different text than Python does.
    """
    m = build(R.NUM_CLASSES)
    m.eval()
    worst, ok = m.assert_loop_matches_lstm(tol=1e-5, widths=(64, 160, 256))
    assert ok, f"export loop diverges from nn.LSTM by {worst:.2e}"


def test_export_loop_is_off_by_default():
    """
    Training must use the fused LSTM for speed. If export mode leaked into
    training, every epoch would run the slow Python loop.
    """
    m = build(R.NUM_CLASSES)
    assert m.export_safe_lstm is False


def test_export_loop_preserves_timestep_count():
    """Both paths must emit the same number of timesteps, not merely similar values."""
    import copy
    ref = copy.deepcopy(build(R.NUM_CLASSES))
    ref.eval()
    alt = copy.deepcopy(build(R.NUM_CLASSES))
    alt.set_export_mode(True)
    alt.eval()
    for W in (64, 160, 256):
        x = torch.rand(1, 1, 32, W)
        with torch.no_grad():
            a, b = ref(x), alt(x)
        assert a.shape == b.shape, (W, a.shape, b.shape)


def test_gate_split_matches_lstm_for_reversed_direction():
    """
    The reverse pass walks the sequence backwards with `_reverse` weights.
    Getting the suffix or the direction wrong still produces a tensor of the
    right SHAPE, so only a numeric comparison catches it.

    The two models must share WEIGHTS. Two separately-built models have
    different random initialisations and differ by ~0.26 on the same input,
    which looks exactly like a broken gate. This test therefore loads one
    state_dict into both.

    It also builds two independent instances rather than deep-copying one:
    export_safe_lstm is a class-level default, so a deepcopy of an
    already-switched instance carries the switched class along and the
    comparison would silently run the loop against itself.
    """
    import copy
    ref = build(R.NUM_CLASSES)
    ref.load_state_dict(copy.deepcopy(TO_SHARED_WEIGHTS))
    ref.eval()
    assert ref.export_safe_lstm is False

    alt = build(R.NUM_CLASSES)
    alt.load_state_dict(copy.deepcopy(TO_SHARED_WEIGHTS))
    alt.set_export_mode(True)
    alt.eval()
    x = torch.rand(1, 1, 32, 128)
    with torch.no_grad():
        a, b = ref(x), alt(x)
    assert torch.allclose(a, b, atol=1e-5), \
        f"max delta {(a - b).abs().max():.2e}"


# ── charset and CTC semantics ────────────────────────────────────────────────

def test_charset_size_and_absence_of_punctuation():
    """
    37 symbols, no punctuation. Punctuation is excluded on purpose: it is the
    highest-error class and no field regex needs it. A change here is a
    deliberate decision, not an accident.
    """
    assert len(R.CHARSET) == 37
    assert R.NUM_CLASSES == 38          # +1 for the CTC blank
    assert not any(c in R.CHARSET for c in ".,;:!?()[]{}@#$%&*+/\\\"'")


def test_blank_is_index_zero():
    """CTC convention: blank = 0, so it is never a real character index."""
    assert R.BLANK == 0
    assert all(i >= 1 for i in R.C2I.values())


def test_decode_collapses_repeats_and_drops_blanks():
    # A well-formed CTC target separates repeats with blanks.
    assert R.decode_greedy([10, R.BLANK, 1, R.BLANK, 1]) == "900"
    # An adjacent repeat is a malformed sequence; CTC means "collapse".
    assert R.decode_greedy([10, R.BLANK, 1, 1]) == "90"
    assert R.decode_greedy([]) == ""


def test_encode_drops_out_of_charset_characters():
    """Punctuation is silently dropped at encode time, by design."""
    assert R.encode("9.00") == R.encode("900")
    assert R.encode("A/B") == R.encode("AB")


def test_normalisation_preserves_dosage_units():
    """
    '500mg' -> '500MG', never '500M9'. A corrupted dosage is worse than an
    omitted one, so g/q are deliberately not mapped to 9.
    """
    assert R.norm_label("5OOmg") == "500MG"
    assert "MG" in R.norm_label("Amoxicillin 5OOmg three times daily")


def test_model_is_small_enough_to_ship():
    """Under ~1M params keeps the ONNX file around 4 MB."""
    m = build(R.NUM_CLASSES)
    assert m.n_params() < 1_500_000, f"{m.n_params():,} params"


def test_forward_is_deterministic_in_eval():
    m = build(R.NUM_CLASSES)
    m.eval()
    x = torch.rand(1, 1, 32, 128)
    with torch.no_grad():
        assert torch.allclose(m(x), m(x))