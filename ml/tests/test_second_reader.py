"""
Lane O tests — the independent second read.

Each test pins one thing that would silently rot: that every reader is
DETERMINISTIC (same crop -> same array twice; a phone runs it twice), that the
preprocessing readers actually CHANGE the pixels (a reader that is secretly the
baseline would make the whole decorrelation measurement a tautology), that the
agreement rule speaks on equality and refuses on difference naming both strings,
and that the decorrelation accounting counts a value wrong in r1 and right in r2
as a RESCUE rather than as an error of the pair.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import second_reader as SR  # noqa: E402


# ── fixtures: a synthetic crop with uneven illumination + a dark glyph block ──

def uneven_crop(h=24, w=120):
    """Background ramps 190 -> 70 across the width (the uneven illumination
    tile-CLAHE exists to normalise) with a row of dark strokes near the middle."""
    ramp = np.tile(np.linspace(190, 70, w), (h, 1)).astype(np.uint8)
    ramp[h // 2 - 3:h // 2 + 3, w // 4:3 * w // 4] = 20
    return ramp


CROP = uneven_crop()
NAMES = sorted(SR.READERS)


# ── determinism: the same crop twice gives the same model input ───────────────

@pytest.mark.parametrize("name", [SR.BASELINE] + NAMES)
def test_reader_is_deterministic(name):
    a = SR.reader_array(name, CROP)
    b = SR.reader_array(name, CROP)
    assert a.shape == b.shape
    assert np.array_equal(a, b), "reader %s is not deterministic" % name


def test_baseline_is_32_tall_and_h48_is_48():
    assert SR.reader_array(SR.BASELINE, CROP).shape[0] == 32
    assert SR.reader_array("r3_h48", CROP).shape[0] == 48


# ── the preprocessing readers must actually change the pixels ─────────────────

def test_clahe_reader_changes_the_crop_not_a_tautology():
    base = SR.reader_array(SR.BASELINE, CROP)
    clahe = SR.reader_array("r2_clahe", CROP)
    assert not np.array_equal(base, clahe), "r2_clahe == baseline: no second read"


def test_otsu_reader_is_bimodal_and_differs_from_clahe():
    # the BINARISATION is bimodal; the resize to CROP_H then interpolates, so
    # assert on the transform itself, not on the resized array.
    raw = SR._clahe_otsu(CROP)
    assert set(np.unique(raw).tolist()) <= {0, 255}, sorted(set(np.unique(raw).tolist()))[:6]
    assert not np.array_equal(SR.reader_array("r2b_otsu", CROP),
                              SR.reader_array("r2_clahe", CROP))


# ── the agreement rule ────────────────────────────────────────────────────────

def test_ensemble_speaks_when_readers_agree():
    spoke, refused = SR.ensemble({"date": "01102026"}, {"date": "01102026"})
    assert spoke == {"date": "01102026"}
    assert refused == []


def test_ensemble_agrees_across_normalisation():
    """Differently written, same normalised value -> agreement."""
    spoke, refused = SR.ensemble({"date": "01102026"}, {"date": "0110 2026"})
    assert spoke == {"date": "01102026"} and not refused


def test_ensemble_refuses_on_disagreement_naming_both_strings():
    spoke, refused = SR.ensemble({"date": "04021879"}, {"date": "04021979"})
    assert spoke == {}
    assert len(refused) == 1
    r = refused[0]
    assert r["field"] == "date"
    assert "04021879" in r["reason"] and "04021979" in r["reason"], r["reason"]


def test_ensemble_refuses_when_second_reader_is_silent():
    spoke, refused = SR.ensemble({"dosage": "500mg"}, {})
    assert spoke == {}
    assert refused[0]["other"] is None and "500mg" in refused[0]["reason"]


# ── decorrelation accounting ──────────────────────────────────────────────────

GT = ["04021979", "01102026"]


def test_wrong_in_r1_right_in_r2_is_a_rescue_not_an_error():
    """The whole point of the lane: r1 reads the date one digit off, r2 reads it
    right. That value must be counted as RESCUED, and the pair must not be
    charged an error for it."""
    r1 = {"date_of_birth": "04021879"}          # wrong (one digit off)
    r2 = {"date_of_birth": "04021979"}          # right
    A = SR.accord(r1, r2, GT)
    assert A["r1_wrong"] == 1
    assert A["r1_wrong_r2_right"] == 1          # RESCUED
    assert A["both_wrong_agree_same"] == 0
    assert A["rescued"] == ["date_of_birth"]


def test_both_wrong_same_string_is_the_irreducible_core():
    r1 = {"date_of_birth": "04021879"}
    r2 = {"date_of_birth": "04021879"}
    A = SR.accord(r1, r2, GT)
    assert A["r1_wrong_r2_right"] == 0
    assert A["both_wrong_agree_same"] == 1      # unfixable by agreement
    assert A["core_same"] == ["date_of_birth"]


def test_both_wrong_different_string_is_refusable_by_agreement():
    r1 = {"date": "01102024"}
    r2 = {"date": "01032026"}
    A = SR.accord(r1, r2, GT)
    assert A["both_wrong_differ"] == 1
    assert A["both_wrong_agree_same"] == 0


def test_r1_right_r2_wrong_is_the_price_of_agreement():
    r1 = {"date": "01102026"}                    # right
    r2 = {"date": "0110202b"}                    # wrong
    A = SR.accord(r1, r2, GT)
    assert A["r1_right_r2_wrong"] == 1
    # and the ensemble refuses it (readers disagree)
    spoke, refused = SR.ensemble(r1, r2)
    assert spoke == {} and refused


def test_wrong_in_r1_and_silent_in_r2_is_not_a_rescue():
    A = SR.accord({"date": "04021879"}, {}, GT)
    assert A["r1_wrong"] == 1
    assert A["r1_wrong_r2_right"] == 0
    assert A["r1_wrong_r2_silent"] == 1


# ── critical_spoken: the policy restricted to the safety-critical kinds ───────

def test_critical_spoken_keeps_critical_drops_free_text():
    lines = [{"text": "DATE 01102026", "box": (0, 0, 120, 32)},
             {"text": "PATIENT JANE DOE DOB 04021979", "box": (0, 0, 200, 32)},
             {"text": "PHARMACY NORTHGATE PHARMACY", "box": (0, 0, 200, 32)}]
    crit, _dec, fields = SR.critical_spoken(lines, "medical")
    assert crit.get("date") == "01102026"
    assert crit.get("date_of_birth") == "04021979"     # DOB is a date -> critical
    assert "patient" not in crit and "pharmacy" not in crit
    assert fields["pharmacy"]["kind"] == "general"


# ── end to end through a stub model: read_page is deterministic per reader ────

class _StubModel:
    """Constant logits -> the greedy decode emits blanks. Enough to exercise the
    recognise() plumbing (softmax, argmax, collapse) without a checkpoint."""

    def __call__(self, t):
        import torch
        tlen = max(1, int(t.shape[-1]) // 4)
        return torch.zeros(1, tlen, len(SR.R.CHARSET) + 1)


def test_read_page_deterministic_through_the_recogniser():
    crops = [CROP, uneven_crop(20, 90)]
    m = _StubModel()
    for name in NAMES + [SR.BASELINE]:
        a = SR.read_page(crops, m, "cpu", name)
        b = SR.read_page(crops, m, "cpu", name)
        assert [x["text"] for x in a] == [x["text"] for x in b]
        assert all("box" in x for x in a)


def test_baseline_read_reuses_read_page_lines(monkeypatch):
    """r1 must be eval_field_recovery.read_page_lines, not a copy — the whole
    baseline claim rests on it."""
    import eval_field_recovery as FR
    seen = {}

    def fake(crops, model, device):
        seen["called"] = True
        return [{"text": "X", "box": (0, 0, 1, 32)}]

    monkeypatch.setattr(FR, "read_page_lines", fake)
    SR.read_page([CROP], object(), "cpu", SR.BASELINE)
    assert seen.get("called") is True
