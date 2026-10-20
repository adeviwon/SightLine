"""
Tests for eval_field_truth.py — the RENDERED-ground-truth field measurement.

All synthetic: no checkpoint, no corpus, no model. The tests pin the SCORING
RULES the three-path table rests on, because the two previous rounds of
measurement were wrong in exactly these places:

  * the ground truth is the RENDERED line's continuation after each spotted
    label, not the two-token evaluate.FIELDS list;
  * a value that passes validation but is WRONG counts as wrong-but-spoken and
    drops precision (the product-safety number);
  * a spot whose label is absent from the true line is a FALSE POSITIVE, and it
    is counted, not skipped;
  * a page nothing was spotted on yields recall 0 and does not crash;
  * the baseline path IS evaluate.fields_found, not a reimplementation;
  * a gain the sample cannot support is INCONCLUSIVE, never IMPROVED.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_field_truth as FT  # noqa: E402
import evaluate as E           # noqa: E402


def L(*texts):
    """Line dicts in page order, with the geometry the harness hands the modules."""
    return [{"text": t, "box": (0, 0, 300, 32)} for t in texts]


# ── a perfect read scores precision 1.0 against the rendered truth ─────────────

def test_perfect_read_scores_precision_one_on_spot_and_validated_paths():
    row = FT.score_page(["SORT CODE 401104"], L("Sort Code: 40-11-04"), "banking")
    assert row["gt_values"] == ["401104"]
    assert row["n_gt"] == 1 and row["n_fp_spots"] == 0
    assert row["spot_hit"] == len(row["spot_spoken"]) == 1
    assert row["valid_hit"] == len(row["valid_spoken"]) == 1
    assert row["valid_wrong"] == []
    assert row["refused"] == 0


def test_perfect_transcription_of_the_dose_line_is_recovered_by_both_paths():
    """The bug that motivated this file: a dose line read character-for-character
    must not score a MISS. GT is the true continuation 'AMOXICILLIN 500MG'."""
    true, ocr = "RX AMOXICILLIN 500MG CAPSULES", "Rx  Amoxicillin 500mg capsules"
    row = FT.score_page([true], L(ocr), "prescription")
    assert ("RX", "AMOXICILLIN 500MG") in row["gt_pairs"]
    assert row["spot_recovered"] == row["gt_values"]      # extractor got it
    assert row["baseline_spoken"] == ["500mg"]            # regex got its token
    assert float(row["baseline_hit"]) == 1.0              # and both count
    assert float(row["spot_hit"]) == 1.0


# ── validation is the safety half: valid but WRONG is the number that matters ──

def test_a_valid_but_wrong_date_is_wrong_but_spoken_and_drops_precision():
    row = FT.score_page(["DATE 01102026"], L("Date: 01/11/2026"), "prescription")
    assert row["gt_values"] == ["01102026"]
    assert row["valid_spoken"] == ["01/11/2026"]          # accepted by the validator
    assert row["refused"] == 0
    assert row["valid_wrong"] == ["01/11/2026"]           # ... and it is WRONG
    assert row["valid_hit"] == 0
    s = FT.summarise([row])
    assert s["valid"]["spoken"] == 1 and s["valid"]["prec_k"] == 0
    assert s["wrong"] == 1


def test_a_wrong_but_spoken_value_lowers_precision_below_a_correct_one():
    good = FT.score_page(["DATE 01102026"], L("Date: 01102026"), "prescription")
    bad = FT.score_page(["DATE 01102026"], L("Date: 01/11/2026"), "prescription")
    sg, sb = FT.summarise([good]), FT.summarise([bad])
    assert sg["valid"]["prec_k"] / sg["valid"]["spoken"] == pytest.approx(1.0)
    assert sb["valid"]["prec_k"] / sb["valid"]["spoken"] == pytest.approx(0.0)


# ── a spot whose label is not in the true line is a FALSE POSITIVE ─────────────

def test_spot_whose_label_is_absent_from_the_true_line_counts_as_false_positive():
    row = FT.score_page(["THE QUICK BROWN FOX"], L("Sort Code: 401104"), "banking")
    assert row["n_fp_spots"] >= 1
    assert row["n_gt"] == 0
    assert row["spot_spoken"]                         # it still spoke a value
    assert float(row["spot_hit"]) == 0.0              # which cannot be correct
    assert FT.summarise([row])["fp_spots"] >= 1


# ── an empty page is scored, never skipped and never crashed on ────────────────

def test_page_with_nothing_spotted_yields_recall_zero_without_crashing():
    row = FT.score_page(["HELLO WORLD"], L("hello world"), "prescription")
    assert row["n_gt"] == 0 and row["n_fp_spots"] == 0
    assert row["spot_spoken"] == [] and row["valid_spoken"] == []
    s = FT.summarise([row])
    assert s["spot"]["recall_k"] == 0 and s["spot"]["gt"] == 0
    assert FT.rate(0, 0) == 0.0                       # no denominator, no claim


def test_blank_page_with_no_lines_is_scored_not_skipped():
    row = FT.score_page([], [], "prescription")
    assert row["n_gt"] == 0 and row["spot_spoken"] == [] and row["refused"] == 0


# ── the baseline path is evaluate.fields_found, not a reimplementation ─────────

def test_baseline_path_is_evaluate_fields_found():
    ocr = L("Sort Code: 40-11-04", "Account Number: 40218877")
    row = FT.score_page(["SORT CODE 401104", "ACCOUNT NUMBER 40218877"], ocr, "banking")
    text = " ".join(l["text"] for l in ocr)
    assert row["baseline_spoken"] == E.fields_found(text, "banking")


# ── ground truth = the rendered line's continuation after the label ────────────

def test_ground_truth_is_truncated_at_the_next_label_on_the_same_line():
    row = FT.score_page(["PATIENT JANE DOE DOB 04021979"],
                        L("Patient: Jane Doe   DOB: 04/02/1979"), "prescription")
    assert row["gt_pairs"] == [("PATIENT", "JANE DOE"), ("DOB", "04021979")]


def test_a_label_printed_twice_matches_two_distinct_true_occurrences():
    row = FT.score_page(["PHARMACY NORTHGATE PHARMACY"],
                        L("Pharmacy: Northgate Pharmacy"), "prescription")
    # the second occurrence has nothing after it, so it carries no value
    assert row["gt_values"] == ["NORTHGATE"]
    assert row["n_fp_spots"] == 0


# ── the paired comparison, judged by compare_runs' own vocabulary ──────────────

def _rows(n, baseline_recovered, valid_recovered):
    return [{"doc": "prescription-%04d" % i, "profile": "studio_clean",
             "doc_type": "prescription", "gt_values": ["X"], "n_gt": 1,
             "baseline_recovered": list(baseline_recovered),
             "spot_recovered": list(valid_recovered),
             "valid_recovered": list(valid_recovered)} for i in range(n)]


def test_paired_mcnemar_names_improved_only_when_the_sample_supports_it():
    row = FT.paired(_rows(6, [], ["X"]), "baseline_recovered", "valid_recovered")
    assert row["method"] == "mcnemar"
    assert row["discordant"] == (0, 6)
    assert row["p"] == pytest.approx(2.0 / 2 ** 6)
    assert row["verdict"].startswith("IMPROVED")


def test_a_two_of_three_gain_is_inconclusive():
    rows = _rows(3, [], [])
    rows[0]["valid_recovered"] = ["X"]                 # valid wins 1
    rows[1]["baseline_recovered"] = ["X"]              # baseline wins 1
    rows[2]["baseline_recovered"] = ["X"]              # baseline wins 2
    row = FT.paired(rows, "baseline_recovered", "valid_recovered")
    assert row["discordant"] == (2, 1)
    assert row["p"] == 1.0
    assert row["verdict"].startswith("INCONCLUSIVE")


def test_wilson_interval_behaves_at_the_edges():
    lo, hi = FT.wilson(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.35
    lo, hi = FT.wilson(10, 10)
    assert hi == pytest.approx(1.0) and 0.65 < lo < 1.0
