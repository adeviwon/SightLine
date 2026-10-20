"""
Tests for eval_field_recovery.py — the end-to-end field-spotting measurement.

All synthetic: no checkpoint, no corpus, no model. The tests pin the SCORING
RULES the headline number rests on, because a wrong denominator here is a wrong
ship/kill decision:

  * a perfect read recovers the field;
  * a digit-confused label read still spots (the spotter folds 0->O);
  * a wrong-length value is REFUSED, never spoken;
  * precision-of-spoken counts only correct spoken values;
  * an empty page is expected-but-not-recovered, never skipped;
  * the threshold sweep reproduces label_spot.spot_in_lines at 0.75;
  * a small apparent gain is INCONCLUSIVE, and only a real one is IMPROVED.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_field_recovery as FR  # noqa: E402
import label_spot as LS           # noqa: E402


def L(*texts):
    """Line dicts in page order, with the geometry the harness hands the modules."""
    return [{"text": t, "box": (0, 0, 300, 32)} for t in texts]


# ── spotting: digit-for-letter tolerance is the whole point ───────────────────

def test_spot_lines_reproduces_spot_in_lines_at_the_default_threshold():
    """The sweep goes through spot_labels(threshold=...) because spot_in_lines
    is frozen WITHOUT a threshold; at 0.75 the two must be identical."""
    lines = L("HSBC UK BANK PLC", "nothing here", "SORT C0DE 40-11-04")
    assert FR.spot_lines(lines, 0.75) == LS.spot_in_lines(lines)


def test_digit_confused_label_still_spots():
    lines = L("ACC0UNT N0MBER 40218877")
    spots = FR.spot_lines(lines, 0.75)
    assert [s["label"] for s in spots] == ["ACCOUNT NUMBER"], spots
    assert spots[0]["line_index"] == 0


def test_strict_threshold_drops_a_3_edit_label_the_loose_one_takes():
    """threshold=0.833 is <=2 edits per 13 chars; 0.75 is <=3. Digit folding is
    free (0->O), real edits are not."""
    assert [s["label"] for s in FR.spot_lines(L("ACCOONT NVMBXR 40218877"), 0.75)]
    assert FR.spot_lines(L("ACCOONT NVMBXR 40218877"), 0.833) == []


# ── a perfect read recovers every expected field ─────────────────────────────

def test_perfect_read_recovers_every_field():
    row = FR.score_page(L("Account Number: 40218877", "Sort Code: 40-11-04"),
                        "banking")
    assert row["expected"] == ["40218877", "40-11-04"]
    assert sorted(row["recovered"]) == ["40-11-04", "40218877"]
    assert row["spoken"] == 2 and row["spoken_ok"] == 2
    assert row["refused"] == 0


def test_digit_confused_read_still_recovers_the_account():
    row = FR.score_page(L("ACC0UNT N0MBER 40218877"), "banking")
    assert row["recovered"] == ["40218877"]
    assert row["spoken_ok"] == 1


def test_masked_value_counts_when_it_is_a_suffix_of_the_expected_one():
    """The app is allowed to mask a long account number; '**** 8877' is a
    product decision, not a miss (bench_hackathon._recovered, one definition)."""
    row = FR.score_page(L("Account Number: **** 8877"), "banking")
    assert row["recovered"] == ["40218877"]
    assert row["spoken_ok"] == 1


# ── validation is the safety half ────────────────────────────────────────────

def test_wrong_length_value_is_refused_and_not_spoken():
    row = FR.score_page(L("Sort Code: 40-11-0", "Account Number: 40218877"),
                        "banking")
    assert row["refused"] == 1, row
    assert row["refusal_reasons"]["6 digits required, got 5"] == 1
    assert row["spoken"] == 1                      # only the account is spoken
    assert row["recovered"] == ["40218877"]
    assert row["baseline"] == ["40218877"]         # 40-11-0 is not ground truth


def test_empty_page_is_expected_but_not_recovered_never_skipped():
    row = FR.score_page([], "prescription")
    assert row["expected"] == ["500mg", "400mg"]
    assert row["recovered"] == [] and row["spoken"] == 0
    assert row["n_spots"] == 0 and row["refused"] == 0


# ── precision-of-spoken: the number that decides whether it is safe ──────────

def test_precision_of_spoken_counts_only_correct_spoken_values():
    rows = [
        FR.score_page(L("Account Number: 40218877", "Sort Code: 12-34-56"),
                      "banking"),          # 1 right, 1 wrong (wrong digits)
        FR.score_page(L("Account Number: 40218877"), "banking"),  # 1 right
    ]
    assert [r["spoken"] for r in rows] == [2, 1]
    assert [r["spoken_ok"] for r in rows] == [1, 1]
    k, n = FR.precision_of_spoken(rows)
    assert (k, n) == (2, 3)


def test_a_refused_value_never_enters_the_spoken_denominator():
    rows = [FR.score_page(L("Sort Code: 40-11-0"), "banking")]
    assert FR.precision_of_spoken(rows) == (0, 0)


# ── the paired comparison and the honesty rule ───────────────────────────────

def _row(doc, profile, doc_type, expected, recovered, baseline):
    return {"doc": doc, "profile": profile, "doc_type": doc_type,
            "expected": expected, "recovered": recovered, "baseline": baseline,
            "spoken": len(recovered), "spoken_ok": len(recovered), "refused": 0,
            "refusal_reasons": {}, "spots": len(recovered), "text": ""}


def test_paired_comparison_reports_improved_only_when_the_sample_supports_it():
    """Spotting fixes 8 baseline failures and loses none -> p=0.0078, IMPROVED."""
    rows = [_row("banking-%04d" % i, "studio_clean", "banking",
                 ["40218877", "40-11-04"],
                 ["40218877", "40-11-04"], []) for i in range(4)]
    c = FR.paired_vs_baseline(rows)
    assert c["method"] == "mcnemar"
    assert (c["b"], c["c"]) == (0, 8), c
    assert c["p"] == pytest.approx(2.0 / 2 ** 8)
    assert c["verdict"].startswith("IMPROVED")
    assert c["k_new"] == 8 and c["k_old"] == 0


def test_a_two_of_three_gain_is_inconclusive_not_a_win():
    """The brief's own rule: a 3-point gain over a dozen items is noise."""
    rows = [_row("banking-%04d" % i, "studio_clean", "banking", ["a"], [], [])
            for i in range(3)]
    rows[0]["recovered"] = ["a"]                   # spotting wins 1, loses 0
    rows[1]["baseline"] = ["a"]                    # baseline wins 1, spotting 0
    rows[2]["baseline"] = ["a"]                    # baseline wins 2, spotting 1
    c = FR.paired_vs_baseline(rows)
    assert (c["b"], c["c"]) == (2, 1), c
    assert c["p"] == 1.0
    assert c["verdict"].startswith("INCONCLUSIVE")


def test_both_paths_answer_the_same_question():
    """recovered and baseline must MEAN the same thing or the A/B is rigged.

    This test used to assert the opposite — that a page whose dosage had been
    read correctly and spoken as "AMOXICILLIN 500MG CAPSULES" scored a MISS,
    because the extracted value is not byte-identical to the ground-truth token
    '500mg', while the whole-page regex scored a HIT by substring search. On the
    real corpus that asymmetry reported 0/300 for the extractor against 214/300
    for the regex baseline, on pages the extractor had transcribed perfectly.
    Recovery is now: the ground-truth field is contained in something we spoke.
    """
    row = FR.score_page(L("Rx  Amoxicillin 500mg capsules"), "prescription")
    assert row["baseline"] == ["500mg"]            # keyword presence anywhere
    assert row["recovered"] == ["500mg"]           # and the spoken dose line counts
    # NOT a blanket pass: a page with nothing spoken recovers nothing.
    miss = FR.score_page(L("PRESCRIPTION"), "prescription")
    assert miss["recovered"] == [] and miss["baseline"] == []


# ── Wilson intervals ────────────────────────────────────────────────────────

def test_wilson_interval_is_reported_and_behaves_at_the_edges():
    from compare_runs import wilson
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.35
    lo, hi = wilson(10, 10)
    assert hi == pytest.approx(1.0) and 0.65 < lo < 1.0
