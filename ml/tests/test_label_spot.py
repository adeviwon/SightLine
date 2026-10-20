"""
Closed-vocabulary label spotting — the blind-user read path, not transcription.

A sighted reader transcribes a page; a blind user holding a phone over a
prescription needs ONE THING: the value beside a label they already know
("ACCOUNT NUMBER", "SORT CODE"). Spotting one of ~15 known labels is
approximate string matching against a CLOSED list, and that survives OCR that
is only ~70% character-accurate — which is where this repo is (the recogniser
reads our own renders at 4.4% exact). Published: a lexicon-constrained decoder
cuts word error 35-41.5% (Weinman et al., PMC3021989); keyword-proximity field
extraction is the standard product pattern.

These tests pin the behaviour that makes it work: digit-for-letter OCR
confusions, a small proportional edit budget, no match inside a longer token,
deterministic overlap resolution, and a miss returning [] rather than raising.

Nothing here needs a checkpoint, a corpus or the network: the matcher is pure
stdlib over text already in hand.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import label_spot as LS  # noqa: E402


# ── the basic contract ────────────────────────────────────────────────────────

def test_exact_hit_offsets_and_score():
    text = "Account Number: 40218877"
    got = LS.spot_labels(text)
    assert [s["label"] for s in got] == ["ACCOUNT NUMBER"], got
    s = got[0]
    assert (s["start"], s["end"]) == (0, 14), s
    assert text[s["start"]:s["end"]] == "Account Number"
    assert s["score"] == 1.0 and set(s) == {"label", "start", "end", "score"}


def test_digit_for_letter_confusions_still_spot():
    # Exactly the failure mode the recogniser produces on our own renders:
    # 0/O, 1/I, 5/S, 7/T, 6/G, 8/B. Both sides are folded before comparing.
    text = "ACC0UNT N0MBER 40218877"
    got = LS.spot_labels(text)
    assert [s["label"] for s in got] == ["ACCOUNT NUMBER"], got
    assert (got[0]["start"], got[0]["end"]) == (0, 14), got
    assert text[got[0]["start"]:got[0]["end"]] == "ACC0UNT N0MBER"
    # ACC0UNT folds free (0->O). N0MBER is N-zero-M-B-E-R: the recogniser put a
    # zero where the label has a U, which is a substitution, not a fold — so one
    # edit, not zero, and still well inside budget.
    assert got[0]["score"] == round(1 - 1 / 13, 4), got


def test_sort_code_with_confused_digit():
    text = "SORT C0DE 40-11-04"
    got = LS.spot_labels(text)
    assert [s["label"] for s in got] == ["SORT CODE"], got
    assert (got[0]["start"], got[0]["end"]) == (0, 9), got
    assert text[got[0]["start"]:got[0]["end"]] == "SORT C0DE"


def test_one_substitution_is_tolerated():
    # PATIENT -> PATENT is one dropped letter: a 70%-accurate read, not a miss.
    got = LS.spot_labels("PATENT: JANE DOE")
    assert [s["label"] for s in got] == ["PATIENT"], got
    assert got[0]["score"] > 0.75


# ── the rule that stops stupid matches ────────────────────────────────────────

def test_label_never_matches_inside_a_longer_token():
    assert LS.spot_labels("TERMINAL 4 IS CLOSED") == []
    assert LS.spot_labels("TERMS AND CONDITIONS APPLY") == []
    assert LS.spot_labels("DATED 01/10/2026") == [] or all(
        s["label"] != "DATE" for s in LS.spot_labels("DATED 01/10/2026"))


def test_ambiguous_line_returns_only_the_longer_label():
    got = LS.spot_labels("LEASE AGREEMENT")
    assert [s["label"] for s in got] == ["LEASE AGREEMENT"], got
    # ...and on a line that only says LEASE, the short label is the right answer.
    got = LS.spot_labels("Clause 3.2  Rent (LEASED TO TENANT)")
    assert "LEASE" not in [s["label"] for s in got], got


def test_two_labels_returned_with_non_overlapping_spans():
    text = "Account Number: 40218877   Sort Code: 40-11-04"
    got = LS.spot_labels(text)
    assert sorted(s["label"] for s in got) == ["ACCOUNT NUMBER", "SORT CODE"], got
    spans = sorted((s["start"], s["end"]) for s in got)
    assert spans[0][1] <= spans[1][0], spans
    assert got == sorted(got, key=lambda s: (s["start"], s["end"]))


def test_too_damaged_line_is_a_miss_not_an_exception():
    # MAKNING (WARNING) is 2 edits on 7 characters — over budget on purpose.
    assert LS.spot_labels("MAKNING MAY AUSE IONSNETE") == []
    assert LS.spot_labels("") == []
    assert LS.spot_labels("the quick brown fox jumps") == []
    assert LS.spot_labels("!!! 1234 ...") == []


def test_determinism():
    text = "Lease Agreement: Term 24 months, Rent payable to the Landlord"
    first = LS.spot_labels(text)
    for _ in range(5):
        assert LS.spot_labels(text) == first
    assert LS.spot_labels(text, 0.5) == LS.spot_labels(text, 0.5)


def test_threshold_is_a_recall_precision_knob():
    text = "PHATMALY NORTHGATE"          # PHATMALY is 2 edits off PHARMACY
    strict = LS.spot_labels(text, 0.95)
    loose = LS.spot_labels(text, 0.6)
    assert all(s["label"] != "PHARMACY" for s in strict), strict
    assert len(loose) >= len(strict)
    for s in LS.spot_labels(text, 0.0):
        assert 0.0 <= s["score"] <= 1.0


# ── lines ─────────────────────────────────────────────────────────────────────

def test_spot_in_lines_indexes_and_keeps_shapes():
    lines = [{"text": "HSBC UK BANK PLC", "box": (0, 0, 100, 20)},
             {"text": "nothing here", "box": (0, 20, 100, 40)},
             {"text": "SORT C0DE 40-11-04", "box": (0, 40, 100, 60)}]
    got = LS.spot_in_lines(lines)
    assert [(s["label"], s["line_index"]) for s in got] == [
        ("HSBC UK BANK PLC", 0), ("SORT CODE", 2)], got
    for s in got:
        assert set(s) == {"label", "start", "end", "score", "line_index"}
        assert 0 <= s["start"] < s["end"] <= len(lines[s["line_index"]]["text"])
    assert LS.spot_in_lines([]) == []


# ── the vocabulary is mined from this repo, not invented ──────────────────────

def test_vocabulary_is_grounded_in_repo_data():
    prov = LS.provenance()
    unattested = sorted(l for l, src in prov.items() if src is None)
    assert unattested == [], f"labels with no repo source: {unattested}"
    assert prov == LS.provenance()          # deterministic
    assert len(LS.VOCAB) >= 30, len(LS.VOCAB)


def test_vocabulary_covers_all_three_document_types():
    for doc_type in ("prescription", "banking", "legal"):
        labels = LS.DOC_LABELS[doc_type]
        assert len(labels) >= 10, (doc_type, labels)
        for lab in labels:
            assert lab in LS.VOCAB


def test_vocabulary_traces_to_a_named_repo_source():
    """Every label is a phrase one of capture.DOCS prints, a keyword
    evaluate.classify_doc matches on, or a field class field_lexicon names —
    i.e. mined from disk, not invented for the test."""
    prov = LS.provenance()
    assert set(prov.values()) <= {
        "capture.DOCS", "evaluate.classify_doc", "field_lexicon.PATTERNS"}, prov
    printed = sum(1 for src in prov.values() if src == "capture.DOCS")
    assert printed >= 30, (printed, "vocabulary drifted away from the documents")
