"""
Tests for the handheld line aligner.

The aligner is the difference between "the model misread 3 lines" and "the
model misread everything because one spurious band shifted the whole page".
Both readings look identical in the output, which is why these cases are
pinned down here.

Run:  bash run.sh test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from eval_ocr import align_lines  # noqa: E402


def pair(pairs):
    """[(got, want)] -> [(want, got)] with None preserved, for easy asserts."""
    return [(w, g) for g, w in pairs]


def scored(pairs):
    """Only the pairs where both sides exist -- what actually gets a CER."""
    return [(g, w) for g, w in pairs if g is not None and w is not None]


# ── the cases that motivated it ─────────────────────────────────────────────

def test_no_spurious_band_is_plainly_positional():
    got = ["GREEN FIELD", "FALAFEL WRAP", "ORDER 6993"]
    want = ["GREEN FIELD", "FALAFEL WRAP", "ORDER 6993"]
    assert scored(align_lines(got, want)) == list(zip(got, want))


def test_one_inserted_line_does_not_shift_the_rest():
    """
    A shadow edge or a plastic crease produces a phantom band. Pure
    positional pairing would compare phantom->line0, line0->line1,
    line1->line2 and score every line wrong. The drift tracking must absorb
    it so line0, line1 and line2 still pair correctly.
    """
    got = ["SHADOW EDGE", "GREEN FIELD", "FALAFEL WRAP", "ORDER 6993"]
    want = ["GREEN FIELD", "FALAFEL WRAP", "ORDER 6993"]
    s = scored(align_lines(got, want))
    assert s == [("GREEN FIELD", "GREEN FIELD"),
                 ("FALAFEL WRAP", "FALAFEL WRAP"),
                 ("ORDER 6993", "ORDER 6993")], \
        f"drift not absorbed: {s}"
    # The phantom is reported as spurious, not silently swallowed.
    assert sum(1 for g, w in align_lines(got, want) if w is None) == 1


def test_one_missed_line_does_not_shift_the_rest():
    got = ["GREEN FIELD", "ORDER 6993"]           # line 1 never detected
    want = ["GREEN FIELD", "FALAFEL WRAP", "ORDER 6993"]
    s = scored(align_lines(got, want))
    # line 0 and line 2 must still pair with themselves.
    assert ("GREEN FIELD", "GREEN FIELD") in s
    assert ("ORDER 6993", "ORDER 6993") in s
    assert sum(1 for g, w in align_lines(got, want) if g is None) == 1


# ── difflib's failure mode must not come back ───────────────────────────────

def test_noisy_recognition_still_pairs_everything():
    """
    difflib.SequenceMatcher on line TEXT returns ZERO pairs when nothing
    matches exactly, which silently collapsed the whole evaluation to
    recall 0% / CER 0%. The aligner must always pair positionally when no
    exact match exists, so CER stays meaningful.
    """
    got = ["NODS DH", "0E5AE55", "GOSES BL8TAN"]      # all wrong
    want = ["GREEN FIELD", "385 RT 9 W", "TOTAL: $5.35"]
    s = scored(align_lines(got, want))
    assert len(s) == 3, f"expected 3 pairs, got {len(s)} -- do not use difflib"
    assert [w for _, w in s] == want


def test_partially_correct_line_still_pairs():
    got = ["GREEN FIELD", "T USUT", "ORDER  6993"]
    want = ["GREEN FIELD", "TOTAL: $5.35", "ORDER 6993"]
    s = scored(align_lines(got, want))
    assert len(s) == 3
    assert s[0] == ("GREEN FIELD", "GREEN FIELD")
    assert s[2] == ("ORDER  6993", "ORDER 6993")


# ── edges ───────────────────────────────────────────────────────────────────

def test_no_want_everything_is_spurious():
    p = align_lines(["A", "B"], [])
    assert all(w is None for _, w in p)
    assert [g for g, _ in p] == ["A", "B"]


def test_no_got_everything_is_missed():
    p = align_lines([], ["A", "B"])
    assert all(g is None for g, _ in p)
    assert [w for _, w in p] == ["A", "B"]


def test_both_empty():
    assert align_lines([], []) == []


def test_more_got_than_want_keeps_the_extras():
    p = align_lines(["A", "B", "C", "D"], ["A", "B"])
    assert sum(1 for _, w in p if w is None) == 2


def test_alignment_is_monotonic():
    """
    Text order must never be permuted. A line read out of order is read out
    of order to the user, so a metric that allowed reordering would hide a
    real defect.
    """
    got = ["ONE", "TWO", "THREE", "FOUR", "FIVE"]
    want = ["ONE", "TWO", "THREE", "FOUR", "FIVE"]
    s = scored(align_lines(got, want))
    idx = [got.index(g) for g, _ in s]
    assert idx == sorted(idx), f"alignment reordered text: {idx}"


def test_exact_match_beats_the_drift_prior():
    """
    An exact match further from the current drift must still win: this is what
    lets a genuinely shifted region recover.
    """
    got = ["JUNK", "JUNK", "JUNK", "TOTAL 5.35"]
    want = ["TOTAL 5.35"]
    p = align_lines(got, want)
    s = scored(p)
    assert s and s[0][0] == "TOTAL 5.35", \
        f"exact match lost to the drift prior: {s}"


def test_drift_prior_breaks_ties_toward_position():
    """With nothing matching exactly, prefer the positionally expected band."""
    got = ["AAA", "BBB"]
    want = ["XXX", "YYY"]
    s = scored(align_lines(got, want))
    assert [g for g, _ in s] == ["AAA", "BBB"]


@pytest.mark.parametrize("n_got,n_want", [(1, 1), (2, 2), (3, 3), (5, 5), (4, 6), (6, 4)])
def test_pair_count_never_exceeds_either_side(n_got, n_want):
    p = align_lines([f"g{i}" for i in range(n_got)],
                    [f"w{i}" for i in range(n_want)])
    s = scored(p)
    assert len(s) <= min(n_got, n_want)
    assert len(p) == n_got + n_want - len(s)   # every item accounted for


def test_same_length_hint_does_not_beat_an_exact_match():
    got = ["WRONGWRONG", "CORRECT!!!"]
    want = ["CORRECT!!!"]
    s = scored(align_lines(got, want))
    assert s[0][0] == "CORRECT!!!"


def test_duplicate_lines_are_not_collapsed():
    """
    Recovering a line's sort position with list.index() finds the FIRST
    occurrence, so every copy of a repeated line collapsed onto one key and
    the pairs came back misordered. Receipts genuinely repeat lines -- a
    second TOTAL, a repeated address -- so this is not hypothetical.
    """
    want = ["ITEM", "TOTAL", "TOTAL", "ITEM"]
    got = ["ITEM", "TOTAL", "TOTAL", "ITEM"]
    p = align_lines(got, want)
    assert [w for _, w in p if w is not None] == want, \
        f"duplicates collapsed or reordered: {[w for _, w in p]}"
    # Each duplicate keeps its own position rather than all landing on one.
    assert len(p) == 4


def test_duplicate_lines_with_one_missed():
    want = ["TOTAL", "TOTAL", "TOTAL"]
    got = ["TOTAL", "TOTAL"]
    p = align_lines(got, want)
    s = scored(p)
    assert [g for g, _ in s] == ["TOTAL", "TOTAL"]
    assert sum(1 for g, _ in p if g is None) == 1


def test_every_input_line_appears_exactly_once():
    """No line may be dropped or double-counted by the aligner."""
    cases = [
        (["A", "B", "C"], ["A", "C"]),
        (["A", "B", "C"], ["B"]),
        (["A", "A", "A"], ["A", "A"]),
        (["X", "A", "Y", "B"], ["A", "B", "C"]),
        ([], ["A", "B"]),
        (["A", "B"], []),
    ]
    for got, want in cases:
        p = align_lines(got, want)
        gs = [g for g, _ in p if g is not None]
        ws = [w for _, w in p if w is not None]
        assert len(gs) == len(set(range(len(gs)))), cases
        # every got line accounted for exactly once (by identity, so
        # duplicates are distinguishable)
        assert sorted(id(x) for x in gs) == sorted(id(x) for x in got), \
            f"got lines lost or duplicated: {got} -> {p}"
        assert sorted(id(x) for x in ws) == sorted(id(x) for x in want), \
            f"want lines lost or duplicated: {want} -> {p}"