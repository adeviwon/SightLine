"""
test_ctc_decode.py — pin the decoder and the A/B statistics.

`ml/src/ctc_decode.py` adds a CTC prefix beam search and a character n-gram
rescorer on top of the shipped greedy decode, and `ml/src/eval_decode_ab.py`
uses it to answer one question: does beam search (with or without the LM) read
held-out crops better than greedy? The whisker this suite stands on:

  1. THE STATISTICS MUST BE CORRECT BEFORE THE NUMBER MEANS ANYTHING.
     A Wilson interval and an exact McNemar p-value are the whole difference
     between "a few points better" and "a sample-size illusion"
     (ml/tests and the validation-gates skill). Both are closed-form, so both
     are pinned here against hand-computed values.
  2. THE LM MUST NEVER SEE THE TEST SPLIT. Building the rescorer on the split it
     is scored on is the leak the rest of this repo is organised to prevent, and
     `collect_labels()` is where it would happen. The monkeypatched test below
     fails if that call site ever asks for anything but "train".
  3. ALPHA=0 MEANS NO LM. Shallow fusion must be switchable, or "beam+LM" and
     "beam" cannot be compared honestly.

Paired design, not two independent runs: `exact_match_bits` returns a 0/1 per
crop for each variant, and `pair_counts` reduces two such vectors to the
discordant counts McNemar actually consumes (b = variant right/greedy wrong,
c = greedy right/variant wrong). Comparing two unpaired accuracies throws that
pairing away and loses most of the power.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_lm as BL        # noqa: E402
import ctc_decode as CD      # noqa: E402
import realdata as R         # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def make_lp(rows, floor=-8.0):
    """A [T, C] log-softmax matrix from per-timestep {class_index: logit} dicts.

    Unlisted classes get `floor` (a real, finite logit, not -inf, so the row is
    a legal distribution) and the row is then normalised to log-probabilities --
    the same thing the model's `log_softmax(-1)` produces, so the decoder is
    exercised on the shape it actually receives.
    """
    m = np.full((len(rows), R.NUM_CLASSES), floor, dtype=np.float64)
    for t, d in enumerate(rows):
        for k, v in d.items():
            m[t, k] = v
    return m - np.log(np.exp(m).sum(-1, keepdims=True))


def make_lm(pairs):
    """A CharLM from {(context, char): count}; contexts are 0-2 chars."""
    lm = BL.CharLM(lam=0.4)
    for (ctx, ch), n in pairs.items():
        for _ in range(n):
            lm.add(ctx, ch)
    return lm


def small_lp(rows, floor=-8.0, ncol=14):
    """A [T, ncol] log-softmax matrix over blank + '0123456789' + A,B,C.

    Narrowed from the full 38 classes so an exhaustive search over every
    alignment is cheap, while A/B/C keep their REAL class indices -- the decoder
    is exercised on the same i2c mapping it uses in production.
    """
    m = np.full((len(rows), ncol), floor, dtype=np.float64)
    for t, d in enumerate(rows):
        for k, v in d.items():
            if k < ncol:
                m[t, k] = v
    return m - np.log(np.exp(m).sum(-1, keepdims=True))


def brute_force_best(lp, blank=None):
    """The best CTC reading by EXHAUSTIVE enumeration of every alignment.

    Sums the probability of every path that collapses to each string and returns
    the argmax. Exponential, hence only usable on the toy alphabet above -- but
    it is the definition of the answer, so it is what the search is checked
    against rather than another decoder's opinion.
    """
    from itertools import product
    NEG = CD.NEG_INF
    T, C = lp.shape
    best = {}
    for path in product(range(C), repeat=T):
        s = R.decode_greedy(list(path))
        sc = 0.0
        for t, c in enumerate(path):
            sc += lp[t, c]
        best[s] = CD._lae(best.get(s, NEG), sc)
    return max(best.items(), key=lambda kv: kv[1])[0]


# ── beam search ───────────────────────────────────────────────────────────────

def test_beam_matches_greedy_when_the_model_is_peaked():
    """On a confident line the two decoders must agree, or one of them is wrong.

    A peaked matrix has one obvious reading, so beam and greedy reaching
    different strings here would mean a bug in the prefix bookkeeping rather
    than a search difference worth reporting.
    """
    a, b, c = R.C2I["A"], R.C2I["B"], R.C2I["C"]
    lp = make_lp([{a: 0.0}, {b: 0.0}, {c: 0.0}])
    assert R.decode_greedy(lp.argmax(-1)) == "ABC"
    assert CD.beam_decode(lp, beam=10) == "ABC"


def test_beam_and_greedy_agree_on_a_repeat_with_a_blank_between():
    """The CTC repeat rule: [A, blank, A] is "AA", not "A".

    Greedy already documents this (realdata's selftest). The beam search has a
    separate code path for it -- a repeat extends only from the blank-ended
    state -- and getting it wrong silently drops a doubled digit, i.e. turns
    40218877 into 4021887.
    """
    a = R.C2I["A"]
    lp = make_lp([{a: 0.0}, {R.BLANK: 0.0}, {a: 0.0}])
    assert R.decode_greedy(lp.argmax(-1)) == "AA"
    assert CD.beam_decode(lp, beam=10) == "AA"


def test_beam_on_an_all_blank_matrix_reads_nothing():
    lp = make_lp([{R.BLANK: 0.0}] * 4)
    assert CD.beam_decode(lp, beam=10) == ""


def test_beam_output_is_always_within_the_charset():
    rng = np.random.default_rng(0)
    lp = rng.normal(size=(12, R.NUM_CLASSES))
    lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
    for beam in (1, 5, 10):
        txt = CD.beam_decode(lp, beam=beam)
        assert set(txt) <= set(R.CHARSET)


def test_beam_matches_an_exhaustive_search_over_all_alignments():
    """The strong check: on a toy alphabet, enumerate EVERY alignment.

    "Best CTC reading" is defined as the string whose alignments sum to the most
    probability, so a search that disagrees with the enumeration on a matrix
    small enough to enumerate is simply wrong. This is what pins the two-state
    prefix bookkeeping (blank-ended vs character-ended) that makes repeats work
    -- hand-tracing that by hand is where a decoder bug hides.
    """
    rng = np.random.default_rng(3)
    for case in range(8):
        rows = []
        for _ in range(3):
            d = {R.BLANK: float(rng.normal(0.0, 1.0))}
            for k in range(1, 14):
                d[k] = float(rng.normal(0.0, 1.5))
            rows.append(d)
        lp = small_lp(rows)
        want = brute_force_best(lp)
        assert CD.beam_decode(lp, beam=200) == want, (case, want)


def test_a_doubled_character_survives_the_beam():
    """A repeat is only reachable through the BLANK-ended state, and the beam
    must find it. Merging the two states -- the classic prefix-beam bug --
    deletes every doubled digit, turning 40218877 into 4021887.

    Here the blank is the likely middle symbol and "AA" is the most probable
    reading of the three timesteps.
    """
    a = R.C2I["A"]
    lp = small_lp([{a: 0.0, R.BLANK: -3.0},
                   {R.BLANK: 0.0, a: -0.5},
                   {a: 0.0, R.BLANK: -3.0}])
    assert brute_force_best(lp) == "AA"
    assert R.decode_greedy(lp.argmax(-1)) == "AA"
    assert CD.beam_decode(lp, beam=10) == "AA"


# ── LM rescorer ───────────────────────────────────────────────────────────────

def test_lm_rescoring_overrides_a_weak_acoustic_preference():
    """"A" then "B" loses to "A" then "C" acoustically, and wins lexically.

    The acoustic row for the second character prefers C by a hair; the LM has
    seen "AB" ten times as often as "AC". With no LM the beam must follow the
    acoustics; with the LM on it must not. That is the entire claim shallow
    fusion makes, and it is checked rather than assumed.

    The LM also has to know "A" can START a string at all (the `("", "A")`
    counts): a rescorer that has only ever seen "A" in the middle would veto the
    first character and answer with nothing, which is the failure mode high
    `alpha` produces on a tiny LM.
    """
    a, b, c = R.C2I["A"], R.C2I["B"], R.C2I["C"]
    lp = make_lp([{a: 0.0, R.BLANK: -6.0}, {c: -0.1, b: -0.5, R.BLANK: -3.0}])
    assert CD.beam_decode(lp, beam=24) == "AC"
    lm = make_lm({("", "A"): 50, ("A", "B"): 100, ("A", "C"): 10})
    assert CD.beam_decode(lp, beam=24, lm=lm, alpha=1.0) == "AB"


def test_alpha_zero_ignores_the_lm_entirely():
    a, b, c = R.C2I["A"], R.C2I["B"], R.C2I["C"]
    lp = make_lp([{a: 0.0, R.BLANK: -6.0}, {c: -0.1, b: -0.5, R.BLANK: -3.0}])
    lm = make_lm({("", "A"): 50, ("A", "B"): 100})
    assert CD.beam_decode(lp, beam=24, lm=lm, alpha=0.0) == CD.beam_decode(lp, beam=24)


def test_lm_scores_a_seen_continuation_above_an_unseen_one():
    lm = make_lm({("T", "H"): 100, ("H", "E"): 90})
    assert lm.logp("TH", "E") > lm.logp("TH", "Z")


def test_lm_is_smoothed_and_never_returns_infinity():
    """An unseen continuation must be unlikely, not impossible.

    Returning -inf would make the beam discard a reading the acoustics are
    confident about, which is the classic way an n-gram rescorer turns a good
    decoder into a worse one.
    """
    lm = make_lm({("T", "H"): 10})
    for ch in R.CHARSET:
        s = lm.logp("TH", ch)
        assert math.isfinite(s) and s <= 0.0


# ── the LM must not see the test split ────────────────────────────────────────

def test_lm_is_built_from_the_train_split_only(monkeypatch):
    """`collect_labels()` may only ever load the TRAIN split.

    Scoring a rescorer on the split it was built from reports a number that is
    partly memorisation. The repo enforces the same rule for the recogniser via
    page-level splits; this is the equivalent gate for the LM.
    """
    seen = []
    real = BL.RC.load

    def spy(*a, **k):
        seen.append(k.get("split"))
        return real(*a, **k)

    monkeypatch.setattr(BL.RC, "load", spy)
    labels = BL.collect_labels()
    assert seen, "collect_labels() never loaded the corpus"
    assert set(seen) == {"train"}, f"the LM loaded splits {seen}"
    assert labels and all(isinstance(s, str) for s in labels)


def test_lm_labels_are_charset_clean():
    assert BL.clean_label("Prescription: 500mg") == "PRESCRIPTION 500MG"
    assert BL.clean_label("40-11-04") == "401104"


# ── statistics ────────────────────────────────────────────────────────────────

def test_wilson_interval_brackets_the_mean_and_clamps():
    lo, hi = CD.wilson(50, 100)
    assert lo < 0.5 < hi
    assert abs(lo - 0.4038) < 0.01 and abs(hi - 0.5962) < 0.01
    # Degenerate ends: an interval on 0/n must still start at 0, and n/n above 0.
    assert CD.wilson(0, 40)[0] == 0.0 and CD.wilson(0, 40)[1] > 0.0
    assert CD.wilson(40, 40)[1] == 1.0 and CD.wilson(40, 40)[0] < 1.0
    assert CD.wilson(0, 0) == (0.0, 0.0)


def test_mcnemar_exact_is_symmetric_and_uses_only_discordant_pairs():
    assert CD.mcnemar_exact(0, 0) == 1.0
    assert CD.mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert CD.mcnemar_exact(0, 7) == CD.mcnemar_exact(7, 0)
    # Concordant pairs are invisible: only the discordant counts enter.
    assert CD.mcnemar_exact(8, 8) == 1.0


def test_mcnemar_exact_matches_the_binomial_tail_by_hand():
    """n=10 discordant, k=2 on the light side: 2 * sum_{i<=2} C(10,i) / 2^10."""
    want = 2 * sum(math.comb(10, i) for i in range(3)) / 2 ** 10
    assert CD.mcnemar_exact(2, 8) == pytest.approx(want)
    assert CD.mcnemar_exact(2, 8) > 0.05      # not significant: noise, not a win


def test_pair_counts_reads_the_discordant_cells_the_right_way_round():
    """b = variant right & greedy wrong; c = greedy right & variant wrong.

    Swapping them inverts the sign of the effect, and McNemar would then report
    a p-value for the opposite hypothesis.
    """
    greedy = [1, 1, 0, 0, 0]
    variant = [1, 0, 1, 0, 0]
    pc = CD.pair_counts(greedy, variant)
    assert pc == {"both_right": 1, "variant_only": 1, "greedy_only": 1, "both_wrong": 2}


def test_exact_match_bits_uses_the_scoring_normalisation():
    """A character the model cannot emit must not count as a mismatch.

    Scoring goes through realdata.norm_label, so "TOTAL 9.00" and "TOTAL 9,OO"
    agree -- the same rule eval_render_corpus uses, and the reason a variant's
    exact-match here is comparable to the number in the docs.
    """
    bits = CD.exact_match_bits(["TOTAL 9.00", "WRONG"], ["TOTAL 9.OO", "RIGHT"])
    assert bits == [1, 0]


def test_cer_aggregation_counts_characters_not_crops():
    """CER here is char-weighted (total edits / total reference chars)."""
    preds = ["ABCX", "Z"]
    refs = ["ABCD", "Z"]
    assert CD.cer_aggregate(preds, refs) == pytest.approx(1.0 / 5.0)


# ── the checkpoint torn-read guard ───────────────────────────────────────────

def test_a_checkpoint_that_changes_while_loading_is_fatal(tmp_path, monkeypatch):
    """Scoring a half-written checkpoint must fail, not report a fake accuracy.

    Observed for real while this was being written: the recogniser is retrained
    in place, and an A/B run that loaded mid-save read a partially written
    crnn.pt. torch raised nothing, the weights were garbage, and the harness
    reported 0/300 exact at CER 0.976 where the real answer was 11/300 at CER
    0.590 -- a decoder failure that never happened. `load_model` stats the file
    either side of torch.load and must raise when it moved.

    torch.load is monkeypatched to simulate the save landing mid-load, so the
    test needs no real checkpoint and cannot be defeated by one being absent.
    """
    import torch
    import eval_decode_ab as EA
    target = tmp_path / "crnn.pt"
    target.write_bytes(b"not a real checkpoint")

    def touching_load(path, **kw):
        # The save lands mid-load: the returned weights are irrelevant because
        # the guard must fire before they are used, so return a plain dict.
        target.write_bytes(b"not a real checkpoint, but longer now")
        return {"num_classes": R.NUM_CLASSES, "model": {}}

    monkeypatch.setattr(torch, "load", touching_load)
    with pytest.raises(RuntimeError, match="changed while it was being loaded"):
        EA.load_model(str(target))
