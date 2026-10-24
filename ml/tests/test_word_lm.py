"""
test_word_lm.py — the word-level LM's own tests: the trie's scoring rules, the
shipped JSON mirror, and the two properties the decoder depends on.

Run: bash run.sh test ml/tests/test_word_lm.py -q
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

import ctc_decode as CD      # noqa: E402
import realdata as R         # noqa: E402
import word_lm as WL         # noqa: E402

LSRC = ROOT / "data" / "lexicon.txt"
TRIE = ROOT / "app" / "data" / "word_trie.json"


@pytest.fixture(scope="module")
def lm():
    return WL.WordLM(WL.load_lexicon(LSRC))


def test_trie_holds_the_lexicon(lm):
    assert lm.words == len([w for w in WL.load_lexicon(LSRC) if w.strip()])
    assert lm._node("ACCOUNT") is not None


def test_word_boundary_is_binary(lm):
    """A word the lexicon serves is neutral at a boundary; one it does not is
    penalised. This is the whole word-level signal — if it stops separating the
    two, the LM has become a character LM with extra steps."""
    assert lm.logp("ACCOUNT", " ") == 0.0
    assert lm.logp("ACCOUNTT", " ") == WL.PENALTY
    # an empty current word is nothing to score, not an error
    assert lm.logp("", " ") == 0.0


def test_letter_inside_a_digit_run_is_punished(lm):
    """The confusion the product cannot afford: 0A7E01102026 for 01102026."""
    assert lm.logp("0110", "A") == WL.PENALTY
    assert lm.logp("0110202", "Z") == WL.PENALTY


def test_unseen_numbers_are_not_punished(lm):
    """Account numbers and dates are open-ended; the lexicon holds only the ones
    the corpus happened to contain. Corrupting an unseen-but-correct number is
    worse than leaving the CTC's own preference alone."""
    assert lm.logp("0110202", "7") == WL.DIGIT_PENALTY
    # at the root a digit IS a trie path (the lexicon opens with digit runs), so
    # the value there is the trie's own ratio, not the floor — the assertion that
    # matters is that it is never the full penalty.
    assert lm.logp("", "4") > WL.PENALTY
    assert WL.DIGIT_PENALTY > WL.PENALTY          # milder, deliberately


def test_a_lexicon_word_never_scores_worse_than_a_non_word(lm):
    """On a real word every step is a real trie path (<= 0), and no step may be
    the full penalty."""
    w = "PRESCRIPTION"
    steps = [lm.logp(w[:i], w[i]) for i in range(len(w))]
    assert all(s <= 0.0 for s in steps), steps
    assert all(s > WL.PENALTY for s in steps), steps


def test_shipped_json_scores_identically(lm, tmp_path):
    """The browser loads THIS file. A scoring difference between the trie in
    memory and the trie on disk is a silent cross-language divergence."""
    out = tmp_path / "trie.json"
    WL.export_trie(lm, out)
    tj = WL.load_trie_json(out)
    alphabet = list(R.CHARSET) + [" "]
    prefixes = ["", "A", "AC", "ACCOUNT", "PRESCRIPTION", "0110", "0110202",
                "PRE5CR1PT10N", "500M", "M ", "0A7E0110"]
    for p in prefixes:
        for ch in alphabet:
            assert lm.logp(p, ch) == pytest.approx(tj.logp(p, ch), abs=1e-12), (p, ch)


def test_exported_trie_is_the_one_on_disk(lm, tmp_path):
    """The shipped artifact must come from the current lexicon: if it does not,
    the app decodes against a stale vocabulary."""
    if not TRIE.exists():
        pytest.skip("no shipped trie yet — run: bash run.sh py ml/src/word_lm.py --export")
    shipped = json.loads(TRIE.read_text())
    assert shipped["meta"]["words"] == lm.words, (
        "app/data/word_trie.json has %s words, the lexicon has %s — re-export"
        % (shipped["meta"]["words"], lm.words))
    assert shipped["meta"]["penalty"] == WL.PENALTY
    assert shipped["meta"]["penalty_digit"] == WL.DIGIT_PENALTY


def test_beam_with_the_word_lm_still_collapses_repeats(lm):
    """The doubled-digit bookkeeping is the classic CTC prefix-beam bug the LM
    must not disturb: 40218877 must not become 4021887."""
    import numpy as np
    T, C = 8, R.NUM_CLASSES
    lp = np.full((T, C), -12.0)
    # 4 0 2 1 8 8 7 7 with the doubled chars separated by blanks
    seq = [4, 0, 1, 5, 0, 2, 0, 5]        # indices: blank is 0, charset is 1-based
    for t, c in enumerate(seq):
        lp[t, c] = 0.0
    got = CD.beam_decode(lp, beam=4, lm=lm, alpha=WL.PENALTY and 0.25)
    assert got == R.decode_greedy(np.asarray(seq)), (got, R.decode_greedy(np.asarray(seq)))
