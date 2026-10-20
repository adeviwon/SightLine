"""
test_constrained_decode.py — pin the alphabet-constrained CTC decode.

`ml/src/constrained_decode.py` restricts a CTC decode to the symbols a field can
plausibly contain, by masking the LOGITS before the search rather than filtering
the output afterwards. The claims this suite holds it to:

  1. THE CLAIM ITSELF. On a logits matrix whose unconstrained argmax picks a
     LETTER, the digits-only decode picks the right DIGIT. That is the whole
     reason the module exists (PRESCRIPTION -> PRE5CR1P710N).
  2. THE MASK CANNOT INVENT. A constrained decode can only ever emit symbols
     from `allowed` (plus blank), whatever the logits say — including with a
     beam, where a disallowed symbol must never enter a hypothesis.
  3. REPEATS STILL SURVIVE. Masking must not disturb CTC's collapse: a doubled
     digit ("00") has to come out as "00", not "0". That is the classic bug
     (ctc_decode.py's header records it) and a mask applied in the wrong place
     reintroduces it.
  4. AN INFEASIBLE MASK RAISES. A constraint that cannot be satisfied is a
     programming error, and returning "" for it would look like a weak model.
  5. UNCONSTRAINED == SHIPPED. With the full charset allowed, this decoder must
     reproduce `realdata.decode_greedy(argmax)` exactly, or the A/B is not
     paired (two different decode rules would be compared, not two masks).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ctc_decode as CD          # noqa: E402
import realdata as R             # noqa: E402
import constrained_decode as CN  # noqa: E402

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def scores_for(chars, T=None, base=-9.0, peak=6.0, second=""):
    """[T, C] logit matrix: chars[t] wins step t, second[t] sits one below it.

    Everything else — including blank — sits at `base`, so when the winner is
    masked away there is a genuine runner-up to fall to rather than a tie with
    blank, and no blank appears mid-sequence by accident. `chars[t] is None`
    puts the peak on blank (an explicit blank step).
    """
    T = T or len(chars)
    a = np.full((T, R.NUM_CLASSES), float(base))
    for t, ch in enumerate(chars):
        a[t, R.BLANK if ch is None else R.C2I[ch]] = peak
    for t, ch in enumerate(second):
        if t < T and ch in R.C2I and ch != chars[t]:
            a[t, R.C2I[ch]] = peak - 1.0
    return a


def unconstrained(logits):
    return R.decode_greedy(logits.argmax(-1))


# ── 1. the claim ──────────────────────────────────────────────────────────────

def test_letter_argmax_is_replaced_by_the_digit_the_mask_allows():
    """PRESCRIPTION's failure mode, reduced to four timesteps: O I S T -> 0 1 5 7.

    The unconstrained argmax picks the LETTER at every step (that is what the
    shipped CRNN does on DATE 01102026 -> 0A7E01102026), so digits-only has to
    pick the digit that is second-best, not the best.
    """
    logits = scores_for("OIST", T=4, second="0157")
    # The letters win every step -- that is what the shipped CRNN does on
    # DATE 01102026 -> 0A7E01102026 -- so the digits are the runner-up.
    for t, d in enumerate("0157"):
        assert logits[t, R.C2I[d]] < logits[t, R.C2I["OIST"[t]]]

    assert unconstrained(logits) == "OIST"
    text, _ = CN.decode_constrained(logits, R.CHARSET, DIGITS)
    assert text == "0157"


def test_a_realistic_misread_line_decodes_to_the_true_number():
    """The product case: a date/number whose letters are digit-shaped.

    The letters win at every timestep (O for 0, I for 1, T for 7, S for 5), so
    the unconstrained read is not a number at all; digits-only falls to the
    runner-up and recovers the field. The truth has no adjacent repeats, because
    a one-timestep-per-character matrix cannot express one and this test is
    about the mask, not about blank bookkeeping (that is the doubled-digit test).
    """
    ref = "0107152"
    wrong = "OIOTISZ"
    logits = scores_for(wrong, second=ref)
    assert unconstrained(logits) == wrong
    text, _ = CN.decode_constrained(logits, R.CHARSET, DIGITS)
    assert text == ref


# ── 2. the mask cannot invent ─────────────────────────────────────────────────

@pytest.mark.parametrize("beam", [1, 5, 20])
def test_decode_emits_only_allowed_symbols_even_from_hostile_logits(beam):
    rng = np.random.default_rng(0)
    logits = rng.uniform(-5, 40, size=(12, R.NUM_CLASSES))
    text, _ = CN.decode_constrained(logits, R.CHARSET, DIGITS, beam=beam)
    assert set(text) <= set(DIGITS)
    assert text                          # random logits must produce SOMETHING


def test_mask_sets_disallowed_classes_to_negative_infinity():
    """Exactness, not post-hoc filtering: the disallowed columns are -inf, so no
    search can ever reach them however large their logit was."""
    logits = scores_for("OIST", T=4)
    logits[:, R.C2I["O"]] = 1e9          # an absurd letter logit
    masked = CN.mask_logits(logits, R.CHARSET, DIGITS)
    assert np.isneginf(masked[:, R.C2I["O"]]).all()
    assert np.isneginf(masked[:, R.C2I["A"]]).all()
    assert np.isfinite(masked[:, R.C2I["0"]]).all()
    assert np.isfinite(masked[:, R.BLANK]).all()   # blank is never masked


def test_beam_keeps_the_constraint_at_every_timestep():
    """A letter that wins every timestep cannot leak into a beam hypothesis."""
    logits = scores_for("OSSTO", T=5, second="01577")
    logits[:, R.C2I[" "]] = 100.0        # space too: not allowed unless asked for
    text, _ = CN.decode_constrained(logits, R.CHARSET, DIGITS, beam=10)
    assert text
    assert set(text) <= set(DIGITS)
    assert " " not in text


def test_space_is_allowed_only_when_asked_for():
    logits = scores_for("0 1")
    assert CN.decode_constrained(logits, R.CHARSET, DIGITS)[0] == "01"
    assert CN.decode_constrained(logits, R.CHARSET, DIGITS,
                                 space=True)[0] == "0 1"


# ── 3. repeats and blanks ─────────────────────────────────────────────────────

def test_blank_still_separates_two_identical_digits():
    """"00" needs a blank between the zeros; without one CTC merges them."""
    merged = scores_for("00", T=2)
    assert CN.decode_constrained(merged, R.CHARSET, DIGITS)[0] == "0"
    separated = scores_for(["0", None, "0"], T=3)
    assert CN.decode_constrained(separated, R.CHARSET, DIGITS)[0] == "00"


def test_doubled_digits_survive_in_the_beam_path_too():
    """The classic prefix-beam bug deletes doubled characters; masking must not
    reintroduce it. Measured on a 40218877-shaped label with two doubled pairs."""
    ref = "40218877"
    logits = np.full((len(ref) * 2, R.NUM_CLASSES), 0.0)
    t = 0
    for c in ref:
        logits[t, R.C2I[c]] = 6.0
        t += 1
        logits[t, R.BLANK] = 4.0          # a blank between every pair of chars
        t += 1
    text, _ = CN.decode_constrained(logits, R.CHARSET, DIGITS, beam=5)
    assert text == ref                     # 88 and 77 both survive
    assert unconstrained(logits) == text   # both decoders agree on the collapse


def test_trailing_blanks_emit_nothing():
    logits = scores_for(["0", None, None, None], T=4)
    assert CN.decode_constrained(logits, R.CHARSET, DIGITS)[0] == "0"


# ── 4. infeasible masks raise ─────────────────────────────────────────────────

def test_infeasible_mask_raises_rather_than_returning_junk():
    logits = scores_for("A0B", T=3)
    with pytest.raises(ValueError):
        CN.decode_constrained(logits, DIGITS, LETTERS)   # charset has no letters
    with pytest.raises(ValueError):
        CN.decode_constrained(logits, R.CHARSET, "")     # nothing allowed
    with pytest.raises(ValueError):
        CN.decode_constrained(np.zeros((3, 7)), R.CHARSET, DIGITS)  # wrong C


def test_allowed_symbols_absent_from_the_charset_are_ignored_not_fatal():
    """'.' and '-' cannot be emitted by a charset that has no such class; the
    digits the mask also names are still decodable."""
    logits = scores_for("1A2", second="1.2")
    text, _ = CN.decode_constrained(logits, R.CHARSET, CN.allowed_for_kind("money"))
    assert text == "12"


# ── 5. unconstrained == shipped decode, so the A/B is paired ──────────────────

def test_full_charset_matches_the_shipped_greedy_decode():
    rng = np.random.default_rng(7)
    logits = rng.uniform(-3, 3, size=(40, R.NUM_CLASSES))
    text, score = CN.decode_constrained(logits, R.CHARSET, R.CHARSET)
    assert text == unconstrained(logits)
    assert np.isfinite(score)


def test_probabilities_are_accepted_as_well_as_logits():
    logits = scores_for("OIST", T=4, second="0157")
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    from_probs, _ = CN.decode_constrained(probs, R.CHARSET, DIGITS)
    from_logits, _ = CN.decode_constrained(logits, R.CHARSET, DIGITS)
    assert from_probs == from_logits == "0157"


def test_greedy_handles_a_different_charset():
    """The charset is an argument, not global state: a 2-symbol alphabet works."""
    logits = np.zeros((3, 3))
    logits[0, 2] = 5.0            # 'b'
    logits[1, 1] = 5.0            # 'a'
    logits[2, 1] = 5.0            # 'a' again -> merges with the previous one
    text, _ = CN.decode_constrained(logits, "ab", "ab")
    assert text == "ba"


# ── allowed_for_kind ──────────────────────────────────────────────────────────

def test_allowed_for_kind_covers_the_numeric_fields_the_product_reads():
    assert set(CN.allowed_for_kind("sort_code")) == set(DIGITS)
    assert set(CN.allowed_for_kind("account")) == set(DIGITS)
    assert set(CN.allowed_for_kind("acct")) == set(DIGITS)
    assert set(CN.allowed_for_kind("date")) == set(DIGITS + "-/")
    assert set(CN.allowed_for_kind("money")) >= set(DIGITS + ".,-+")
    assert set(CN.allowed_for_kind("dosage")) == set(DIGITS + ".")
    assert CN.allowed_for_kind("Dosage") == CN.allowed_for_kind("dosage")


def test_unknown_field_kind_raises():
    with pytest.raises(ValueError):
        CN.allowed_for_kind("bank_holiday")


def test_every_kind_is_a_subset_of_the_model_charset():
    """A kind that names a symbol the model cannot emit is a silent no-op."""
    for kind in ("sort_code", "account", "date", "money", "dosage"):
        allowed = set(CN.allowed_for_kind(kind))
        assert allowed & set(R.CHARSET), kind
        assert set(CN.decode_constrained(
            scores_for("OIST", T=4), R.CHARSET, allowed)[0]) <= allowed


# ── the statistics this module is scored with ─────────────────────────────────

def test_wilson_and_mcnemar_are_the_repo_implementation():
    """Reused, not reimplemented: the A/B must be scorable the same way as every
    other experiment in the repo (eval_decode_ab.py)."""
    lo, hi = CD.wilson(5, 100)
    assert 0.0 <= lo < 0.05 < hi <= 1.0
    assert CD.mcnemar_exact(0, 0) == 1.0
    assert CD.mcnemar_exact(10, 0) < 0.01
