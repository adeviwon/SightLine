"""
ctc_decode.py — decode the CRNN's CTC output better than argmax, and decide
whether the improvement is real.

WHAT IS HERE
------------
* `beam_search` / `beam_decode` — CTC PREFIX beam search (Hannun et al. 2014),
  with optional character n-gram shallow fusion (`lm`, `alpha`).
* `lev`, `wilson`, `mcnemar_exact`, `pair_counts`, `exact_match_bits`,
  `cer_aggregate` — the scoring and the statistics that decide whether a
  difference between two decoders is evidence or noise.

WHY PREFIX BEAM AND NOT A TOKEN BEAM
------------------------------------
Greedy decoding (`realdata.decode_greedy`) takes the argmax per timestep, which
throws away the rest of the distribution before the CTC collapse. On a blank-position
ambiguity -- the model is 51% blank and 49% a real character at one timestep -- that
is a coin flip, and a wrong blank merges two readable symbols into one. The fix is
to search over the SET of collapsed prefixes, summing the probability of every
alignment that produces each one, which is exactly the prefix beam.

The bookkeeping that matters and is easy to get wrong: a prefix carries TWO
scores, one for alignments ending in a blank and one for alignments ending in the
last character. A repeat ("AA") is only reachable from the blank-ended score;
repeats arriving from the character-ended score collapse into the SAME prefix, not
a new one. Merging those two states is the classic bug, and it silently deletes
doubled digits -- turning 40218877 into 4021887.

SHALLOW FUSION, AND WHY THE LM IS SCORED ONLY ON EMIT
-----------------------------------------------------
`alpha` weights the LM's log-probability of the newly emitted character given the
prefix (`+ alpha * log P(c | prefix[-2:])`). It is added ONLY when a character is
emitted -- never on a blank, never on a repeat-collapse -- because the LM models
text, and a blank emits no text. Scoring blanks would just scale every timestep by
a constant and, worse, make long prefixes look lexically plausible for staying
alive.

`alpha = 0` disables the LM completely, so "beam" and "beam + LM" are the same
code path with one number changed and can be compared honestly.

NO LENGTH NORMALISATION
-----------------------
Deliberate. A per-character length bonus raises the score of long prefixes and is
a known way to make a decoder invent characters; every field this product extracts
(`500mg`, `40218877`, `01102026`) is short and numeric, where an invented extra
digit is a false positive the extractor will happily report. If a length term is
ever wanted, add it next to `alpha` as an explicit knob and measure it -- do not
bury it in the search.

WHY THE STATISTICS LIVE IN A DECODER MODULE
-------------------------------------------
Because the answer to "is beam better?" is a paired comparison on identical crops,
not two independent accuracies. `wilson` gives each variant an interval; the
McNemar counts (`pair_counts`, `mcnemar_exact`) compare greedy and the best beam
variant on the SAME crops. Six variants within a few points and no significance is
a sample-size problem, and this module is where that gets said out loud.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import realdata as R  # noqa: E402

NEG_INF = -1e30


def _lae(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) without numpy on the hot path.

    Called once per (prefix, class) per timestep -- millions of times on a full
    evaluation -- and np.logaddexp on scalars is roughly an order of magnitude
    slower than this two-branch version.
    """
    if a < b:
        a, b = b, a
    if b <= NEG_INF or a - b > 30.0:
        return a
    return a + math.log1p(math.exp(b - a))


# ── prefix beam search ────────────────────────────────────────────────────────

def beam_search(log_probs, beam=10, blank=None, lm=None, alpha=0.0, lm_gate=0.0):
    """
    CTC prefix beam search over a [T, C] matrix of log-probabilities.

    Returns [(text, log_probability), ...] sorted best-first. `text` is already
    collapsed (blanks removed, repeats merged), so it is directly comparable to
    `realdata.decode_greedy`'s output.

    `lm` is anything with `.logp(prefix_str, char) -> float` (see build_lm.CharLM);
    `alpha` is its weight and 0.0 disables it.

    `lm_gate` makes that weight PER FRAME instead of constant:
    `alpha_t = alpha * (1 - lm_gate * p_max(t))`. Where the CTC is sure of a frame
    the LM is largely silenced, and where the frame is flat the LM is at full
    strength. It exists because a constant alpha has to be small enough not to
    overrule confident frames — which is what costs the crops a word-LM BREAKS
    (measured: 3-4 of 800 against 23-24 fixed) — while those confident frames are
    exactly where the LM was not needed. 0.0 is the shipped constant-alpha
    behaviour, bit for bit.
    """
    blank = R.BLANK if blank is None else int(blank)
    lp = np.asarray(log_probs, dtype=np.float64)
    T, C = lp.shape
    i2c = R.I2C
    use_lm = lm is not None and alpha > 0.0
    gated = use_lm and lm_gate > 0.0

    # prefix (str of emitted chars) -> [logP(alignments ending in blank),
    #                                  logP(alignments ending in that char)]
    beams = {"": [0.0, NEG_INF]}
    for t in range(T):
        row = lp[t]
        l_blank = row[blank]
        alpha_t = alpha
        if gated:
            # how certain the CTC is of this frame, as a plain probability
            alpha_t = alpha * (1.0 - lm_gate * math.exp(float(row.max())))
        nxt = {}
        for prefix, (p_b, p_nb) in beams.items():
            p_tot = _lae(p_b, p_nb)
            # (a) blank: the prefix survives unchanged.
            e = nxt.get(prefix)
            if e is None:
                e = nxt[prefix] = [NEG_INF, NEG_INF]
            e[0] = _lae(e[0], p_tot + l_blank)
            for c in range(C):
                if c == blank:
                    continue
                ch = i2c.get(c)
                if ch is None:
                    continue
                lpc = row[c]
                if prefix and ch == prefix[-1]:
                    # (b) repeat: from a blank it EXTENDS ("A" -> "AA");
                    #     from the char it COLLAPSES ("A" stays "A").
                    ext = nxt.get(prefix + ch)
                    if ext is None:
                        ext = nxt[prefix + ch] = [NEG_INF, NEG_INF]
                    if use_lm:
                        lpc_lm = lpc + alpha_t * lm.logp(prefix, ch)
                    else:
                        lpc_lm = lpc
                    ext[1] = _lae(ext[1], p_b + lpc_lm)
                    e[1] = _lae(e[1], p_nb + lpc)
                else:
                    # (c) new character.
                    ext = nxt.get(prefix + ch)
                    if ext is None:
                        ext = nxt[prefix + ch] = [NEG_INF, NEG_INF]
                    if use_lm:
                        ext[1] = _lae(ext[1], p_tot + lpc + alpha_t * lm.logp(prefix, ch))
                    else:
                        ext[1] = _lae(ext[1], p_tot + lpc)
        if len(nxt) > beam:
            nxt = dict(sorted(nxt.items(),
                              key=lambda kv: -_lae(kv[1][0], kv[1][1]))[:beam])
        beams = nxt if nxt else beams
    ranked = sorted(beams.items(), key=lambda kv: -_lae(kv[1][0], kv[1][1]))
    return [(p, _lae(pb, pnb)) for p, (pb, pnb) in ranked]


def beam_decode(log_probs, beam=10, blank=None, lm=None, alpha=0.0, lm_gate=0.0):
    """The single best-reading string from `beam_search`."""
    out = beam_search(log_probs, beam=beam, blank=blank, lm=lm, alpha=alpha,
                      lm_gate=lm_gate)
    return out[0][0] if out else ""


# ── scoring ───────────────────────────────────────────────────────────────────

def lev(a: str, b: str) -> int:
    """Levenshtein edit distance. Two rows, no dependency."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def exact_match_bits(preds, refs):
    """0/1 per item, under the repo's scoring normalisation (realdata.norm_label).

    Returning BITS rather than a rate is what makes the comparison paired: the
    same crop index in two bit vectors is the same crop, so the discordant cells
    below are a genuine McNemar table.
    """
    return [int(R.norm_label(p) == R.norm_label(r)) for p, r in zip(preds, refs)]


def cer_aggregate(preds, refs) -> float:
    """Character-weighted CER: total edits / total reference characters.

    The standard definition, and the one that does not let a batch of tiny
    labels dominate a batch of long ones the way a mean-of-per-item-rates does.
    """
    edits = chars = 0
    for p, r in zip(preds, refs):
        pn, rn = R.norm_label(p), R.norm_label(r)
        edits += lev(pn, rn)
        chars += max(len(rn), 1)
    return edits / max(chars, 1)


def pair_counts(greedy_bits, variant_bits) -> dict:
    """
    The 2x2 McNemar table for two decoders on the SAME crops.

    both_right / both_wrong are the concordant cells (they carry no information
    about which decoder is better), variant_only is b, greedy_only is c. Swapping
    b and c inverts the hypothesis, so they are named rather than positional.
    """
    both_right = variant_only = greedy_only = both_wrong = 0
    for g, v in zip(greedy_bits, variant_bits):
        if g and v:
            both_right += 1
        elif v and not g:
            variant_only += 1
        elif g and not v:
            greedy_only += 1
        else:
            both_wrong += 1
    return {"both_right": both_right, "variant_only": variant_only,
            "greedy_only": greedy_only, "both_wrong": both_wrong}


# ── statistics ────────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for k successes in n trials (95% at z=1.96).

    Preferred over the normal approximation because at the accuracies this
    recogniser actually reaches (single digits to tens of percent) the normal
    interval produces a NEGATIVE lower bound, which is not a number a report can
    print.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value over the discordant pairs.

    Exact rather than the chi-square approximation: with tens of discordant
    pairs -- the normal case here -- the approximation's p-value is simply wrong,
    and it is wrong in the direction that invents significance.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    # Integer arithmetic end to end: `2.0 ** n` raises OverflowError for n > 1023
    # (hit for real on a 1,152-item paired run), and `2.0 * tail` overflows too once
    # the tail is past the double range. Python ints are arbitrary precision, and
    # int/int true division rounds the quotient (<= 2 here) correctly, so this is
    # exact for every n -- the numerator and denominator never become floats.
    return min(1.0, (2 * tail) / (1 << n))
