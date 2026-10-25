"""
word_lm.py — a WORD-level language model for CTC decoding, built from a lexicon.

    bash run.sh py ml/src/word_lm.py --check          # self-check + stats
    bash run.sh py ml/src/word_lm.py --export         # write app/data/word_trie.json

WHY A TRIE AND NOT AN N-GRAM
----------------------------
`ctc_decode.beam_search` scores an LM exactly once per EMITTED character via
`lm.logp(prefix, ch)`. A word-level LM wants to say "how plausible is the word
being spelled?" — which is a per-CHARACTER question if you express it through a
trie: P(c | word-so-far) = count(node(w+c)) / count(node(w)), where node counts
are the number of lexicon words passing through. This gives the published
word-LM behaviour (lexicon-constrained continuations, IAM ~19pp WER) inside the
existing char-emit hook, with zero changes to the beam search itself.

WHAT IT KNOWS
-------------
`data/lexicon.txt` — 2,878 tokens harvested from the corpora this product
serves: document keys (PRESCRIPTION, ACCOUNT, TOTAL), units (MG ML KG),
medicine names, and the digit strings real fields take (account numbers,
dates). A digit-run continuation is therefore guided by evidence, and the
letter/digit confusions that make a dosage wrong (500mg -> 5O0mg) get
penalised when no lexicon word continues that way.

SCORING
-------
* continuation that exists:  log(count(child) / count(parent))  (<= 0)
* continuation no lexicon word takes: PENALTY (a fixed log-eps), because that
  is the information a word-LM has: this string is not a word we serve.
* space emits: score the word just completed — terminal words get a bonus
  relative to their node count; non-terminal words get the penalty (a word
  that ends where no lexicon word ends).
* empty current word (double space, leading space): neutral 0.0.

Alpha semantics match `ctc_decode`: the value is multiplied into the returned
log-prob by the beam search (`alpha`), so the trie returns RAW log-probs and
never scales itself.

MEASURED, AND REJECTED — do not re-litigate these without new data
------------------------------------------------------------------
Every row below was decoded from the SAME logits as the shipped config on the
FULL held-out split (3,375 crops), paired against the shipped config with exact
McNemar (`ml/src/eval_decode_confirm.py`; b/c = crops the row fixes / breaks):

shipped  plain lexicon, beam 4, alpha 0.25          70.7%  (+114/-23 vs greedy)
counts   frequency-weighted lexicon (word_counts.py) +6/-4   p=0.75  — no effect
bigram40 + this file's word bigram, weight 0.4       +3/-13  p=0.021 — WORSE
bigram80 + word bigram, weight 0.8                   +3/-19  p=0.0009 — worse
beta75   word-insertion bonus 0.75                   +0/-8   p=0.008 — worse
beam8    beam width 8                                +0/-7   p=0.016 — worse, 2.29x slower
alpha20 / alpha30 / alpha35                          +6/-9 p=0.61 (noise) / +0/-6 p=0.031 / +2/-13 p=0.007
lm_gate  confidence-weighted per-frame fusion        -1.75pp exact at gate 1.0 (800-crop subset)
backoff  char-LM backoff for unseen continuations    0 crops changed (800-crop subset)

THE TRAP THIS TABLE EXISTS FOR: the sweep ranks cells against GREEDY, so
bigram40 looked like "+10 crops over shipped" (71.0% vs 70.7%) — and paired
against shipped it is 10 crops NET WORSE. A shared baseline hides the comparison
that decides the ship; use eval_decode_confirm.py before believing a sweep.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import realdata as R  # noqa: E402

LEXICON = Path(__file__).resolve().parents[2] / "data" / "lexicon.txt"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "app" / "data" / "word_trie.json"

# penalty for a continuation no lexicon word takes. -6 ~ 1/400; strong enough to
# break an O/0 tie the CTC itself cannot, weak enough that a 12-char line of
# them (-72) still loses to any real alternative rather than banning readings.
PENALTY = -6.0
# A DIGIT continuation inside a digit run that is not in the lexicon is NOT an
# error: account numbers, sort codes and dates are open-ended, and the lexicon
# only holds the ones the corpus happened to contain. Punishing them at -6 per
# character would make the decoder corrupt every unseen number, which is the
# one field class this product cannot get wrong. A LETTER inside a digit run
# still takes the full penalty — that is the O/0, I/1, S/5 confusion the
# decoder exists to fix ("0A7E01102026" for 01102026).
DIGIT_PENALTY = -0.5
# smoothing floor for the ratio: a child with count 1 under a parent with count
# 300 must not read as -log(300) ~ -5.7 *for a continuation that EXISTS*.
EPS = 0.5
# Word-insertion bonus: added per COMPLETED word (at the space). 0.0 is neutral;
# a positive value makes the decoder more willing to break a line into words, a
# negative one more willing to run characters together. It exists because the
# per-character ratios above only price "is this a word", never "how many words
# should this line have" — a lever the published word-LM decoders carry and this
# one did without.
BETA = 0.0
# How much weight the WORD-BIGRAM context gets inside the trie's space score:
# `+ BIGRAM_W * log P(word | previous word)`. The trie alone cannot know that
# "JANE" is likelier after "PATIENT" than after "TOTAL"; the bigram is mined from
# the same train labels as the counts. 0.0 disables it.
BIGRAM_W = 0.0
# How much weight a CHARACTER model gets when the trie has NO continuation,
# replacing part of the flat PENALTY with a graded score: the flat -6 says "not a
# word" and throws away the information that the character was still plausible.
# 0.0 keeps the shipped flat behaviour.
BACKOFF_W = 0.0


def _is_digit_run(w):
    return all("0" <= c <= "9" for c in w)


def _new_node():
    """[pass_weight, terminal_weight, children] — children is a plain dict."""
    return [0.0, 0.0, {}]


def clean_token(s):
    """The trie's alphabet — the model's OUTPUT alphabet (upper + charset), NOT
    `realdata.norm_label` (that maps O->0/S->5/B->8 for SCORING only, and a trie
    built with it would hold "ACC0UNT" while the beam asks for "ACCOUNT")."""
    return " ".join("".join(c for c in str(s).upper() if c in R.CHARSET).split())


class WordLM:
    """Trie over upper-case charset-clean lexicon words. `logp(prefix, ch)`."""

    def __init__(self, words, penalty=PENALTY, digit_penalty=DIGIT_PENALTY, beta=BETA,
                 eps=EPS, bigrams=None, bigram_w=BIGRAM_W, backoff=None,
                 backoff_w=BACKOFF_W):
        # node = [pass_weight, terminal_weight, children]
        # Weights are COUNTS when the lexicon carries corpus frequencies and 1 per
        # word when it does not, so the same trie serves both and the scored
        # quantity stays comparable: (child + EPS) / (node + EPS).
        self.root = _new_node()
        self.words = 0
        self.total = 0.0
        self.penalty, self.digit_penalty, self.beta, self.eps = (
            penalty, digit_penalty, beta, eps)
        # (prev_word, word) -> count over TRAIN label text; context the trie cannot express
        self.bigrams = dict(bigrams or {})
        self.bigram_w = bigram_w
        # totals per previous word, and the vocabulary size for smoothing — derived
        # here so a caller cannot supply bigrams and forget them.
        self.bigram_prev = {}
        for (p_, _w), c in self.bigrams.items():
            self.bigram_prev[p_] = self.bigram_prev.get(p_, 0.0) + c
        self.bigram_vocab = max(1, len(set(w_ for _p, w_ in self.bigrams)))
        # any object with .logp(prefix, ch) used ONLY where the trie has nothing
        self.backoff = backoff
        self.backoff_w = backoff_w
        for item in words:
            if isinstance(item, (tuple, list)):
                w, cnt = item[0], float(item[1])
            else:
                w, cnt = item, 1.0
            if cnt <= 0:
                continue
            w = clean_token(w)
            if not w or len(w) > 48:
                continue
            node = self.root
            node[0] += cnt
            for ch in w:
                node = node[2].setdefault(ch, _new_node())
                node[0] += cnt
            node[1] += cnt
            self.words += 1
            self.total += cnt
        if not self.words:
            raise ValueError("empty lexicon: refusing to build a useless LM")

    def _word_prefix(self, prefix):
        """The current (partial) word: the suffix after the last space."""
        return prefix.rsplit(" ", 1)[-1] if prefix else ""

    def logp(self, prefix: str, ch: str) -> float:
        w = self._word_prefix(prefix)
        # a space completes the word being spelled: the word-LM's one binary
        # judgement is "is this a word we serve", plus the word-insertion bonus.
        if ch == " ":
            if not w:
                return 0.0  # double space / leading space: nothing to score
            node = self._node(w)
            done = node is not None and node[1] > 0
            score = self.beta if done else self.beta + (
                self.digit_penalty if _is_digit_run(w) else self.penalty)
            # context the trie has no opinion about: how likely this word is to
            # FOLLOW the previous one. Applied on the completed word only.
            if self.bigram_w and self.bigrams:
                score += self.bigram_w * self.bigram_logp(prefix, w)
            return score
        node = self._node(w)
        child = node[2].get(ch) if node is not None else None
        if child is not None and child[0] > 0:
            return math.log((child[0] + self.eps) / (node[0] + self.eps))
        # not a path any lexicon word takes: the flat penalty, graded by a
        # character model when one is attached (see BACKOFF_W).
        back = (self.backoff_w * self.backoff.logp(prefix, ch)
                if self.backoff is not None and self.backoff_w else 0.0)
        if "0" <= ch <= "9" and (w == "" or _is_digit_run(w)):
            return self.digit_penalty + back
        return self.penalty + back

    def bigram_logp(self, prefix, w):
        """log P(w | previous word), with an unseen pair priced as one
        pseudo-count over the previous word's total. `prefix` ends with `w`."""
        words = prefix.split()
        if len(words) < 2:
            return 0.0                     # nothing to condition on
        prev = words[-2]
        if prev == w:
            return 0.0                     # a repeated word is not evidence
        total = self.bigram_prev.get(prev, 0.0)
        if total <= 0:
            return 0.0                     # an unseen context says nothing
        seen = self.bigrams.get((prev, w), 0.0)
        return math.log((seen + self.eps) / (total + self.eps * self.bigram_vocab))

    def _node(self, w):
        node = self.root
        for ch in w:
            node = node[2].get(ch)
            if node is None:
                return None
        return node


def load_lexicon(path=LEXICON):
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def export_trie(lm: WordLM, out=DEFAULT_OUT):
    """Ship form: nested dicts, counts inline. Small enough to precache."""
    def enc(node):
        return {"p": node[0], "t": node[1],
                "c": {ch: enc(ch_node) for ch, ch_node in node[2].items()}}
    # meta carries every constant the JS mirror needs, so the browser cannot
    # drift from the Python the sweep tuned: the numbers are READ from the
    # artifact, never retyped in word_beam.js.
    data = {"meta": {"words": lm.words, "total": lm.total,
                     "penalty": lm.penalty, "penalty_digit": lm.digit_penalty,
                     "beta": lm.beta, "eps": lm.eps},
            "root": enc(lm.root)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data, separators=(",", ":")))
    return Path(out).stat().st_size


def load_trie_json(path):
    """The JS-side mirror loads the SAME json; this loader exists for parity
    checks — Python must be able to reproduce the JS scoring byte for byte."""
    data = json.loads(Path(path).read_text())

    class _TrieLM:
        def __init__(self, root, penalty, penalty_digit, beta=0.0, eps=EPS):
            self.root, self.penalty = root, penalty
            self.penalty_digit, self.beta, self.eps = penalty_digit, beta, eps

        def _node(self, w):
            node = self.root
            for ch in w:
                node = node["c"].get(ch)
                if node is None:
                    return None
            return node

        def logp(self, prefix, ch):
            w = prefix.rsplit(" ", 1)[-1] if prefix else ""
            if ch == " ":
                if not w:
                    return 0.0
                node = self._node(w)
                if node is not None and node["t"] > 0:
                    return self.beta
                return self.beta + (self.penalty_digit if _is_digit_run(w)
                                    else self.penalty)
            node = self._node(w)
            if node is not None:
                child = node["c"].get(ch)   # _node returns the NODE, not its children
                if child is not None and child["p"] > 0:
                    return math.log((child["p"] + self.eps) / (node["p"] + self.eps))
            if "0" <= ch <= "9" and (w == "" or _is_digit_run(w)):
                return self.penalty_digit
            return self.penalty

    root = data["root"]
    meta = data["meta"]
    return _TrieLM(root, meta["penalty"], meta["penalty_digit"],
                   meta.get("beta", 0.0), meta.get("eps", EPS))


def _self_check():
    lm = WordLM(load_lexicon())
    assert lm.words > 0
    # every internal step of a lexicon word is a real path, and completing it
    # at a boundary is neutral; a word nobody serves is not a word.
    w = clean_token("ACCOUNT")
    assert lm._node(w) is not None, "lexicon must contain ACCOUNT"
    s = sum(lm.logp(w[:i], w[i]) for i in range(len(w)))
    done = lm.logp(w, " ")
    assert s <= 0.0, s
    assert done == 0.0, done
    assert lm.logp("ACCOUNTT", " ") == PENALTY
    # the letter/digit confusion the decoder exists to fix: a LETTER inside a
    # digit run takes the full penalty, an unseen NUMBER does not.
    assert lm.logp("0110", "A") == PENALTY, "a letter inside a digit run"
    assert lm.logp("0110202", "7") == DIGIT_PENALTY, "an unseen number is fine"
    assert lm.logp("", "4") <= 0.0
    # parity: the shipped JSON must score identically on every (prefix, ch)
    # the beam search can ask, including spaces and unknown words
    import random
    rng = random.Random(7)
    shipped = Path("/tmp/_word_trie_parity.json")
    export_trie(lm, shipped)
    tj = load_trie_json(shipped)
    alphabet = list(R.CHARSET) + [" "]
    words = load_lexicon()
    cases = [(rng.choice(words), rng.choice(alphabet)) for _ in range(500)]
    cases += [("PRE5CR1PT10N", c) for c in alphabet]
    cases += [("A", " "), ("", "X"), ("0110", "A"), ("0110202", "7")]
    for prefix, ch in cases:
        a, b = lm.logp(prefix, ch), tj.logp(prefix, ch)
        assert abs(a - b) < 1e-9, (prefix, ch, a, b)
    shipped.unlink()
    print("self-check OK: %d lexicon words; %d (prefix, ch) cases identical "
          "through the shipped JSON" % (lm.words, len(cases)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    if a.check or not (a.export or a.check):
        _self_check()
    if a.export:
        n = export_trie(WordLM(load_lexicon()), a.out)
        print("exported %s (%d B)" % (a.out, n))
