"""
build_lm.py — a character n-gram language model, trained on TRAIN labels only.

WHY A CHARACTER N-GRAM AND NOT A WORD LM
----------------------------------------
The CRNN emits characters with no word segmentation, and the target strings are
mixes of words, prices, dates and account numbers ("PRESCRIPTION", "01102026",
"40218877"). A word LM would have to be applied after a segmentation the decoder
does not produce; a character n-gram applies at every emission, which is where
shallow fusion needs it. It is also tiny -- tens of thousands of counts, a few MB
of JSON -- and needs no dependency the repo does not already have.

WHY STUPID BACKOFF AND NOT KATZ / KNESER-NEY
--------------------------------------------
Stupid backoff (Brants et al. 2007) is three lines: a trigram count when one
exists, otherwise 0.4x the bigram, otherwise 0.4^2x the unigram, never
renormalised. It is not a probability distribution, which does not matter here --
the score is a RANKING signal added to an acoustic score with a weight (`alpha`)
that has to be tuned anyway. Proper smoothing buys a fraction of a percent of
perplexity for substantially more code, and perplexity is not the metric; exact
match on held-out crops is.

THE LEAK THIS FILE IS STRUCTURED AROUND
---------------------------------------
The LM trains on `render_corpus.load(split="train")` plus the SROIE training
labels. Evaluating the rescorer on the TEST split is the point; building it from
the test split would publish a number that is partly memorisation, the same leak
`render_corpus.assign_splits` (by page) and `realdata.split_by_document` exist to
prevent. `collect_labels()` is the single call site, its split is a module
constant, and ml/tests/test_ctc_decode.py fails if it ever asks for anything else.

    bash run.sh py ml/src/build_lm.py                       # -> artifacts/char_lm.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np      # noqa: E402
import render_corpus as RC   # noqa: E402
import realdata as R         # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SROIE_CACHE = ROOT / "artifacts" / "ocr_train.npz"
DEFAULT_OUT = ROOT / "artifacts" / "char_lm.json"

# The ONLY split the LM may be built from. Changing this is the leak.
CORPUS_SPLIT = "train"

BOS = "\x02"          # start-of-string context, so trigrams exist at position 0
BACKOFF = 0.4         # stupid-backoff discount


def clean_label(s: str) -> str:
    """The string the LM is allowed to model: charset-clean, upper-cased.

    Deliberately identical to `render_corpus.train_label`, because the LM has to
    score exactly the strings the recogniser is asked to emit. Modelling
    punctuation the model cannot produce would put LM mass on readings the
    decoder can never return.
    """
    kept = "".join(c for c in str(s).upper() if c in R.CHARSET)
    return " ".join(kept.split())


def collect_labels():
    """Every distinct TRAIN label available to the LM.

    Two sources, and both are train-side:
      * `data/render_corpus` TRAIN split -- our own rendered documents, the
        domain the rescorer is judged on. Served through `load(split=...)`.
      * `artifacts/ocr_train.npz` -- the SROIE receipt word labels, which give
        the model the numerals and shop-name vocabulary of photographed paper.

    Deduplicated: the corpus repeats the same line many times (a page rendered
    at three scales, the same header on every page) and the SROIE cache holds one
    row per crop with many repeats, so raw counts would weight a few strings by
    their crop multiplicity rather than by their language.
    """
    labels = [str(x) for x in RC.load(split=CORPUS_SPLIT)["labels"]]
    if SROIE_CACHE.exists():
        z = np.load(SROIE_CACHE, allow_pickle=True)
        labels += [str(x) for x in z["labels"]]
    clean = {clean_label(s) for s in labels}
    clean.discard("")
    return sorted(clean)


class CharLM:
    """
    Trigram character LM with stupid backoff, in log space.

    Counts only; `logp(prefix, char)` is the only query. Kept as plain dicts
    because that is what makes it fast enough to call inside the beam search:
    ~3 dict lookups per candidate emission.
    """

    def __init__(self, lam: float = BACKOFF):
        self.lam = float(lam)
        self.c1 = {}
        self.c2 = {}
        self.c3 = {}
        self.total = 0
        self.vocab = set()

    def add(self, ctx: str, ch: str) -> None:
        """Count one character `ch` observed after context `ctx` (0-2 chars)."""
        b = ctx[-1] if ctx else BOS
        a = ctx[-2] if len(ctx) >= 2 else BOS
        self.c1[ch] = self.c1.get(ch, 0) + 1
        self.c2[b + ch] = self.c2.get(b + ch, 0) + 1
        self.c3[a + b + ch] = self.c3.get(a + b + ch, 0) + 1
        self.total += 1
        self.vocab.add(ch)

    def add_string(self, s: str) -> None:
        s = clean_label(s)
        for i, ch in enumerate(s):
            self.add(s[max(0, i - 2):i], ch)

    def logp(self, ctx: str, ch: str) -> float:
        """log P(ch | ctx), stupid backoff. Never -inf: an unseen continuation
        must be unlikely, not impossible, or fusion will veto a reading the
        acoustics are confident about."""
        if self.total <= 0:
            return 0.0
        b = ctx[-1] if ctx else BOS
        a = ctx[-2] if len(ctx) >= 2 else BOS
        n3 = self.c3.get(a + b + ch)
        n2_ctx = self.c2.get(a + b)
        if n3 and n2_ctx:
            return math.log(n3 / n2_ctx)
        n2 = self.c2.get(b + ch)
        if n2:
            return math.log(self.lam * n2 / max(self.c1.get(b, 0), 1))
        n1 = self.c1.get(ch)
        if n1:
            return math.log(max(self.lam ** 2 * n1 / self.total, 1e-12))
        return math.log(max(self.lam ** 3 / (self.total * max(len(self.vocab), 1)), 1e-12))

    # ── persistence ─────────────────────────────────────────────────────────
    def to_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "lam": self.lam, "total": self.total,
            "c1": self.c1, "c2": self.c2, "c3": self.c3,
        }, ensure_ascii=False) + "\n")
        return path


def build(labels=None, lam: float = BACKOFF) -> CharLM:
    lm = CharLM(lam=lam)
    for s in (collect_labels() if labels is None else labels):
        lm.add_string(s)
    return lm


def load_model(path=None) -> CharLM:
    d = json.loads(Path(path or DEFAULT_OUT).read_text())
    lm = CharLM(lam=d["lam"])
    lm.c1, lm.c2, lm.c3 = d["c1"], d["c2"], d["c3"]
    lm.total = d["total"]
    lm.vocab = set(d["c1"])
    return lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--lam", type=float, default=BACKOFF)
    a = ap.parse_args()

    labels = collect_labels()
    lm = CharLM(lam=a.lam)
    for s in labels:
        lm.add_string(s)
    p = lm.to_json(a.out)
    print("char LM: %d distinct %s labels, %d chars, %d bigrams, %d trigrams"
          % (len(labels), CORPUS_SPLIT, lm.total, len(lm.c2), len(lm.c3)))
    print("wrote %s (%.1f MB)" % (p, p.stat().st_size / 1e6))

    # Self-check: the LM must favour a real continuation over a random one, and
    # must never veto. Both are the reason it exists.
    assert lm.logp("PRESCR", "I") > lm.logp("PRESCR", "Z"), "backoff ordering is broken"
    assert all(math.isfinite(lm.logp("PRESCR", c)) for c in R.CHARSET)
    print("self-check: ordering + finite-backoff OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
