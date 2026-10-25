"""
word_counts.py — the word-LM's VOCABULARY and its FREQUENCIES, mined from the
train split of the data this product serves.

    bash run.sh py ml/src/word_counts.py --out data/lexicon_counts.tsv
    bash run.sh py ml/src/word_counts.py --coverage      # train vs held-out only

WHY FREQUENCIES, AND WHY THEY ARE THE LEVER
-------------------------------------------
`data/lexicon.txt` is a SET: every token counts once, so the trie's
P(c | word-so-far) = (#words taking this continuation) / (#words sharing the
prefix). That is a COVERAGE prior — it knows "ACCOUNT" is a word, not that a
document says it 40x more often than "ACCUMULATOR". Two consequences, both
measurable: a rare-but-real continuation is priced the same as a common one, and
nothing distinguishes a lexicon word that appears on every page from one that
appeared once in a transcript.

Weighting each word by how often it is actually WRITTEN turns the same trie into
a frequency-weighted conditional, which is what the published word-LM decoders
use (their counts come from the training transcriptions, not a dictionary).

WHERE THE COUNTS COME FROM — TRAIN ONLY
---------------------------------------
`field_lexicon._domain_lines()` (the module that builds `data/lexicon.txt`) once
mined with `RC.load()` and no split, so it saw train + val + TEST labels; that
was fixed to `split="train"`. This module takes the same line sources with the
same restriction, and `--coverage` exists so the restriction is CHECKED: it
reports how much of the held-out (test) vocabulary the trie covers, and the
suite's leakage tests stay meaningful because no test label contributes a count.

A word the corpus never repeats still gets weight 1, so coverage never shrinks
below the set it started from.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import field_lexicon as FL   # noqa: E402
import realdata as R         # noqa: E402
import render_corpus as RC   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "lexicon_counts.tsv"
_TOK = re.compile(r"[0-9A-Z]+")


def _train_lines():
    """Domain text, TRAIN split only (see the module docstring)."""
    lines = []
    try:
        import capture
        for doc_lines, _ in capture.DOCS:
            lines.extend(doc_lines)
    except Exception as e:                                    # pragma: no cover
        print("  (capture.DOCS unavailable: %s)" % e)
    try:
        d = RC.load(split="train")
        lines.extend(str(x) for x in d["labels"])
    except Exception as e:                                    # pragma: no cover
        print("  (render corpus TRAIN labels unavailable: %s)" % e)
    try:
        for _, _, tlines in R.handheld_pages():
            lines.extend(tlines)
    except Exception as e:                                    # pragma: no cover
        print("  (handheld transcripts unavailable: %s)" % e)
    return lines


def tokens_of(text):
    """Tokens in the model's OWN output alphabet: the recogniser cannot emit
    anything else, so a lexicon entry it cannot spell is unreachable.

    The membership test is per CHARACTER. `t in _ALPHABET` (a set of characters)
    silently kept only the single-character tokens — it reported 5 types for
    1,125 held-out tokens, which reads as "the lexicon covers almost nothing"
    and is really the filter eating every word."""
    return [t for t in _TOK.findall(str(text).upper())
            if all(c in _ALPHABET for c in t)]


_ALPHABET = set(R.CHARSET)


def mine():
    """word -> count over the train domain text, seeded with every existing
    lexicon word at weight 1 so the mined set is a SUPERSET, never a replacement."""
    counts = collections.Counter()
    for w in (ROOT / "data" / "lexicon.txt").read_text().split():
        w = w.strip().upper()
        if w and len(w) <= 48:
            counts[w] += 1
    listed = len(counts)
    for line in _train_lines():
        for t in tokens_of(line):
            if len(t) <= 48:
                counts[t] += 1
    return counts, listed


def mine_bigrams(min_count=1, prune=True):
    """(prev_word, word) -> count over the TRAIN domain text.

    Bigrams are pairs of ADJACENT TOKENS in a label line, which is what the
    decoder sees: the beam emits a whole line, so a pair straddling the line
    break never occurs and must not be counted."""
    bi = collections.Counter()
    for line in _train_lines():
        toks = tokens_of(line)
        for a, b in zip(toks, toks[1:]):
            if len(a) <= 48 and len(b) <= 48:
                bi[(a, b)] += 1
    if min_count > 1:
        bi = collections.Counter({k: v for k, v in bi.items() if v >= min_count})
    if prune:
        # Keep the pairs the decoder can actually reach a decision with, and cap
        # the fan-out per context: a context with 400 continuations costs JSON on
        # every install and moves few decisions.
        per = collections.defaultdict(list)
        for (a, b), v in bi.items():
            per[a].append((v, b))
        pruned = collections.Counter()
        for a, lst in per.items():
            lst.sort(reverse=True)
            for v, b in lst[:12]:
                pruned[(a, b)] = v
        bi = pruned
    return bi


def held_out_coverage(counts):
    """How much of the HELD-OUT vocabulary does the trie cover, and what fraction
    of held-out tokens are known? Reported for the plain set and the weighted
    one, so 'the vocabulary did not shrink' is a measurement, not a hope."""
    d = RC.load(split="test")
    toks = collections.Counter()
    for label in d["labels"]:
        for t in tokens_of(label):
            toks[t] += 1
    known = sum(c for t, c in toks.items() if t in counts)
    total = sum(toks.values())
    plain = set((ROOT / "data" / "lexicon.txt").read_text().split())
    known_plain = sum(c for t, c in toks.items() if t in plain)
    return {"types": len(toks), "types_covered": sum(1 for t in toks if t in counts),
            "tokens": total, "tokens_covered": known,
            "tokens_covered_plain": known_plain,
            "types_covered_plain": sum(1 for t in toks if t in plain)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--bigrams", default=str(OUT.with_name("word_bigrams.json")))
    ap.add_argument("--min-bigram", type=int, default=1)
    ap.add_argument("--no-prune", action="store_true",
                    help="keep every bigram (default: top-12 per context)")
    ap.add_argument("--coverage", action="store_true",
                    help="print held-out coverage and exit (writes nothing)")
    a = ap.parse_args()

    counts, listed = mine()
    cov = held_out_coverage(counts)
    print("lexicon.txt: %d words (each weight 1)" % listed)
    print("mined:       %d types, %d tokens in TRAIN domain text"
          % (len(counts), int(sum(counts.values()))))
    print("held-out (TEST) coverage: %d/%d types (%.1f%%), %d/%d tokens (%.1f%%)"
          % (cov["types_covered"], cov["types"], 100.0 * cov["types_covered"] / cov["types"],
             cov["tokens_covered"], cov["tokens"], 100.0 * cov["tokens_covered"] / cov["tokens"]))
    print("  plain lexicon.txt on the same labels: %d/%d types, %d/%d tokens"
          % (cov["types_covered_plain"], cov["types"], cov["tokens_covered_plain"], cov["tokens"]))
    assert cov["types_covered"] >= cov["types_covered_plain"], \
        "the weighted lexicon must cover at least what the plain set covered"
    assert cov["tokens_covered"] >= cov["tokens_covered_plain"], \
        "the weighted lexicon must cover at least as many held-out tokens"
    if a.coverage:
        return 0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            fh.write("%s\t%d\n" % (w, c))
    print("wrote %s (%d words, %.1f KB)" % (out, len(counts), out.stat().st_size / 1024.0))

    bi = mine_bigrams(min_count=a.min_bigram, prune=not a.no_prune)
    bi_out = Path(a.bigrams)
    bi_out.write_text(json.dumps({"%s\u0000%s" % k: v for k, v in bi.items()},
                                 separators=(",", ":")))
    kept = sum(1 for k in bi if k[0] in counts and k[1] in counts)
    print("wrote %s (%d bigrams, %d with both words in the lexicon, %.1f KB)"
          % (bi_out, len(bi), kept, bi_out.stat().st_size / 1024.0))
    print("top: " + ", ".join("%s(%d)" % kv for kv in counts.most_common(8)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
