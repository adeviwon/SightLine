"""
dose_lexicon.py — the word LM's vocabulary for MEDICATION DOSES, built from real
medication labels instead of from our own renders.

    bash run.sh py ml/src/dose_lexicon.py --build        # lexicon + bigrams + try
    bash run.sh py ml/src/dose_lexicon.py --coverage     # what it covers, vs shipped

WHY A SEPARATE LEXICON
----------------------
The shipped trie is 2,878 words learned from the render corpus's own training
labels — the vocabulary of DOCUMENTS WE RENDERED. The product reads dose panels, and
a dose panel's language is narrow, repetitive and highly predictable:

    DOSAGE AND ADMINISTRATION   ACTIVE INGREDIENT      TAKE ONE TABLET
    TWICE DAILY                 EVERY 6 HOURS          DO NOT EXCEED
    KEEP OUT OF REACH OF CHILDREN                      MG  ML  CAPSULE

None of that has to be guessed at: `fetch_dose_labels.py` already pulled 1,239 real
FDA labels (15,501 panel lines, 5,312 unique) from openFDA's structured sections.
Their vocabulary and their word order are the real thing, so the LM can be trained on
what a doses tab actually says rather than on what our renderer happened to draw.

WHAT IT EMITS
-------------
* `data/dose_lexicon.txt`   word<TAB>count, the frequency-weighted vocabulary
* `data/dose_bigrams.json`  P(next word | word) counts over real adjacent pairs
* `data/dose_trie.json`     the browser's trie format (word_lm.export_trie), i.e.
                            something `app/js/word_beam.js` can consume unchanged

The counts are a PRIOR, not a rule: a word the panels never use is still decodable,
it just pays `penalty`. That distinction is why the trie is the right structure here
and a hard vocabulary whitelist is not.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

import word_lm as WL          # noqa: E402

PANELS = ROOT / "data" / "dose_panels.txt"
LEX = ROOT / "data" / "dose_lexicon.txt"
BIGRAMS = ROOT / "data" / "dose_bigrams.json"
TRIE = ROOT / "app" / "data" / "dose_trie.json"
SHIPPED_TRIE = ROOT / "app" / "data" / "word_trie.json"

# The charset the recogniser can emit. A lexicon word outside it could never be
# produced by the model, so it would only ever add dead weight to the trie.
KEEP = re.compile(r"[^A-Z0-9]")

# Dose language is dominated by units and directions; these tiny tokens carry a lot
# of the panel's meaning and are exactly what a character LM gets wrong, so they are
# never pruned for being short or rare.
CORE = {
    "MG", "ML", "G", "MCG", "IU", "TABLET", "TABLETS", "CAPSULE", "CAPSULES",
    "DOSE", "DOSES", "DOSAGE", "DAILY", "TWICE", "THRICE", "EVERY", "HOURS",
    "HOUR", "TAKE", "ORAL", "ORALLY", "SWALLOW", "CHEW", "DISSOLVE", "DROP",
    "DROPS", "ML", "APPLY", "INJECT", "ADULT", "ADULTS", "CHILDREN", "YEARS",
    "ACTIVE", "INACTIVE", "INGREDIENT", "INGREDIENTS", "WARNINGS", "DIRECTIONS",
    "PURPOSE", "STORAGE", "INDICATIONS", "USAGE", "ADMINISTRATION", "REACH",
    "KEEP", "OUT", "OF", "EXCEED", "PHYSICIAN", "DOCTOR", "PRESCRIPTION",
    "STRENGTH", "TABLETSPER", "USP", "RX", "ONLY", "MGPER", "PER", "DAY",
}


def tokens(text):
    """Words as the recogniser can spell them: upper case, charset only."""
    out = []
    for raw in str(text).replace("\r", " ").replace("\n", " ").split():
        w = KEEP.sub("", raw.upper())
        if w:
            out.append(w)
    return out


def mine(min_count=1):
    """Vocabulary and word bigrams from the real panel text."""
    if not PANELS.exists():
        raise SystemExit("no real panel text at %s — run: bash run.sh fetch-dose-labels"
                         % PANELS)
    lines = [l for l in PANELS.read_text().splitlines() if l.strip()]
    uni, bi = Counter(), Counter()
    for line in lines:
        ws = tokens(line)
        uni.update(ws)
        for a, b in zip(ws, ws[1:]):
            bi[(a, b)] += 1
    # SINGLETONS ARE KEPT at weight 1 — that is the interesting half. A word used
    # once in 15,501 real lines is still real dose language, and dropping it is how a
    # lexicon quietly becomes a list of the most common words only.
    kept = {w: c for w, c in uni.items() if c >= min_count or w in CORE}
    # A bigram seen once is noise; twice is a phrase someone actually wrote twice.
    bi = {k: v for k, v in bi.items() if v >= 2}
    return kept, bi, len(lines), len(uni)


def build(min_count=1, bigram_w=0.0, out_trie=None, extra=None,
          bigram_min=3, bigram_cap=12000, union_scale=0.0):
    uni, bi, n_lines, n_types = mine(min_count)
    LEX.write_text("\n".join("%s\t%d" % (w, c) for w, c in
                             sorted(uni.items(), key=lambda kv: -kv[1])))
    BIGRAMS.write_text(json.dumps({"%s\u0000%s" % k: v for k, v in bi.items()}, indent=0))
    words = dict(uni)
    if extra:
        # UNION ON A SHARED TOKEN BUDGET, not with a placeholder count of 1.
        #
        # Scoring is log(count(child)/count(parent)), so a word's count IS its weight:
        # giving the document lexicon's 2,878 words a count of 1 against dose counts in
        # the hundreds would leave them effectively unscored, i.e. the union would
        # silently behave like the doses-only lexicon — which is exactly what the first
        # union measurement showed (dose lexicon +9.3 pts on dose panels, -3.4 pts on
        # rendered documents). Rescaling the incoming counts so both lexicons contribute
        # comparable mass is what makes it a UNION.
        edict = dict(extra) if not isinstance(extra, dict) else dict(extra)
        scale = float(union_scale) if union_scale > 0 else (
            sum(uni.values()) / max(1.0, float(sum(edict.values()))))
        for w, c in edict.items():
            words[w] = words.get(w, 0.0) + max(1.0, float(c) * scale)
        print("union: %d dose tokens + %d document tokens x scale %.3f"
              % (len(uni), len(extra), scale))
    lm = WL.WordLM(sorted(words.items(), key=lambda kv: -kv[1]),
                   bigrams=bi, bigram_w=bigram_w)
    out = Path(out_trie or TRIE)
    out.parent.mkdir(parents=True, exist_ok=True)
    WL.export_trie(lm, out=out, bigram_min=bigram_min, bigram_cap=bigram_cap)
    print("real panels %d lines, %d word types -> kept %d (%d bigrams, %d phrases>=2)"
          % (n_lines, n_types, len(uni), len(bi), sum(1 for v in bi.values() if v >= 2)))
    print("wrote %s, %s, %s" % (LEX.name, BIGRAMS.name,
                                out.relative_to(ROOT) if str(out).startswith(str(ROOT)) else out))
    assert uni and bi, "empty lexicon or bigrams — check the panel source"
    # A handful of tokens do not survive WordLM's own alphabet filter (the model
    # cannot emit them anyway), so the counts differ slightly by design. A large
    # difference is still a bug: report it instead of hiding it.
    lost = len(words) - lm.words
    print("lexicon %d tokens (incl. any union), trie holds %d (%d outside the "
          "model's alphabet)" % (len(words), lm.words, lost))
    assert 0 <= lost <= max(20, len(words) // 100), "trie dropped far too many words"
    return lm


def coverage():
    """How the doses vocabulary compares with the shipped one, honestly."""
    if not LEX.exists():
        raise SystemExit("run --build first")
    dose = {l.split("\t")[0] for l in LEX.read_text().splitlines() if l.strip()}
    shipped = set(WL.load_lexicon(ROOT / "data" / "lexicon.txt"))
    print("doses lexicon   %5d words" % len(dose))
    print("shipped lexicon %5d words" % len(shipped))
    print("shared          %5d" % len(dose & shipped))
    print("doses only      %5d  e.g. %s" % (len(dose - shipped),
                                            ", ".join(sorted(dose - shipped)[:12])))
    # What the model is actually asked to read, on the two held-out sets that exist.
    for name, path, key in (("render test", ROOT / "data" / "render_corpus" / "crops.npz", "split"),
                            ("bottle test", ROOT / "data" / "bottle_corpus" / "crops.npz", "split")):
        if not path.exists():
            continue
        import numpy as np
        z = np.load(path, allow_pickle=True)
        m = np.array([str(s) == "test" for s in z[key]])
        toks = [t for lab in z["labels"][m] for t in tokens(lab)]
        d = sum(1 for t in toks if t in dose)
        s = sum(1 for t in toks if t in shipped)
        both = sum(1 for t in toks if t in dose or t in shipped)
        print("%s: %d tokens — doses covers %.1f%%, shipped %.1f%%, together %.1f%%"
              % (name, len(toks), 100.0 * d / max(1, len(toks)),
                 100.0 * s / max(1, len(toks)), 100.0 * both / max(1, len(toks))))
    print("NOTE coverage is not accuracy: a word the lexicon lacks is still decodable, "
          "it just pays the penalty. The lexicon only decides which readings get a "
          "prior — measure with eval-decoder-doses, never infer from this table.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--bigram-weight", type=float, default=0.0)
    ap.add_argument("--out-trie", default="")
    ap.add_argument("--union-scale", type=float, default=0.0,
                    help="0 = auto (equalise token mass between the two lexicons)")
    ap.add_argument("--union-shipped", action="store_true",
                    help="union with data/lexicon.txt (the product also reads documents)")
    a = ap.parse_args()
    if a.coverage:
        return coverage()
    if a.build or not (a.coverage or a.build):
        if a.union_shipped:
            # Prefer the FREQUENCY-WEIGHTED document lexicon when it is on disk: the
            # plain one is a word list, and unioning counts against a list of ones
            # weights the product's own vocabulary at a thousandth of the dose text.
            counts = ROOT / "data" / "lexicon_counts.tsv"
            if counts.exists():
                extra = {}
                for line in counts.read_text().splitlines():
                    if not line.strip():
                        continue
                    parts = line.split("\t")
                    extra[parts[0]] = float(parts[1]) if len(parts) > 1 else 1.0
            else:
                extra = {w: 1.0 for w in WL.load_lexicon(ROOT / "data" / "lexicon.txt")}
        else:
            extra = None
        build(a.min_count, a.bigram_weight, a.out_trie or None, extra,
              union_scale=a.union_scale)
        coverage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
