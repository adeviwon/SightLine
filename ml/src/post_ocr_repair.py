"""A confusion-matrix-driven post-OCR repair, measured before it is shipped.

WHY THIS AND NOT SOMETHING ELSE. This is the standard post-processing stack used by
self-run OCR pipelines, in the order the literature puts it:

  1. the decoder already biases toward observed word forms (we have that: the word LM
     inside the beam, Schulz & Kuhn's lexical adaptation);
  2. what follows is repair of the tokens the decoder still got wrong, using the
     recogniser's OWN error statistics -- a character confusion matrix -- to search a
     lexicon (Holley/dlib; the post-OCR survey hal-03320329; Taghva & Agarwal measured
     +16.6% over generic spelling suggestions by using the confusion matrix rather than
     generic edit distance);
  3. and the regime rule from ICADL 2020: lexicon repair HELPS when CER > 2% and WER >
     10%, and HURTS below that. Our shipped CER is 14.8%, so we are in the regime where
     it should help -- but the rule is why this file MEASURES instead of assuming.

THE SAFETY RULE, which the literature does not have to care about and this product does:
a repair must never touch a token that carries a figure. Doses, dates, account numbers
and reference numbers are the whole point of the app, and "correcting" `500` toward a
lexicon entry is the failure mode the product exists to prevent. Tokens containing a
digit are never repaired, and a repair is only applied when the candidate is in the
lexicon AND the original is not AND it wins by a margin.

    env -u PYTHONPATH -u VIRTUAL_ENV ./.venv/bin/python ml/src/post_ocr_repair.py [n]
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

import tesseract_compare as TC      # noqa: E402  (CRNN: the shipped graph, the shipped decode)
import build_dataset as B           # noqa: E402  (crop_to_array: the canonical 32px conversion)
import render_corpus as RC          # noqa: E402

MAX_EDITS = 2
MARGIN = 1.15                       # the winner must beat the runner-up by this much


def load_lexicon():
    toks = set()
    for line in (ROOT / "data" / "lexicon.txt").read_text(encoding="utf-8").splitlines():
        t = line.strip().upper()
        if t:
            toks.add(t)
    return toks


def align(g, h):
    """Levenshtein WITH the operation trace, so confusions can be counted."""
    n, m = len(g), len(h)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (g[i - 1] != h[j - 1]))
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (g[i - 1] != h[j - 1]):
            if g[i - 1] != h[j - 1]:
                pairs.append((g[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            pairs.append((g[i - 1], ""))
            i -= 1
        else:
            pairs.append(("", h[j - 1]))
            j -= 1
    return list(reversed(pairs)), D[n][m]


def build_confusion(gt_list, hyp_list, alpha=0.5):
    """P(hyp | gt) with add-alpha smoothing over the observed character set."""
    counts, gt_chars = Counter(), Counter()
    for g, h in zip(gt_list, hyp_list):
        pairs, _ = align(g, h)
        for a, b in pairs:
            counts[(a, b)] += 1
            gt_chars[a] += 1
    chars = set()
    for g in gt_list:
        chars.update(g)
    chars.update(["", " "])
    return counts, gt_chars, chars, alpha


def sub_cost(conf, gt_chars, chars, a, b, alpha):
    """-log P(HYP b | GT a) from the recogniser's own confusion counts."""
    if a == b:
        return 0.0
    n = gt_chars.get(a, 0)
    c = conf.get((a, b), 0)
    p = (c + alpha) / (n + alpha * (len(chars) + 1))
    import math
    return -math.log(p)


def edit_cost(tok, cand, conf, gt_chars, chars, alpha, cap=MAX_EDITS + 1):
    """Confusion-weighted edit distance from `tok` (the READ) to `cand` (a LEXICON
    word): the confusion matrix says which of the read's characters plausibly stand for
    which true ones, which is exactly what plain edit distance throws away."""
    n, m = len(tok), len(cand)
    if abs(n - m) > MAX_EDITS:
        return cap
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i
    for j in range(1, m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(
                D[i - 1][j] + 1.0,
                D[i][j - 1] + 1.0,
                D[i - 1][j - 1] + sub_cost(conf, gt_chars, chars, cand[j - 1], tok[i - 1], alpha),
            )
    return D[n][m] if D[n][m] <= cap else cap


def repair_line(line, lex, conf, gt_chars, chars, alpha, by_len):
    out, changed, detail = [], 0, []
    for tok in line.split(" "):
        if not tok or len(tok) < 3 or re.search(r"\d", tok):
            out.append(tok)              # never touch a token carrying a figure
            continue
        if tok in lex:
            out.append(tok)
            continue
        best = None
        for cand in by_len.get(len(tok), ()):
            c = edit_cost(tok, cand, conf, gt_chars, chars, alpha)
            if c >= MAX_EDITS + 1:
                continue
            if best is None or c < best[1]:
                best, second = (cand, c), best
            elif second is None or c < second[1]:
                second = (cand, c)
        if best and (second is None or second[1] > best[1] * MARGIN):
            detail.append((tok, best[0], round(best[1], 2)))
            out.append(best[0])
            changed += 1
        else:
            out.append(tok)
    return " ".join(out), changed, detail


def word_acc(gt, hyp):
    g, h = set(gt.split(" ")), set(hyp.split(" "))
    g.discard("")
    return len(g & h) / len(g) if g else 1.0


def cer(gt, hyp):
    pairs, d = align(gt, hyp)
    return d / max(1, len(gt))


def main(n=1500):
    lex = load_lexicon()
    print("lexicon: %d tokens   shipped graph: %s\n" % (len(lex), TC.ONNX.name))
    crnn = TC.CRNN()

    tr = RC.load(split="train")
    te = RC.load(split="test")
    toks = lambda arr, k: [str(x) for x in arr[:k]]
    # the corpus stores crops at native size; the recogniser wants the canonical 32px form
    from PIL import Image
    crops32 = lambda arr, k: [B.crop_to_array(Image.fromarray(np.asarray(c, np.uint8)))
                              for c in arr[:k]]

    print("building the confusion matrix from the TRAIN split ...")
    gt_tr = toks(tr["labels"], n)
    hyp_tr = [crnn.read(np.asarray(c, np.float32)) for c in crops32(tr["crops"], n)]
    conf, gt_chars, chars, alpha = build_confusion(gt_tr, hyp_tr)
    top = Counter(conf).most_common(6)
    print("  confusions that matter: " + ", ".join("%s->%s x%d" % (a, b, c) for (a, b), c in top if a != b))
    print("  (a letter standing in for a digit, or the reverse, is the O/0 I/1 S/5 class)")

    by_len = {}
    for w in lex:
        by_len.setdefault(len(w), []).append(w)

    gt_te = toks(te["labels"], n)
    hyp_te = [crnn.read(np.asarray(c, np.float32)) for c in crops32(te["crops"], n)]

    before = {"exact": 0, "word": 0.0, "cer": 0.0}
    after = {"exact": 0, "word": 0.0, "cer": 0.0}
    helped, hurt, total_changes = 0, 0, 0
    samples = []
    for g, h in zip(gt_te, hyp_te):
        r, ch, det = repair_line(h, lex, conf, gt_chars, chars, alpha, by_len)
        total_changes += ch
        b_ex, a_ex = (h.strip() == g.strip()), (r.strip() == g.strip())
        before["exact"] += b_ex
        after["exact"] += a_ex
        before["word"] += word_acc(g, h)
        after["word"] += word_acc(g, r)
        before["cer"] += cer(g, h)
        after["cer"] += cer(g, r)
        if ch:
            if cer(g, r) < cer(g, h):
                helped += 1
            elif cer(g, r) > cer(g, h):
                hurt += 1
            if len(samples) < 5 and det:
                samples.append((g, h, r, det))

    N = len(gt_te)
    print("\nheld-out test crops: %d\n" % N)
    print("  %-16s %-10s %-10s" % ("", "before", "after repair"))
    print("  %-16s %-10.2f%% %-10.2f%%" % ("line exact", 100 * before["exact"] / N, 100 * after["exact"] / N))
    print("  %-16s %-10.2f%% %-10.2f%%" % ("word accuracy", 100 * before["word"] / N, 100 * after["word"] / N))
    print("  %-16s %-10.4f %-10.4f" % ("CER", before["cer"] / N, after["cer"] / N))
    print("\n  lines changed: %d   of those helped %d, hurt %d" % (total_changes, helped, hurt))
    print("  (a repair that changes a line must be right more often than it is wrong:")
    print("   the ICADL rule is that lexicon repair pays above CER 2% / WER 10% and costs below it)")
    for g, h, r, det in samples:
        print("\n   GT %r\n   read %r\n   rep  %r   %s" % (g[:70], h[:70], r[:70], det[:4]))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500))
