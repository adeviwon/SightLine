"""
Word-level recognition vs line-level recognition, on the same held-out crops.

WHY. The recogniser reads ONE squeezed crop per line. Exact-match on an
18-character line needs 98.8% character accuracy; on an 8-character word it
needs only 97.2% ((1-CER)^L = 0.8). Tesseract reads word crops constrained by a
lexicon and never asks a model to read a whole line in one shot — that is why it
beats our CRNN on clean text despite a weaker net. This measures the same idea
on our own stack:

  * WORD level  — segment each detected line at its ink gaps into word crops,
                  recognise each, compare to its word label
  * LINE level  — recognise the whole line crop, the shipped path (baseline:
                  4.4% exact / CER 0.53 / 8.3% <=1 edit over 781 crops)

Reported per capture profile and overall, with Wilson 95% CIs and segmentation
health (words found vs words in the label, over/under-split counts), so a bad
number can be attributed to the SEGMENTER or to the RECOGNISER instead of being
ambiguous. A line whose words are all read exactly but whose squeezed line read
is wrong is the case this lane exists to find, so those are counted separately
(McNemar-style paired discordance).

    bash run.sh py ml/src/eval_word_level.py --split test
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as B       # noqa: E402
import eval_ocr as EO           # noqa: E402
import eval_render_corpus as ERC  # noqa: E402
import field_lexicon as FL      # noqa: E402
import realdata as R            # noqa: E402
import render_corpus as RC      # noqa: E402
import word_segment as WS       # noqa: E402

Z = 1.96


def wilson(k, n, z=Z):
    """Wilson score interval for a proportion. (lo, hi), 0..1."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _blank():
    return {
        "n_lines": 0, "n_lab_words": 0, "n_found_words": 0,
        # word-level, end-to-end over label words (misses count as errors)
        "w_exact": 0.0, "w_cer": 0.0, "w_w1": 0.0,
        # line-level, same crops
        "l_exact": 0.0, "l_cer": 0.0, "l_w1": 0.0,
        # segmentation health
        "match_lines": 0, "over_lines": 0, "under_lines": 0, "zero_lines": 0,
        "total_over": 0, "total_under": 0,
        # paired discordance on "all words of the line read exactly" vs line read
        "both_ok": 0, "b_word_ok_line_bad": 0, "c_line_ok_word_bad": 0,
        "both_bad": 0,
        # paired-only word metrics (segmentation loss excluded)
        "p_n": 0, "p_exact": 0.0, "p_cer": 0.0, "p_w1": 0.0,
    }


def score_split(data, model, device, limit=None, show_misses=8):
    per = defaultdict(_blank)
    misses = []
    for i, (crop, label, prof) in enumerate(zip(data["crops"], data["labels"],
                                                data["profile"])):
        if limit and i >= limit:
            break
        g = np.asarray(crop, dtype=np.uint8)
        ref = R.norm_label(label)
        lab_words = [R.norm_label(w) for w in str(label).split()]
        lab_words = [w for w in lab_words if w]
        row = per[prof]
        row["n_lines"] += 1
        row["n_lab_words"] += len(lab_words)

        # ── line level: one squeezed crop, the shipped path ──────────────────
        arr = B.crop_to_array(Image.fromarray(g))
        txt, _ = EO.recognise(model, arr, device)
        got = R.norm_label(txt)
        l_ok = got == ref
        row["l_exact"] += 1.0 if l_ok else 0.0
        row["l_cer"] += R.cer(got, ref)
        row["l_w1"] += 1.0 if ERC.lev(got, ref) <= 1 else 0.0
        if not l_ok and len(misses) < show_misses:
            misses.append(("LINE", ref, got))

        # ── word level: segment at ink gaps, read each word crop ─────────────
        boxes = WS.segment_words(g, g.shape[1], (0, 0, g.shape[1], g.shape[0]))
        row["n_found_words"] += len(boxes)
        n_pairs = min(len(boxes), len(lab_words))
        all_words_ok = len(boxes) == len(lab_words)

        for j, wref in enumerate(lab_words):
            if j < n_pairs:
                x0, y0, x1, y1 = boxes[j]
                sub = g[y0:y1, x0:x1]
                if sub.size == 0:
                    wgot = ""
                else:
                    warr = B.crop_to_array(Image.fromarray(sub))
                    wgot, _ = EO.recognise(model, warr, device)
                    wgot = R.norm_label(wgot)
                ok = wgot == wref
                row["p_n"] += 1
                row["p_exact"] += 1.0 if ok else 0.0
                row["p_cer"] += R.cer(wgot, wref)
                row["p_w1"] += 1.0 if ERC.lev(wgot, wref) <= 1 else 0.0
                row["w_exact"] += 1.0 if ok else 0.0
                row["w_cer"] += R.cer(wgot, wref)
                row["w_w1"] += 1.0 if ERC.lev(wgot, wref) <= 1 else 0.0
                if not ok:
                    all_words_ok = False
                    if len(misses) < show_misses:
                        misses.append(("WORD", wref, wgot))
            else:
                # label word with no detected box -> segmentation miss, full error
                all_words_ok = False
                row["w_exact"] += 0.0
                row["w_cer"] += 1.0
                row["w_w1"] += 0.0

        # segmentation health
        if len(boxes) == len(lab_words):
            row["match_lines"] += 1
        elif len(boxes) > len(lab_words):
            row["over_lines"] += 1
            row["total_over"] += len(boxes) - len(lab_words)
        else:
            row["under_lines"] += 1
            row["total_under"] += len(lab_words) - len(boxes)
        if not boxes:
            row["zero_lines"] += 1

        # paired discordance: word-level line success vs squeezed-line success
        if all_words_ok and l_ok:
            row["both_ok"] += 1
        elif all_words_ok and not l_ok:
            row["b_word_ok_line_bad"] += 1
        elif l_ok and not all_words_ok:
            row["c_line_ok_word_bad"] += 1
        else:
            row["both_bad"] += 1
    return per, misses


def _pct(x):
    return "%.1f%%" % (100.0 * x)


def _ci(k, n):
    lo, hi = wilson(k, n)
    return "%.1f-%.1f" % (100.0 * lo, 100.0 * hi)


def _avg(d):
    n = max(1, d["n_lines"])
    wn = max(1, d["n_lab_words"])
    pn = max(1, d["p_n"])
    return {
        "n_lines": d["n_lines"], "n_lab_words": d["n_lab_words"],
        "w_exact": d["w_exact"] / wn, "w_cer": d["w_cer"] / wn,
        "w_w1": d["w_w1"] / wn,
        "l_exact": d["l_exact"] / n, "l_cer": d["l_cer"] / n,
        "l_w1": d["l_w1"] / n,
        "p_exact": d["p_exact"] / pn, "p_cer": d["p_cer"] / pn,
        "p_w1": d["p_w1"] / pn, "p_n": d["p_n"],
        "n_found_words": d["n_found_words"],
        "match_lines": d["match_lines"], "over_lines": d["over_lines"],
        "under_lines": d["under_lines"], "zero_lines": d["zero_lines"],
        "total_over": d["total_over"], "total_under": d["total_under"],
        "both_ok": d["both_ok"], "b_word_ok_line_bad": d["b_word_ok_line_bad"],
        "c_line_ok_word_bad": d["c_line_ok_word_bad"], "both_bad": d["both_bad"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    data = RC.load(split=a.split)
    n = data["crops"].shape[0]
    if not n:
        print("no crops: build the corpus first "
              "(bash run.sh py ml/src/render_corpus.py --pages 120)")
        return 1

    lex = Path(FL.ROOT) / "data" / "lexicon.txt"
    if not lex.exists():
        rep = FL.build()
        print("built lexicon: %d tokens -> %s" % (rep["total"], rep["out_path"]))
    lex_tokens = set(lex.read_text().split())

    model, device = ERC.load_model(a.ckpt or None)
    per, misses = score_split(data, model, device, limit=a.limit or None)

    print("=" * 88)
    print("WORD-LEVEL vs LINE-LEVEL — split=%s, %d crops, checkpoint %s"
          % (a.split, n if not a.limit else min(n, a.limit),
             Path(a.ckpt or EO.CKPT).name))
    print("=" * 88)
    print("%-14s %5s %6s | %6s %5s %6s | %6s %5s %6s | %s"
          % ("profile", "lines", "words", "wExact", "wCER", "w<=1",
             "lExact", "lCER", "l<=1", "wExact 95% CI"))
    print("-" * 88)
    agg = defaultdict(lambda: defaultdict(float))
    for prof in sorted(per):
        d = _avg(per[prof])
        print("%-14s %5d %6d | %6s %5.2f %6s | %6s %5.2f %6s | %s"
              % (prof, d["n_lines"], d["n_lab_words"], _pct(d["w_exact"]),
                 d["w_cer"], _pct(d["w_w1"]), _pct(d["l_exact"]), d["l_cer"],
                 _pct(d["l_w1"]), _ci(d["w_exact"] * d["n_lab_words"],
                                      d["n_lab_words"])))
        for k, v in d.items():
            agg[k][prof] = v
    print("-" * 88)

    # overall, pooled across profiles
    tot = _blank()
    for prof in per:
        for k in tot:
            tot[k] += per[prof][k]
    d = _avg(tot)
    print("%-14s %5d %6d | %6s %5.2f %6s | %6s %5.2f %6s | %s"
          % ("ALL", d["n_lines"], d["n_lab_words"], _pct(d["w_exact"]),
             d["w_cer"], _pct(d["w_w1"]), _pct(d["l_exact"]), d["l_cer"],
             _pct(d["l_w1"]), _ci(tot["w_exact"], tot["n_lab_words"])))

    print("\npaired-only word metrics (segmentation loss excluded):")
    print("  %-14s %6s %6s %8s %6s" % ("profile", "pairs", "exact", "CER", "<=1"))
    for prof in sorted(per):
        d = _avg(per[prof])
        print("  %-14s %6d %6s %8.2f %6s"
              % (prof, d["p_n"], _pct(d["p_exact"]), d["p_cer"], _pct(d["p_w1"])))
    print("  %-14s %6d %6s %8.2f %6s"
          % ("ALL", d["p_n"], _pct(d["p_exact"]), d["p_cer"], _pct(d["p_w1"])))

    print("\nsegmentation health (words found vs words in the label):")
    print("  %-14s %7s %7s %7s %7s %7s %7s"
          % ("profile", "label_w", "found_w", "matched", "over", "under", "zero"))
    for prof in sorted(per):
        d = _avg(per[prof])
        print("  %-14s %7d %7d %7d %7d %7d %7d"
              % (prof, d["n_lab_words"], d["n_found_words"], d["match_lines"],
                 d["over_lines"], d["under_lines"], d["zero_lines"]))
    d = _avg(tot)
    print("  %-14s %7d %7d %7d %7d %7d %7d"
          % ("ALL", d["n_lab_words"], d["n_found_words"], d["match_lines"],
             d["over_lines"], d["under_lines"], d["zero_lines"]))
    print("  split excess words: %d over-split, %d under-split"
          % (tot["total_over"], tot["total_under"]))

    print("\npaired discordance (same crops; line = all words read exactly):")
    b, c = tot["b_word_ok_line_bad"], tot["c_line_ok_word_bad"]
    print("  both correct      %d" % tot["both_ok"])
    print("  word ok/line bad  %d   <- word reading rescues these" % b)
    print("  line ok/word bad  %d   <- word reading loses these" % c)
    print("  both wrong        %d" % tot["both_bad"])
    if b + c:
        print("  McNemar chi2 (continuity-corrected) = %.2f  (b=%d, c=%d)"
              % ((abs(b - c) - 1) ** 2 / max(1, b + c), b, c))

    print("\nlexicon: %d tokens from data on disk (%s)"
          % (len(lex_tokens), lex))
    if misses:
        print("\nfirst misses (kind, ground truth -> read):")
        for kind, ref, got in misses:
            print("  %-4s %-30s -> %s" % (kind, ref[:30], got[:40]))

    if a.out:
        import json
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({
            "split": a.split, "ckpt": str(a.ckpt or EO.CKPT),
            "per_profile": {p: _avg(per[p]) for p in per},
            "overall": _avg(tot),
            "lexicon_tokens": len(lex_tokens),
        }, indent=2) + "\n")
        print("\nwrote " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
