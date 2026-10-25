"""
eval_decode_sweep.py — pick the word-LM's hyper-parameters on held-out crops.

    bash run.sh py ml/src/eval_decode_sweep.py --limit 1000 --out artifacts/decode_sweep.json

ONE FORWARD PASS, EVERY DECODER SCORED ON THE SAME CROPS
-------------------------------------------------------
Greedy is decoded once; then every (LM kind, beam width, alpha) cell is decoded
from the SAME logits, so each cell's McNemar table against greedy is paired by
construction and the cells are comparable to each other. A sweep whose cells were
measured on different crops would rank noise.

WHAT DECIDES THE WINNER
-----------------------
Exact match (the product metric) first, CER as the tie-break and the instrument:
a cell that wins exact match while raising CER is fixing whole fields and losing
characters elsewhere, which for this product is still a win but must be visible
in the report rather than hidden behind one number.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as B      # noqa: E402
import ctc_decode as CD        # noqa: E402
import eval_decode_ab as AB    # noqa: E402
import realdata as R           # noqa: E402
import render_corpus as RC     # noqa: E402
import build_lm as BL          # noqa: E402
import word_lm as WL           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_weighted(path):
    """`word\tcount` (data/lexicon_counts.tsv) or one-word-per-line."""
    pairs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        pairs.append((parts[0], float(parts[1])) if len(parts) > 1 else (parts[0], 1.0))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--beams", default="4,8")
    ap.add_argument("--alphas", default="0.25,0.5,1.0")
    ap.add_argument("--kinds", default="word,both")
    # The word-LM's own knobs. `--lexicons counts` swaps the plain SET lexicon for
    # the frequency-weighted one (data/lexicon_counts.tsv, mined TRAIN-only), and
    # `--betas` drives the word-insertion bonus. All cells decode the SAME logits,
    # so every comparison is paired by construction.
    ap.add_argument("--lexicons", default="plain",
                    help="plain (data/lexicon.txt) and/or counts (data/lexicon_counts.tsv)")
    ap.add_argument("--betas", default="0.0")
    ap.add_argument("--penalties", default="", help="blank = the module default")
    ap.add_argument("--digit-penalties", default="", help="blank = the module default")
    # context and backoff: the two levers the plain trie cannot express
    ap.add_argument("--bigram-weights", default="0.0")
    ap.add_argument("--backoff-weights", default="0.0")
    ap.add_argument("--lm-gates", default="0.0",
                    help="per-frame LM weighting: alpha_t = alpha*(1 - gate*p_max)")
    ap.add_argument("--bigrams", default=str(ROOT / "data" / "word_bigrams.json"))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.split == "train":
        print("refusing to score on the TRAIN split", file=sys.stderr)
        return 2

    import torch
    torch.set_num_threads(max(1, a.threads))

    data = RC.load(split=a.split)
    n = min(a.limit or data["crops"].shape[0], data["crops"].shape[0])
    from PIL import Image
    arrays = [B.crop_to_array(Image.fromarray(np.asarray(c, np.uint8)))
              for c in data["crops"][:n]]
    refs = [str(x) for x in data["labels"][:n]]
    model, ckpt = AB.load_model(a.ckpt or None)

    t0 = time.time()
    lps = AB.log_probs_for(model, arrays)
    t_fwd = time.time() - t0

    word_path = Path(__file__).resolve().parents[2] / "app" / "data" / "word_trie.json"
    char_path = Path(__file__).resolve().parents[2] / "artifacts" / "char_lm.json"
    if not word_path.exists():
        print("no word trie at %s -- run `bash run.sh py ml/src/word_lm.py --export`"
              % word_path, file=sys.stderr)
        return 1

    greedy = [R.decode_greedy(lp.argmax(-1)) for lp in lps]
    gbits = CD.exact_match_bits(greedy, refs)
    g_cer = CD.cer_aggregate(greedy, refs)
    print("split=%s n=%d ckpt=%s forward %.1fs"
          % (a.split, n, Path(ckpt).name, t_fwd))
    print("greedy: exact %d/%d (%.2f%%) CER %.4f"
          % (sum(gbits), n, 100.0 * sum(gbits) / n, g_cer))

    cells = []
    beams = [int(b) for b in a.beams.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    kinds = [k for k in a.kinds.split(",") if k]

    # beam without an LM: the control that says whether the LM or the beam did it
    for b in beams:
        preds = [CD.beam_decode(lp, beam=b) for lp in lps]
        bits = CD.exact_match_bits(preds, refs)
        pc = CD.pair_counts(gbits, bits)
        cells.append({"kind": "none", "beam": b, "alpha": 0.0,
                      "exact": sum(bits), "exact_rate": sum(bits) / n,
                      "cer": CD.cer_aggregate(preds, refs),
                      "p_vs_greedy": CD.mcnemar_exact(pc["variant_only"],
                                                      pc["greedy_only"]),
                      **pc})

    words_path = {"plain": ROOT / "data" / "lexicon.txt",
                  "counts": ROOT / "data" / "lexicon_counts.tsv"}
    bigram_w = [float(x) for x in a.bigram_weights.split(",") if x != ""]
    backoff_w = [float(x) for x in a.backoff_weights.split(",") if x != ""]
    gates = [float(x) for x in a.lm_gates.split(",") if x != ""]
    bigrams_plain = {}
    if any(bigram_w) and Path(a.bigrams).exists():
        bigrams_plain = {tuple(k.split("\u0000")): v
                         for k, v in json.loads(Path(a.bigrams).read_text()).items()}
        print("bigrams: %d pairs from %s" % (len(bigrams_plain), Path(a.bigrams).name))
    char_lm_plain = BL.load_model(char_path) if any(backoff_w) and char_path.exists() else None
    lexicons = [x for x in a.lexicons.split(",") if x]
    betas = [float(x) for x in a.betas.split(",") if x != ""]
    penalties = [None if a.penalties == "" else float(x)
                 for x in a.penalties.split(",")]
    digit_penalties = [None if a.digit_penalties == "" else float(x)
                       for x in a.digit_penalties.split(",")]

    for kind, b, alpha, lex, beta, pen, dpen, bw, ow, gate in itertools.product(
            kinds, beams, alphas, lexicons, betas, penalties, digit_penalties,
            bigram_w, backoff_w, gates):
        if kind == "word" or kind == "both":
            src = words_path[lex]
            if not src.exists():
                print("  missing %s — run `bash run.sh py ml/src/word_counts.py`"
                      % src, file=sys.stderr)
                continue
            kw = {}
            if pen is not None:
                kw["penalty"] = pen
            if dpen is not None:
                kw["digit_penalty"] = dpen
            kw["beta"] = beta
            if bw:
                kw["bigrams"], kw["bigram_w"] = bigrams_plain, bw
            if ow and char_lm_plain is not None:
                kw["backoff"], kw["backoff_w"] = char_lm_plain, ow
            lm = WL.WordLM(load_weighted(src), **kw)
        else:
            lm = AB.load_lm(kind, str(char_path), str(word_path))
        if lm is None:
            continue
        t1 = time.time()
        preds = [CD.beam_decode(lp, beam=b, lm=lm, alpha=alpha, lm_gate=gate)
                 for lp in lps]
        dt = time.time() - t1
        bits = CD.exact_match_bits(preds, refs)
        pc = CD.pair_counts(gbits, bits)
        cells.append({"kind": kind, "beam": b, "alpha": alpha,
                      "lexicon": lex, "beta": beta,
                      "bigram_w": bw, "backoff_w": ow, "lm_gate": gate,
                      "penalty": pen if pen is not None else WL.PENALTY,
                      "digit_penalty": (dpen if dpen is not None
                                        else WL.DIGIT_PENALTY),
                      "exact": sum(bits), "exact_rate": sum(bits) / n,
                      "cer": CD.cer_aggregate(preds, refs),
                      "decode_s": dt,
                      "p_vs_greedy": CD.mcnemar_exact(pc["variant_only"],
                                                      pc["greedy_only"]),
                      "changed": sum(1 for p, q in zip(preds, greedy) if p != q),
                      **pc})
        print("  %-5s %-6s beam=%-2d a=%-4.2f beta=%-4.2f bg=%-4.2f bo=%-4.2f exact %6.2f%% "
              "CER %.4f (+%d/-%d vs greedy, p=%.4f, %.0fs)"
              % (kind, lex, b, alpha, beta, bw, ow, 100.0 * sum(bits) / n, cells[-1]["cer"],
                 pc["variant_only"], pc["greedy_only"], cells[-1]["p_vs_greedy"],
                 dt), flush=True)

    best = min(cells, key=lambda c: (-c["exact"], c["cer"]))
    report = {"split": a.split, "n": n, "ckpt": ckpt,
              "greedy": {"exact": sum(gbits), "exact_rate": sum(gbits) / n,
                         "cer": g_cer},
              "cells": cells, "best": best,
              "note": ("winner by exact match, CER as tie-break; every cell decoded "
                       "from the SAME logits, paired McNemar against greedy")}
    print("\nBEST: kind=%s beam=%d alpha=%.2f -> exact %.2f%% (greedy %.2f%%), CER %.4f"
          % (best["kind"], best["beam"], best["alpha"],
             100.0 * best["exact_rate"], 100.0 * sum(gbits) / n, best["cer"]))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1))
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
