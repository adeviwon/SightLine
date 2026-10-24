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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--beams", default="4,8")
    ap.add_argument("--alphas", default="0.25,0.5,1.0")
    ap.add_argument("--kinds", default="word,both")
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

    for kind, b, alpha in itertools.product(kinds, beams, alphas):
        lm = AB.load_lm(kind, str(char_path), str(word_path))
        if lm is None:
            continue
        t1 = time.time()
        preds = [CD.beam_decode(lp, beam=b, lm=lm, alpha=alpha) for lp in lps]
        dt = time.time() - t1
        bits = CD.exact_match_bits(preds, refs)
        pc = CD.pair_counts(gbits, bits)
        cells.append({"kind": kind, "beam": b, "alpha": alpha,
                      "exact": sum(bits), "exact_rate": sum(bits) / n,
                      "cer": CD.cer_aggregate(preds, refs),
                      "decode_s": dt,
                      "p_vs_greedy": CD.mcnemar_exact(pc["variant_only"],
                                                      pc["greedy_only"]),
                      "changed": sum(1 for p, q in zip(preds, greedy) if p != q),
                      **pc})
        print("  %-10s beam=%-2d alpha=%-4.2f exact %6.2f%% CER %.4f (+%d/-%d vs "
              "greedy, p=%.4f, %.0fs)"
              % (kind, b, alpha, 100.0 * sum(bits) / n, cells[-1]["cer"],
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
