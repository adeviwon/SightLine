"""
eval_decode_confirm.py — confirm a decoder choice on the FULL held-out split,
with every pair tested against every other pair.

    bash run.sh py ml/src/eval_decode_confirm.py --limit 0 --configs shipped,counts,bigram

WHY A SECOND HARNESS, WHEN eval_decode_sweep.py EXISTS
------------------------------------------------------
The sweep ranks cells against GREEDY only. Every cell it prints shares the greedy
baseline, so the table looks decisive while telling you nothing about whether cell
A beats cell B — and the differences at the top are 1-2 crops out of 800, which is
the noise regime `sightline-ml-validation-gates` §15 is about. A candidate is only
worth shipping if it beats the CURRENT SHIPPED CONFIG, paired, on the full split.

Output: per config exact/CER, then a pairwise matrix of exact McNemar (b = A right
and B wrong, c = B right and A wrong) with its p-value, so "no better than what we
ship" is a measurement rather than an omission.
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
import build_lm as BL          # noqa: E402
import ctc_decode as CD        # noqa: E402
import eval_decode_ab as AB    # noqa: E402
import realdata as R           # noqa: E402
import render_corpus as RC     # noqa: E402
import word_lm as WL           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# name -> how to build the LM. `shipped` must stay EXACTLY the app's config
# (app/js/recognizer.js WORD_BEAM / WORD_ALPHA + app/data/word_trie.json).
CONFIGS = {
    "shipped": dict(lexicon="plain", beam=4, alpha=0.25),
    "counts": dict(lexicon="counts", beam=4, alpha=0.25),
    "bigram40": dict(lexicon="counts", beam=4, alpha=0.25, bigram_w=0.4),
    "bigram80": dict(lexicon="counts", beam=4, alpha=0.25, bigram_w=0.8),
    "beta75": dict(lexicon="plain", beam=4, alpha=0.25, beta=0.75),
    "beam8": dict(lexicon="plain", beam=8, alpha=0.25),
    # alpha was chosen on a 1,000-crop sweep; these check it on the whole split
    "alpha20": dict(lexicon="plain", beam=4, alpha=0.20),
    "alpha30": dict(lexicon="plain", beam=4, alpha=0.30),
    "alpha35": dict(lexicon="plain", beam=4, alpha=0.35),
}
LEXICONS = {"plain": ROOT / "data" / "lexicon.txt",
            "counts": ROOT / "data" / "lexicon_counts.tsv"}


def load_weighted(path):
    pairs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        pairs.append((parts[0], float(parts[1])) if len(parts) > 1 else (parts[0], 1.0))
    return pairs


def build_lm(cfg, bigrams):
    kw = {"beta": cfg.get("beta", WL.BETA)}
    if cfg.get("bigram_w"):
        kw["bigrams"], kw["bigram_w"] = bigrams, cfg["bigram_w"]
    return WL.WordLM(load_weighted(LEXICONS[cfg["lexicon"]]), **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--configs", default="shipped,counts,bigram40,bigram80,beta75,beam8")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="artifacts/decode_confirm.json")
    a = ap.parse_args()

    if a.split == "train":
        print("refusing to score on the TRAIN split", file=sys.stderr)
        return 2

    import torch
    torch.set_num_threads(max(1, a.threads))
    from PIL import Image

    data = RC.load(split=a.split)
    n = a.limit or data["crops"].shape[0]
    n = min(n, data["crops"].shape[0])
    arrays = [B.crop_to_array(Image.fromarray(np.asarray(c, np.uint8)))
              for c in data["crops"][:n]]
    refs = [str(x) for x in data["labels"][:n]]
    model, ckpt = AB.load_model(a.ckpt or None)

    t0 = time.time()
    lps = AB.log_probs_for(model, arrays)
    print("split=%s n=%d ckpt=%s forward %.0fs"
          % (a.split, n, Path(ckpt).name, time.time() - t0), flush=True)

    bi_path = ROOT / "data" / "word_bigrams.json"
    bigrams = {}
    if bi_path.exists():
        bigrams = {tuple(k.split("\u0000")): v
                   for k, v in json.loads(bi_path.read_text()).items()}

    preds = {"greedy": [R.decode_greedy(lp.argmax(-1)) for lp in lps]}
    bits = {k: CD.exact_match_bits(v, refs) for k, v in preds.items()}
    rows = {}

    names = [x for x in a.configs.split(",") if x in CONFIGS]
    for name in names:
        cfg = CONFIGS[name]
        lm = build_lm(cfg, bigrams)
        t1 = time.time()
        p = [CD.beam_decode(lp, beam=cfg["beam"], lm=lm, alpha=cfg["alpha"])
             for lp in lps]
        dt = time.time() - t1
        preds[name] = p
        bits[name] = CD.exact_match_bits(p, refs)
        pc = CD.pair_counts(bits["greedy"], bits[name])
        rows[name] = {"config": cfg, "exact": int(sum(bits[name])),
                      "exact_rate": float(sum(bits[name])) / n,
                      "cer": float(CD.cer_aggregate(p, refs)),
                      "decode_s": dt, "vs_greedy": pc,
                      "p_vs_greedy": CD.mcnemar_exact(pc["variant_only"],
                                                      pc["greedy_only"])}
        print("  %-9s beam=%d alpha=%.2f beta=%.2f bigram=%.2f  exact %5.1f%% "
              "(%d/%d)  CER %.4f  +%d/-%d vs greedy p=%.4f  %.0fs"
              % (name, cfg["beam"], cfg["alpha"], cfg.get("beta", 0.0),
                 cfg.get("bigram_w", 0.0), 100.0 * rows[name]["exact_rate"],
                 int(sum(bits[name])), n, rows[name]["cer"],
                 pc["variant_only"], pc["greedy_only"], rows[name]["p_vs_greedy"], dt),
              flush=True)

    # ── pairwise: does any candidate beat the SHIPPED config? ──────────────────
    pairs = []
    print("\npairwise exact McNemar (b = row right & col wrong, c = col right & row wrong):")
    base = "greedy" if "greedy" in bits else names[0]
    for x, y in itertools.combinations(["greedy"] + names, 2):
        pc = CD.pair_counts(bits[x], bits[y])
        p = CD.mcnemar_exact(pc["variant_only"], pc["greedy_only"])
        # pair_counts(bits[x], bits[y]) counts "y only" as variant_only; name both
        pairs.append({"a": x, "b": y, "p": p,
                      "a_only": pc["greedy_only"], "b_only": pc["variant_only"]})
        print("  %-9s vs %-9s  +%d/-%d  p=%.4f%s"
              % (x, y, pc["greedy_only"], pc["variant_only"], p,
                 "   <-- SHIPPED comparison" if x == "shipped" or y == "shipped" else ""),
              flush=True)

    out = {"split": a.split, "n": n, "ckpt": ckpt, "configs": rows,
           "pairwise": pairs, "greedy_exact": int(sum(bits["greedy"])),
           "note": ("every config decoded from the SAME logits; pairwise exact "
                    "McNemar so 'no better than shipped' is measured, not assumed")}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print("wrote %s" % a.out)
    _ = base
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
