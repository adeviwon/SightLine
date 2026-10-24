"""
eval_decode_ab.py — does beam search (with or without an LM) read held-out
crops better than greedy?

    bash run.sh py ml/src/eval_decode_ab.py --split test

ONE FORWARD PASS PER CROP, THREE DECODINGS OF IT
------------------------------------------------
Greedy, beam and beam+LM all read the SAME logits for the same crop, so the
comparison is paired by construction. That is not a convenience: an unpaired
comparison of two accuracies on the same data throws away the pairing and loses
most of the power to detect a real difference, which is exactly the difference
between "beam is 2 points better" and "beam is 2 points better on a sample this
size".

WHAT IS REPORTED, AND WHAT IS NOT CLAIMED
-----------------------------------------
Per variant: exact match with a Wilson 95% interval, and character-weighted CER.
Greedy vs the best beam variant: the discordant counts (beam-right/greedy-wrong
against greedy-right/beam-wrong) and the two-sided exact McNemar p-value.

Exact match is the product metric -- a field is right or it is wrong, and one
wrong character in "40218877" is a wrong field. CER is the instrument panel: a
falling CER with a flat exact match means the decoder is fixing most characters
and still getting one wrong per line.

If the best beam variant is within its interval of greedy and p is large, the
honest conclusion is that this sample cannot tell the two apart, NOT that beam
search failed -- and the size of the effect that would have been detectable is
what `--split test` on 3270 crops can and cannot resolve.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as B      # noqa: E402
import build_lm as BL          # noqa: E402
import ctc_decode as CD        # noqa: E402
import word_lm as WL           # noqa: E402
import eval_ocr as EO          # noqa: E402
import realdata as R           # noqa: E402
import render_corpus as RC     # noqa: E402
from ocr_model import build    # noqa: E402


def load_model(ckpt=None):
    """
    Load the CRNN, and REFUSE to score a checkpoint that changed while loading.

    This is not defensive padding. The recogniser is retrained in place
    (`ml/tools/retrain_rendered.sh` writes models/ocr/crnn.pt whenever validation
    improves), and an A/B run takes minutes. Observed for real: a run that loaded
    while the retrain was mid-save read a PARTIALLY WRITTEN checkpoint -- torch
    raised nothing, the weights were garbage, and the report came out 0/300 exact
    at CER 0.976 instead of 11/300 at CER 0.590. A number like that looks like a
    decoder failure and would have been written up as one. A torn read must fail
    loudly, so the stat is taken either side of torch.load and any change is fatal.
    """
    import torch
    path = str(ckpt or EO.CKPT)

    def _stamp():
        st = os.stat(path)
        return (st.st_size, st.st_mtime_ns)

    before = _stamp()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    after = _stamp()
    if before != after:
        raise RuntimeError(
            "%s changed while it was being loaded (%s -> %s). A retrain is "
            "writing this checkpoint right now; scoring a half-written file "
            "produces a meaningless accuracy. Re-run when the retrain is done."
            % (path, before, after))
    model = build(ck.get("num_classes", R.NUM_CLASSES))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, path


def log_probs_for(model, arrays, batch=16):
    """
    One batched forward pass per distinct crop width -> log-softmax per crop.

    Grouping by width is what makes a full test-split evaluation finish in
    minutes rather than the ~14 minutes a per-crop forward takes on this CPU --
    and the CPU is shared with a training job, so the win matters twice. The
    logits are identical to one-at-a-time inference: eval mode, GroupNorm
    (no batch statistics), and no padding inside a group because every crop in a
    group has the same width.
    """
    import torch
    out = [None] * len(arrays)
    by_w = {}
    for i, a in enumerate(arrays):
        by_w.setdefault(a.shape[1], []).append(i)
    for w, idxs in by_w.items():
        for j in range(0, len(idxs), batch):
            chunk = idxs[j:j + batch]
            x = np.stack([arrays[i] for i in chunk]).astype(np.float32)
            x = (x / 255.0 - 0.5) / 0.5
            with torch.no_grad():
                lg = model(torch.from_numpy(x[:, None]))
            lp = torch.log_softmax(lg, dim=-1).cpu().numpy()
            for k, i in enumerate(chunk):
                out[i] = lp[k]
    return out


class CombinedLM:
    """Shallow fusion over two LMs: the beam asks ONCE and gets the sum, so the
    two sources cannot be weighted inconsistently inside one search."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def logp(self, prefix, ch):
        return self.a.logp(prefix, ch) + self.b.logp(prefix, ch)


def load_lm(kind, char_path, word_path=None):
    """`kind` in {none, char, word, both} -> an LM object with .logp, or None."""
    char_lm = None
    if kind in ("char", "both") and Path(char_path).exists():
        char_lm = BL.load_model(char_path)
    word_lm = None
    if kind in ("word", "both"):
        wp = word_path or (Path(__file__).resolve().parents[2] / "app" / "data"
                           / "word_trie.json")
        if Path(wp).exists():
            word_lm = WL.load_trie_json(wp)
    if kind == "both":
        return CombinedLM(char_lm, word_lm) if char_lm and word_lm else (char_lm or word_lm)
    return char_lm or word_lm


def score(refs, lps, beam, lm, alpha, tag="LM"):
    """Every decoder over the SAME crops -> one predictions list per variant."""
    variants = {"greedy": [R.decode_greedy(lp.argmax(-1)) for lp in lps],
                "beam%d" % beam: [CD.beam_decode(lp, beam=beam) for lp in lps]}
    if lm is not None:
        variants["beam%d+%s" % (beam, tag)] = [
            CD.beam_decode(lp, beam=beam, lm=lm, alpha=alpha) for lp in lps]
    rows = {}
    for name, preds_ in variants.items():
        bits = CD.exact_match_bits(preds_, refs)
        k = sum(bits)
        lo, hi = CD.wilson(k, len(bits))
        rows[name] = {"n": len(bits), "exact": k, "exact_rate": k / max(len(bits), 1),
                      "ci": [lo, hi], "cer": CD.cer_aggregate(preds_, refs),
                      "bits": bits, "preds": preds_}
    return rows


def best_variant(rows):
    """The best beam variant by exact match, ties broken by CER then by name."""
    cand = [n for n in rows if n != "greedy"]
    return min(cand, key=lambda n: (-rows[n]["exact"], rows[n]["cer"], n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    help="render_corpus split to score on (never 'train')")
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="LM weight for shallow fusion (0 disables the LM)")
    ap.add_argument("--lm", default=str(BL.DEFAULT_OUT))
    ap.add_argument("--lm-kind", default="char", choices=["char", "word", "both"],
                    help="char = the existing character n-gram LM, word = the "
                         "lexicon-trie word LM, both = shallow fusion of the two")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2,
                    help="keep low: a training job owns the other cores")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.split == "train":
        print("refusing to score on the TRAIN split: that number is memorisation",
              file=sys.stderr)
        return 2

    import torch
    torch.set_num_threads(max(1, a.threads))

    data = RC.load(split=a.split)
    n = data["crops"].shape[0]
    if a.limit:
        n = min(n, a.limit)
    if not n:
        print("no crops in split %r: build the corpus first "
              "(bash run.sh render-corpus --pages 400)" % a.split, file=sys.stderr)
        return 1

    lm = load_lm(a.lm_kind, a.lm) if a.alpha > 0 else None
    if a.alpha > 0 and lm is None:
        print("note: no LM for kind=%s -- reporting greedy vs beam only"
              % a.lm_kind, file=sys.stderr)
    lm_note = "alpha=%.2f (LM %s, kind=%s)" % (
        a.alpha, "on" if lm else "off", a.lm_kind)

    print("preparing %d %s crops ..." % (n, a.split), flush=True)
    from PIL import Image
    arrays = [B.crop_to_array(Image.fromarray(np.asarray(c, np.uint8)))
              for c in data["crops"][:n]]
    refs = [str(x) for x in data["labels"][:n]]
    model, ckpt = load_model(a.ckpt or None)

    t0 = time.time()
    lps = log_probs_for(model, arrays)
    t_fwd = time.time() - t0
    t0 = time.time()
    tag = {"char": "charLM", "word": "wordLM", "both": "char+wordLM"}[a.lm_kind]
    rows = score(refs, lps, a.beam, lm, a.alpha, tag=tag)
    t_dec = time.time() - t0

    lm_key = "beam%d+%s" % (a.beam, tag)
    order = ["greedy", "beam%d" % a.beam] + ([lm_key] if lm_key in rows else [])
    bench = best_variant(rows)

    print("=" * 78)
    print("CTC DECODER A/B — split=%s, %d crops, beam=%d, %s"
          % (a.split, n, a.beam, lm_note))
    print("checkpoint %s (%s, %.0f MB)"
          % (ckpt, time.strftime("%Y-%m-%d %H:%M", time.localtime(Path(ckpt).stat().st_mtime)),
             Path(ckpt).stat().st_size / 1e6))
    print("forward %.1fs (batched), decode %.1fs, SAME crops for every variant"
          % (t_fwd, t_dec))
    print("=" * 78)
    print("%-14s %6s %9s %-22s %8s" % ("variant", "right", "exact", "Wilson 95% CI", "CER"))
    for name in order:
        r = rows[name]
        print("%-14s %6d %8.2f%% [%6.2f%%, %6.2f%%]   %8.4f"
              % (name, r["exact"], 100 * r["exact_rate"], 100 * r["ci"][0],
                 100 * r["ci"][1], r["cer"]))

    g = rows["greedy"]
    b = rows[bench]
    pc = CD.pair_counts(g["bits"], b["bits"])
    p = CD.mcnemar_exact(pc["variant_only"], pc["greedy_only"])
    print("-" * 78)
    print("PAIRED comparison (exact McNemar), greedy vs %s, same %d crops:" % (bench, n))
    print("  both right            %6d" % pc["both_right"])
    print("  %-14s %6d   <- beam right, greedy wrong" % (bench + " only", pc["variant_only"]))
    print("  greedy only           %6d   <- greedy right, beam wrong" % pc["greedy_only"])
    print("  both wrong            %6d" % pc["both_wrong"])
    delta = 100 * (b["exact_rate"] - g["exact_rate"])
    print("  exact match delta     %+6.2f pts   (%d -> %d crops)"
          % (delta, g["exact"], b["exact"]))
    print("  McNemar exact two-sided p = %.4g   (discordant pairs n=%d)"
          % (p, pc["variant_only"] + pc["greedy_only"]))
    if p >= 0.05:
        print("  -> NOT significant at 0.05. On this sample the two decoders are"
              " indistinguishable;")
        print("     report the interval, not the point estimate.")
    elif delta > 0:
        print("  -> beam is better on this sample (p < 0.05).")
    else:
        print("  -> beam is WORSE on this sample (p < 0.05). Keep greedy.")

    def show(name, k=6):
        print("\nfirst %s disagreements (ground truth | greedy -> %s):" % (k, name))
        shown = 0
        for ref, gp, bp in zip(refs, g["preds"], b["preds"]):
            if CD.exact_match_bits([gp], [ref])[0] == 0 and \
               CD.exact_match_bits([bp], [ref])[0] == 1:
                print("  %-30s | %-30s -> %s" % (R.norm_label(ref)[:30],
                                                  R.norm_label(gp)[:30], R.norm_label(bp)[:30]))
                shown += 1
                if shown >= k:
                    break
        if not shown:
            print("  (none)")
    show(bench)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({
            "split": a.split, "n": n, "beam": a.beam, "alpha": a.alpha,
            "lm": a.lm if lm else None, "ckpt": ckpt,
            "variants": {k: {kk: vv for kk, vv in v.items()
                             if kk not in ("bits", "preds")} for k, v in rows.items()},
            "paired": {"best": bench, **pc, "p_exact_mcnemar": p,
                       "delta_pts": delta},
        }, indent=2) + "\n")
        print("\nwrote " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
