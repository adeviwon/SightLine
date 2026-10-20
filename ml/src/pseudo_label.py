"""
pseudo_label.py — self-training on UNLABELLED crops, with the precision MEASURED
before anything is trusted to it.

    bash run.sh py ml/src/pseudo_label.py --dry-run --limit 200     # report only
    bash run.sh py ml/src/pseudo_label.py --yes --threshold 0.90    # write a cache

WHAT IT DOES
    Runs a frozen recogniser over the TRAIN split of a corpus, keeps the
    predictions it is confident about AND whose length is plausible, and writes
    the survivors in the trainer's own cache schema
    (`render_corpus.write_cache` -> images/widths/labels/variants/splits/page),
    so `train_ocr.py --data a.npz,b.npz` consumes them without a code change.

WHY THE DRY RUN IS THE DEFAULT
    The published failure mode of pseudo-labelling is CONFIRMATION BIAS: wrong
    high-confidence predictions are re-fed as ground truth and the model
    reinforces its own errors. The published guards are (i) a confidence
    threshold, (ii) a held-out validation set, (iii) a cap on the pseudo-label
    share of the batch mix. This tool supplies (i) and MEASURES (ii) where truth
    exists. It cannot supply (iii) — that is a training-mix decision — so it
    writes nothing until you pass --yes, and it prints the precision estimate
    first so the threshold is chosen from a number rather than from taste.

WHY IT REFUSES THE TEST SPLIT
    Pseudo-labelling the evaluation split makes every later number
    self-confirming: the model is scored on strings it generated itself, so a
    "gain" measures agreement with its own mistakes and the held-out set stops
    being held out. The val split is excluded for the same reason — it selects
    the checkpoint. Only `train` is read, and the guard is an assertion, not a
    convention.

WHERE THE PRECISION ESTIMATE COMES FROM
    This corpus's labels are DRAWN, not detected: `capture.render_document`
    places line i at a computable y, so the label is what was printed and no
    detector error can leak into it (`render_corpus.py`). That is exactly what
    makes it the right calibration set for a threshold that will later be applied
    to photographs, where no word-level truth exists.

    What that number does NOT transfer to: a photographed page. The precision
    here is measured on the render distribution, so it is an UPPER BOUND on the
    precision you will get on real captures, where the model is weaker. Calibrate
    on renders, then treat the real-photo pass as unverified.

WHY THE EXISTING RENDERER IS THE WRONG PLACE TO USE THIS
    Read the plan in docs/18 before spending a retrain on this. For crops from
    our own renderer the exact label is FREE, so pseudo-labelling them is
    strictly worse than using the drawn labels. This tool earns its keep on data
    whose labels are absent or untrustworthy: the hand-photographed pages
    (positional transcript pairing is a measured negative — HANDOFF.md §3), and
    real captures in a distribution we cannot label. It does not fix a weak
    model by itself — it amplifies whatever the current checkpoint already knows.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as B             # noqa: E402
import eval_ocr as EO                 # noqa: E402
import eval_render_corpus as ER       # noqa: E402
import realdata as R                  # noqa: E402
import render_corpus as RC            # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "models" / "ocr" / "crnn_before_render.pt"
DEFAULT_OUT = ROOT / "data" / "pseudo" / "pseudo_train.npz"

# The longest label that is ever plausible for a single line of these documents.
# ocr_model.py's own note puts the worst case at 47 characters, so 48 is the cap
# rather than a tunable. A prediction past it is the decoder looping on padding.
MAX_LABEL_LEN = 48

# Length plausibility, MEASURED rather than guessed. `ocr_model.py` reports the
# dataset's median timesteps-per-character at 2.16, i.e. a typical crop emits
# chars ~= 0.46 * T, and only 0.4% of real samples fall below 1.0 (the CTC
# feasibility floor). A WIDE crop emitting a couple of characters is therefore a
# collapse, not a short word: at T=40 this floor asks for 6 characters, roughly a
# quarter of the typical read. It is deliberately permissive -- it is a
# plausibility gate, not an accuracy filter, and the confidence threshold does
# the discriminating.
MIN_LEN_FRAC = 0.15


def length_is_plausible(text, timesteps, min_len_frac=MIN_LEN_FRAC,
                        max_len=MAX_LABEL_LEN):
    """True when a prediction could plausibly be the text of this crop.

    Returns (ok, reason) so the drops are attributable instead of silent.
    """
    n = len(R.encode(text))
    if n == 0:
        return False, "empty"
    if n > max_len:
        return False, "too-long"
    if n > timesteps:
        # Greedy decode cannot exceed T non-blank timesteps, so this is only
        # reachable if a caller changes the decoder. Keep the check anyway: it is
        # the CTC-validity contract train_ocr.py filters on, and a target longer
        # than T makes the loss infinite.
        return False, "ctc-infeasible"
    floor = int(min_len_frac * max(1, timesteps))
    if n < floor:
        return False, "collapsed"
    return True, "kept"


def histogram(values):
    """Compact text histogram of a 1-D list of ints -> a list of lines."""
    if not values:
        return ["    (none)"]
    lo, hi = min(values), max(values)
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    out = []
    for v in range(lo, hi + 1):
        c = counts.get(v, 0)
        bar = "#" * max(1, int(round(40.0 * c / top))) if c else ""
        out.append("    %3d chars |%-40s %5d" % (v, bar, c))
    return out


def run(corpus_dir, ckpt, threshold, limit, min_len_frac):
    """Score the TRAIN split and return (kept, stats).

    `kept` is a dict in `render_corpus.write_cache`'s input shape; `stats` is the
    report that goes to stdout and the sidecar JSON.
    """
    # CPU IS SHARED. The default thread count is every core, and this box runs a
    # 4-thread training job: unbounded here measures a throughput that belongs to
    # the other job and slows it down in exchange. One thread is enough for a
    # single-crop-per-step pass and makes the runtime predictable. Override with
    # PSEUDO_LABEL_THREADS when the box is idle.
    torch.set_num_threads(int(os.environ.get("PSEUDO_LABEL_THREADS", "1")))

    # split="train" is the guard: render_corpus.load filters by it, so a test or
    # val crop cannot reach the halfway point.
    data = RC.load(corpus_dir, split="train")
    n_split = int(data["crops"].shape[0])
    if not n_split:
        raise SystemExit("no TRAIN crops in %s — build it first: "
                         "bash run.sh py ml/src/render_corpus.py --pages N"
                         % corpus_dir)
    if not np.all(data["split"] == "train"):
        raise SystemExit("refusing to run: a non-train crop reached the reader")

    model, device = ER.load_model(str(ckpt))

    idx = list(range(n_split if not limit else min(n_split, limit)))
    kept_crops, kept_labels, kept_doc, kept_prof = [], [], [], []
    lengths, dropped = [], {}
    n_scored = n_empty = 0
    # Two separate accumulators, deliberately. `*_all` is the base rate over
    # everything scanned (context); `*_kept` is the PRECISION OF THE SET WE WOULD
    # ACTUALLY TRAIN ON, which is the number the threshold has to be chosen from
    # and the only one that matters. Sharing one counter here reported the base
    # rate and called it precision -- a threshold that keeps 5% of crops would
    # then look no better than no threshold at all.
    n_exact_all = n_exact_kept = 0
    cer_all = cer_kept = 0.0
    misses = []

    for i in idx:
        crop = np.asarray(data["crops"][i], dtype=np.uint8)
        # The SAME transform training and eval_render_corpus use, so a
        # pseudo-labelled crop is geometrically identical to a real one and the
        # model cannot tell which it is looking at.
        arr = B.crop_to_array(Image.fromarray(crop))
        txt, conf = EO.recognise(model, arr, device)
        n_scored += 1
        if not str(txt).strip():
            n_empty += 1

        # Precision is only computable where the corpus carries a true label.
        # Every render does; a genuinely unlabelled corpus would report n/a
        # rather than a fiction.
        truth = str(data["labels"][i])
        got_n = want_n = ""
        ok = None
        if truth:
            got_n, want_n = R.norm_label(txt), R.norm_label(truth)
            ok = int(got_n == want_n)
            e = R.cer(got_n, want_n)
            n_exact_all += ok
            cer_all += e
            if not ok and len(misses) < 6:
                misses.append((want_n, got_n))

        if conf < threshold:
            dropped["low-confidence"] = dropped.get("low-confidence", 0) + 1
            continue
        ts = model.timesteps(int(arr.shape[1]))
        plausible, why = length_is_plausible(txt, ts, min_len_frac)
        if not plausible:
            dropped[why] = dropped.get(why, 0) + 1
            continue

        if ok is not None:
            n_exact_kept += ok
            cer_kept += R.cer(got_n, want_n)

        kept_crops.append(crop)
        kept_labels.append(txt.strip().upper())
        kept_doc.append(str(data["doc"][i]))
        kept_prof.append(str(data["profile"][i]))
        lengths.append(len(txt.strip()))

    crops_arr = np.empty(len(kept_crops), dtype=object)
    for i, c in enumerate(kept_crops):
        crops_arr[i] = c
    kept = {
        "crops": crops_arr,
        "labels": np.array(kept_labels, dtype=object),
        "doc": np.array(kept_doc, dtype=object),
        "profile": np.array(kept_prof, dtype=object),
        # Pseudo-labels are TRAINING signal, so every survivor is a train crop.
        # Writing a val/test string here would make it a label the checkpoint was
        # selected on -- the leak this module's header refuses.
        "split": np.array(["train"] * len(kept_labels), dtype=object),
    }

    n_kept = len(kept_labels)
    stats = {
        "corpus": str(corpus_dir),
        "ckpt": str(ckpt),
        "split_used": "train",
        "train_crops_available": n_split,
        "threshold": float(threshold),
        "min_len_frac": float(min_len_frac),
        "max_len": MAX_LABEL_LEN,
        "scanned": n_scored,
        "kept": n_kept,
        "kept_frac_of_scanned": (n_kept / n_scored) if n_scored else 0.0,
        "empty_predictions": n_empty,
        "dropped": dropped,
        "label_lengths": {"min": min(lengths) if lengths else None,
                          "max": max(lengths) if lengths else None,
                          "mean": (sum(lengths) / len(lengths)) if lengths else None},
        # PRECISION OF THE KEPT SET: the fraction of retained pseudo-labels that
        # match the drawn label exactly. This is the number the threshold is
        # chosen from.
        "precision_exact_match_kept": (n_exact_kept / n_kept) if n_kept else None,
        "mean_cer_kept": (cer_kept / n_kept) if n_kept else None,
        # Base rate over everything scanned, for context: precision of the kept
        # set must beat this or the threshold is not selecting anything.
        "exact_match_all_scanned": (n_exact_all / n_scored) if n_scored else None,
        "mean_cer_all_scanned": (cer_all / n_scored) if n_scored else None,
        "n_with_truth": n_scored,
    }
    return kept, stats, lengths, misses


def main():
    ap = argparse.ArgumentParser(
        description="Pseudo-label the TRAIN split of a corpus with a frozen "
                    "recogniser. Dry run unless --yes.")
    ap.add_argument("--corpus", default=str(RC.OUT),
                    help="corpus dir (holds crops.npz/index.json)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="frozen checkpoint; default is the pre-render copy so a "
                         "live training job cannot tear the read")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="mean emit-probability floor for a kept prediction")
    ap.add_argument("--limit", type=int, default=0, help="cap crops scanned (0=all)")
    ap.add_argument("--min-len-frac", type=float, default=MIN_LEN_FRAC)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--yes", action="store_true",
                    help="actually write the cache (default: dry run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="explicit no-op; the default, and it wins over --yes")
    a = ap.parse_args()

    write = bool(a.yes) and not bool(a.dry_run)
    ckpt = Path(a.ckpt)
    if not ckpt.exists():
        print("missing checkpoint %s\nrun:  bash run.sh train-ocr 40" % ckpt,
              file=sys.stderr)
        return 1

    kept, stats, lengths, misses = run(a.corpus, ckpt, a.threshold,
                                       a.limit, a.min_len_frac)

    print("=" * 74)
    print("PSEUDO-LABELLING — %s"
          % ("WRITE" if write else "DRY RUN (nothing will be written)"))
    print("=" * 74)
    print("  corpus      %s" % stats["corpus"])
    print("  checkpoint  %s" % ckpt)
    print("  split used  TRAIN only")
    print("  TEST split  NOT TOUCHED. Pseudo-labelling the held-out set scores the")
    print("              model on strings it produced itself, so any later gain")
    print("              would measure agreement with its own mistakes. Same for")
    print("              val: it selects the checkpoint. Only train is read.")
    print()
    print("  train crops available   %6d" % stats["train_crops_available"])
    print("  scanned                 %6d" % stats["scanned"])
    print("  threshold (conf >=)     %6.3f" % stats["threshold"])
    print("  KEPT                    %6d  (%.1f%% of scanned)"
          % (stats["kept"], 100 * stats["kept_frac_of_scanned"]))
    if stats["dropped"]:
        print("  dropped:")
        for k, v in sorted(stats["dropped"].items(), key=lambda kv: -kv[1]):
            print("    %-16s %6d" % (k, v))
    if stats["empty_predictions"]:
        print("  empty predictions       %6d" % stats["empty_predictions"])

    print()
    print("  label-length distribution (kept set):")
    for line in histogram(lengths):
        print(line)

    print()
    p = stats["precision_exact_match_kept"]
    print("  PRECISION (measured on render labels — see the module header):")
    base = stats["exact_match_all_scanned"]
    base_s = ("%6.1f%%" % (100 * base)) if base is not None else "   n/a"
    if p is not None:
        print("    KEPT set     %6.1f%% exact over %d kept"
              % (100 * p, stats["kept"]))
        print("    mean CER     %6.3f" % stats["mean_cer_kept"])
        print("    base rate    %s exact over %d scanned (no filtering)"
              % (base_s, stats["scanned"]))
        print("    NOTE: render labels are DRAWN, not detected, so this is an")
        print("    UPPER BOUND. On photographed pages the same threshold buys")
        print("    strictly worse precision — the model is weaker there.")
        if p < 0.95:
            print("    WARNING: under ~0.95, which semi-supervised work treats as")
            print("    the point below which pseudo-labels hurt more than help")
            print("    (docs/18 §2). Raise --threshold, or do not train on this.")
    elif stats["n_with_truth"]:
        # Zero kept is NOT "no truth available" — say which it is, or the two
        # read identically and the operator raises a threshold that is fine.
        print("    NO KEPT SET: no prediction cleared --threshold, so there is")
        print("    nothing to measure. For context, the unfiltered base rate is")
        print("    %s exact over %d scanned, mean CER %.3f."
              % (base_s, stats["scanned"], stats["mean_cer_all_scanned"]))
        print("    Either lower --threshold or use a checkpoint worth training on.")
    else:
        print("    n/a — the corpus carries no true labels")
    if misses:
        print("\n  first misses (true -> predicted):")
        for want, got in misses:
            print("    %-32s -> %s" % (want[:32], got[:32]))

    if not write:
        print("\nDRY RUN: wrote nothing. Re-run with --yes to write %s" % a.out)
        if stats["kept"] == 0:
            print("(and note that ZERO crops were kept — --threshold is too high "
                  "or the checkpoint reads nothing)")
        return 0

    out = Path(a.out)
    if stats["kept"] == 0:
        print("\nREFUSING TO WRITE: 0 crops survived --threshold, so the cache "
              "would be empty.\nLower --threshold (and read the precision "
              "estimate it prints), or fix the checkpoint.")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    written = RC.write_cache(kept, out_path=out)
    z = np.load(written, allow_pickle=True)
    stats["out"] = str(written)
    stats["out_crops"] = int(z["images"].shape[0])
    if int((z["splits"] != "train").sum()):
        raise SystemExit("refusing to write: a non-train split reached the cache")
    Path(str(written) + ".stats.json").write_text(
        json.dumps(stats, indent=2) + "\n")
    print("\nwrote %s  %d crops  maxW %d  all split=train"
          % (written, z["images"].shape[0], z["images"].shape[2]))
    print("wrote %s.stats.json" % written)
    print("\nnext:  bash run.sh train-ocr 40 --data artifacts/ocr_train.npz,%s"
          % written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
