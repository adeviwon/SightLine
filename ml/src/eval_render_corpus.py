"""
eval_render_corpus.py — the recogniser's accuracy ON OUR OWN DOCUMENTS.

The number that matters for reliability, measured where the product actually
operates. Crop-level on purpose: every crop is the true ink box of a line we
rendered, labelled by the text we drew, so this isolates the RECOGNISER from the
line detector. (End-to-end, detector included, is `bash run.sh bench-pipeline`.)

Reports, per capture profile and overall:
  * exact match  — the product metric: a field is right or it is wrong, and one
                   wrong character in "40218877" is a wrong field
  * CER          — the instrument panel (see eval_ocr.py's header)
  * within 1 edit — how close the model is when it is wrong: the difference
                   between "needs a better decoder" and "needs more training"

    bash run.sh py ml/src/render_corpus.py --pages 120     # build the corpus
    bash run.sh eval-render                                # score it
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as B     # noqa: E402
import eval_ocr as EO         # noqa: E402
import realdata as R          # noqa: E402
import render_corpus as RC    # noqa: E402
from ocr_model import build   # noqa: E402


def load_model(ckpt=None):
    """Load the checkpoint, refusing a file that changed while it was being read.

    THE FAILURE THIS PREVENTS (measured): loading models/ocr/crnn.pt while a
    training run was mid-save produced GARBAGE weights with no exception — a run
    reported 0/300 exact at CER 0.976 where the same file minutes earlier gave
    11/300 at CER 0.590. `torch.save` is not atomic, so a partial file is a valid
    zip with plausible tensors. Any harness that measures while training runs must
    stat the file either side of the load and refuse the result.
    """
    import os
    import torch
    path = ckpt or str(EO.CKPT)
    before = os.stat(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    after = os.stat(path)
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(
            "%s changed while it was being loaded (training is writing it): the "
            "weights in memory are a torn read and any number from them is a lie. "
            "Re-run when the training job is idle, or pass --ckpt <frozen copy>."
            % path)
    model = build()
    model.load_state_dict(ck["model"])
    model.eval()
    return model, "cpu"


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_split(data, model, device, limit=None, show_misses=6):
    per = {}
    misses = []
    n_calls = 0
    n_empty = 0
    for i, (crop, label, prof) in enumerate(zip(data["crops"], data["labels"],
                                                data["profile"])):
        if limit and i >= limit:
            break
        arr = B.crop_to_array(Image.fromarray(np.asarray(crop, dtype=np.uint8)))
        txt, _ = EO.recognise(model, arr, device)
        n_calls += 1
        if not str(txt).strip():
            n_empty += 1
        got, ref = R.norm_label(txt), R.norm_label(label)
        e = 1.0 if got == ref else 0.0
        c = R.cer(got, ref)
        w1 = 1.0 if lev(got, ref) <= 1 else 0.0
        row = per.setdefault(prof, {"n": 0, "exact": 0.0, "cer": 0.0, "w1": 0.0})
        row["n"] += 1
        row["exact"] += e
        row["cer"] += c
        row["w1"] += w1
        if not e and len(misses) < show_misses:
            misses.append((ref, got))
    # A checkpoint that reads NOTHING is not a weak model, it is a broken or
    # mid-training file (measured: a half-saved crnn.pt loaded fine and emitted
    # blanks for every crop, giving CER ~1.0 and 0% exact — indistinguishable from
    # a terrible model). Refuse the measurement instead of reporting it.
    if n_calls >= 20 and n_empty == n_calls:
        raise SystemExit(
            "every one of the %d crops read as EMPTY: this checkpoint is not a "
            "measured model. It is mid-training or corrupt. Re-run against a "
            "frozen copy (--ckpt models/ocr/crnn_before_render.pt) or wait for "
            "training to finish." % n_calls)

    for row in per.values():
        n = max(1, row["n"])
        row["exact"] /= n
        row["cer"] /= n
        row["w1"] /= n
    return per, misses


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
    model, device = load_model(a.ckpt or None)
    per, misses = score_split(data, model, device, limit=a.limit or None)

    print("=" * 74)
    print("CRNN on OUR OWN RENDERED DOCUMENTS — split=%s, %d crops, checkpoint %s"
          % (a.split, n if not a.limit else min(n, a.limit),
             Path(a.ckpt or EO.CKPT).name))
    print("=" * 74)
    print("%-16s %6s %10s %8s %10s" % ("profile", "n", "exact", "CER", "<=1 edit"))
    tot = {"n": 0, "exact": 0.0, "cer": 0.0, "w1": 0.0}
    for prof in sorted(per):
        r = per[prof]
        print("%-16s %6d %9.1f%% %7.2f %9.1f%%"
              % (prof, r["n"], 100 * r["exact"], r["cer"], 100 * r["w1"]))
        for k in tot:
            tot[k] += r[k] * (r["n"] if k != "n" else 1)
    n_tot = max(1, tot["n"])
    print("-" * 74)
    print("%-16s %6d %9.1f%% %7.2f %9.1f%%"
          % ("ALL", tot["n"], 100 * tot["exact"] / n_tot, tot["cer"] / n_tot,
             100 * tot["w1"] / n_tot))
    if misses:
        print("\nfirst misses (ground truth -> read):")
        for ref, got in misses:
            print("  %-34s -> %s" % (ref[:34], got[:34]))

    if a.out:
        import json
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"split": a.split, "ckpt": str(a.ckpt or EO.CKPT), "per_profile": per,
             "overall": {k: (tot[k] / n_tot if k != "n" else tot["n"]) for k in tot}},
            indent=2) + "\n")
        print("\nwrote " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
