"""
Build handheld training data by CONSENSUS ALIGNMENT.

THE PROBLEM THIS SOLVES
-----------------------
Pairing detected band k with transcript line k is only valid if the detector
finds exactly the text lines, in order. It does not. Measured across the 193
hand-held pages:

    bands/line ratio: p10 0.25   p50 0.79   p90 1.11
    pages within [0.85, 1.15]:  37%

On the first page the detector returns 22 bands for 15 transcript lines, and
band 2 is 149 px tall against a 20 px median -- two or three text lines merged
into one band by a shadow. Positional pairing therefore assigns "Long Beach,
CA 908O4" to a band containing three unrelated lines. That is not noisy
supervision, it is WRONG supervision, and training on it teaches the model to
emit text that is not in the image.

Discarding the 63% of pages with a bad count ratio would leave ~846 crops and
throw away most of the real-photo data, which is the data that actually matters
for this product.

THE FIX: use the recogniser to align, then keep only what it agrees on
-------------------------------------------------------------------
The current checkpoint reads held-out real receipts at ~29% word accuracy /
~14% CER. That is far too weak to trust as a label source -- but it is more
than strong enough to answer a much easier question: WHICH transcript line does
this band correspond to?

1. Detect bands, read each one with the current model.
2. Align readings to transcript lines: exact-match anchors first (a band read
   perfectly gives a hard anchor), then positional fill for the rest.
3. Keep only pairs whose CER against their label is below MAX_PAIR_CER.

Step 3 is what makes this safe. A wrong pairing almost never produces a
confident read, so a strict CER threshold discards the mis-paired samples and
keeps the well-paired ones. The model is used as a noisy JOIN KEY, never as an
authority over the label: the label always comes from the human transcription.

The alternative -- training on all positional pairs -- would have produced a
dataset that looks reasonable in every summary statistic and teaches the model
to hallucinate.
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import realdata as R                      # noqa: E402
from build_handheld import (               # noqa: E402
    detect_lines, prepare, legible,
)
from eval_ocr import align_lines          # noqa: E402
from ocr_model import build               # noqa: E402

# A pair is kept only if the model's reading of that band is this close to the
# human transcription. 0.30 is strict on purpose: we are filtering for
# trustworthy pairs, and the dataset is far larger than the model can use in
# 45 CPU epochs anyway.
MAX_PAIR_CER = 0.30

# Per-character pixel budget after scaling. CTC needs timesteps >= chars and
# timesteps = width/8, so below ~8 px/char adjacent glyphs cannot be separated.
# 12 leaves headroom and drops the unlearnable tail.
MIN_PX_PER_CHAR = 12.0


@torch.no_grad()
def read_band(model, crop, device, export_w=256):
    """Run the recogniser on one prepared crop -> normalised text."""
    arr, w = prepare(crop)
    h, ww = arr.shape
    # Right-pad to the fixed export width, exactly as the browser does.
    if ww < export_w:
        arr = np.pad(arr, ((0, 0), (0, export_w - ww)))
    x = (arr.astype(np.float32) / 255.0 - 0.5) / 0.5
    t = torch.from_numpy(x[None, None]).to(device)
    logits = model(t)[0]
    pred = logits.argmax(-1).cpu().numpy()
    # Trim to the crop's true timestep count so padding cannot invent text.
    t_true = min(max(1, w // 8), pred.shape[0])
    return R.norm_label(R.decode_greedy(pred[:t_true]))


def build(out_npz, ckpt=None, device=None, verbose=True):
    from PIL import Image

    if ckpt is None:
        ckpt = R.ROOT / "models" / "ocr" / "crnn.pt"
    if device is None:
        device = torch.device("cpu")
    torch.set_num_threads(2)
    ckd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model = build(ckd["num_classes"])
    model.load_state_dict(ckd["model"])
    model.eval()
    if verbose:
        print(f"aligner checkpoint: epoch {ckd['epoch']}, "
              f"val word-acc {ckd['val_word_acc']*100:.1f}%")

    pages = R.handheld_pages()
    imgs, labs, pgid, widths = [], [], [], []
    split = []
    stats = defaultdict(int)
    samples = []

    for (name, jp, lines) in pages:
        want = [l.strip() for l in lines if len(l.strip()) >= 3]
        if not want:
            stats["page_no_label"] += 1
            continue
        a = np.asarray(Image.open(jp).convert("L"), np.uint8)
        boxes = detect_lines(a)
        if not boxes:
            stats["page_no_bands"] += 1
            continue
        stats["pages"] += 1
        stats["bands"] += len(boxes)

        # Read every band. This is the expensive part: one forward per band.
        reads, kept = [], []
        for (y0, y1, x0, x1) in boxes:
            raw = a[y0:y1, x0:x1]
            if not legible(raw):
                stats["band_illegible"] += 1
                continue
            t = read_band(model, raw, device)
            reads.append(t)
            kept.append((y0, y1, x0, x1))
        stats["bands_read"] += len(reads)

        if not reads:
            continue

        # Align the model's readings to the transcript.
        pairs = align_lines(reads, [R.norm_label(w) for w in want])
        # align_lines returns (got, want) but we need the BOX, so re-walk it
        # in the same order the pairs were produced.
        # Simpler and safer: align indices instead of strings.
        gi = 0
        anchored = []
        # Pass 1 mirror of align_lines, tracking indices.
        used_g, matched_w = set(), set()
        j = 0
        for wi, wn in enumerate([R.norm_label(w) for w in want]):
            k = j
            while k < len(reads):
                if reads[k] == wn:
                    anchored.append((k, wi, True))
                    used_g.add(k)
                    matched_w.add(wi)
                    j = k + 1
                    break
                k += 1
        rest_g = [i for i in range(len(reads)) if i not in used_g]
        rest_w = [i for i in range(len(want)) if i not in matched_w]
        for gi_, wi_ in zip(rest_g, rest_w):
            anchored.append((gi_, wi_, False))
            used_g.add(gi_)
            matched_w.add(wi_)
        stats["anchored_by_exact_read"] += sum(1 for _, _, e in anchored if e)

        for (gi_, wi_, exact) in anchored:
            read = reads[gi_]
            wl = want[wi_]
            norm = R.norm_label(wl)
            if not read or not norm:
                stats["empty_read"] += 1
                continue
            pair_cer = R.cer(read, norm)
            if pair_cer > MAX_PAIR_CER:
                stats["pair_cer_rejected"] += 1
                continue
            raw = a[kept[gi_][0]:kept[gi_][1], kept[gi_][2]:kept[gi_][3]]
            arr_i, sw = prepare(raw)
            t = len(norm)
            if sw / max(t, 1) < MIN_PX_PER_CHAR:
                stats["too_dense_to_read"] += 1
                continue
            imgs.append(arr_i)
            widths.append(sw)
            labs.append(norm)
            pgid.append(name)
            stats["crops"] += 1
            if exact:
                stats["crop_from_anchor"] += 1
            else:
                stats["crop_from_positional"] += 1
            if len(samples) < 8:
                samples.append((name, gi_, wl, read, round(pair_cer, 2), sw, t))

    if not imgs:
        print("HANDHELD CONSENSUS BUILD PRODUCED NO CROPS")
        return None

    # Split by PAGE: two crops from one photo share lighting and perspective.
    uniq = sorted(set(pgid))
    n = len(uniq)
    n_tr = max(1, int(n * 0.70))
    n_va = max(1, int(n * 0.15))
    tr, va = set(uniq[:n_tr]), set(uniq[n_tr:n_tr + n_va])
    te = set(uniq[n_tr + n_va:])
    for p in pgid:
        split.append(0 if p in tr else (1 if p in va else 2))

    maxw = max(c.shape[1] for c in imgs)
    arr = np.zeros((len(imgs), R.CROP_H, maxw), np.uint8)
    for i, c in enumerate(imgs):
        arr[i, :, :c.shape[1]] = c

    np.savez_compressed(
        out_npz, images=arr, labels=np.array(labs, dtype=object),
        page=np.array(pgid, dtype=object), split=np.array(split, np.int8),
        widths=np.array(widths, np.int32),
    )

    if verbose:
        tot_r = stats["pair_cer_rejected"] + stats["crops"]
        print(f"\nconsensus crops: {len(imgs)} from {stats['pages']} pages "
              f"({n} with at least one kept crop)")
        print(f"  bands detected {stats['bands']}, read {stats['bands_read']} "
              f"(dropped {stats['band_illegible']} illegible)")
        print(f"  pairs REJECTED for CER > {MAX_PAIR_CER}: "
              f"{stats['pair_cer_rejected']}  ({stats['pair_cer_rejected']/max(tot_r,1)*100:.0f}% of pairs)")
        print(f"  kept from exact-read anchors: {stats['crop_from_anchor']}")
        print(f"  kept from positional fill:   {stats['crop_from_positional']}")
        print(f"  also dropped: too_dense {stats['too_dense_to_read']}, "
              f"empty_read {stats['empty_read']}")
        for s in range(3):
            c = sum(1 for x in split if x == s)
            u = len({p for p, x in zip(pgid, split) if x == s})
            print(f"  split {s}: {c} crops from {u} pages")
        w = np.array(widths)
        pl = w / np.array([len(l) for l in labs])
        print(f"  scaled width p5 {int(np.percentile(w,5))} p50 {int(np.percentile(w,50))} "
              f"p95 {int(np.percentile(w,95))}")
        print(f"  px/char     p5 {np.percentile(pl,5):.1f} p50 {np.percentile(pl,50):.1f}")
        print(f"  array {arr.shape} ({arr.nbytes/1e6:.1f} MB)")
        print("  kept pairs (page, band, transcript, model read, pair CER, w, n):")
        for s in samples:
            print(f"    {s[0]} b{s[1]}  {s[2][:26]!r} ~ {s[3][:26]!r} "
                  f"cer={s[4]} w={s[5]} n={s[6]}")
    return {"n": len(imgs), "pages": n, "split": np.array(split)}


if __name__ == "__main__":
    out = R.ROOT / "artifacts" / "ocr_handheld.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    build(str(out))
