"""
Train on the hand-photographed receipts, not just clean scans.

WHY THIS EXISTS
---------------
The recogniser's training set is SROIE: real receipts, but FLATBED SCANS --
even lighting, square to the lens, no perspective. It is then evaluated on the
Zenodo hand-photographed set: night shots, receipts in plastic sleeves,
crooked, uneven lighting, glare.

That is a domain gap, and it showed up exactly where you would expect. On
held-out SROIE the model reaches ~29% word accuracy / ~14% CER. On the
handheld photos the same checkpoint gives ~56% CER on lines the detector found.
Nothing is broken -- the model simply has never seen an image like this.

So we build line crops from the handheld pages and train on them too, using the
dataset's own whitespace-aligned transcriptions as labels.

THE LABELS ARE USABLE, WITH A CAVEAT
------------------------------------
The Zenodo label files are the full receipt transcription with runs of spaces
preserving line breaks, so label line k corresponds to detected band k. That
gives us real, human-written ground truth on real photographs. It is NOT word
level -- we get one string per line -- which is fine for CTC, since CTC needs
only the character sequence, not per-character boxes. That is a genuine
advantage of CTC over a word-segmentation approach and worth stating.

SPLIT BY PAGE IDENTITY
----------------------
Same rule as SROIE: two crops from the same photograph share lighting,
perspective and paper texture, so splitting them across train/val would leak
and inflate the score. Pages are split, never crops.
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import realdata as R  # noqa: E402


def detect_lines(gray):
    """
    Find text bands in a hand-held photo.

    The same approach the evaluator uses, kept here so training and evaluation
    see the SAME crops -- if they diverged, the evaluation would be measuring
    a preprocessing the model never trained on, which is the classic way to
    get a number that means nothing.

    Ink is isolated by LOCAL CONTRAST against a median-blurred paper estimate.
    Plain Otsu does not work here: these are bright receipts on a dark table
    at night, and any global threshold marks the paper as a solid block, so
    the row projection is full width everywhere and no bands are found.

    Returns a list of (y0, y1, x0, x1) boxes.
    """
    a = gray
    H, W = a.shape
    paper = cv2.medianBlur(a, 31)
    ink = (a.astype(np.int16) < paper.astype(np.int16) - 18)

    ys = np.where(ink.sum(axis=1) > W * 0.02)[0]
    if len(ys) <= 20:
        return []
    y_lo, y_hi = int(ys[0]) - 5, int(ys[-1]) + 6
    y_lo = max(0, y_lo)
    y_hi = min(H, y_hi)

    sub = ink[y_lo:y_hi, :]
    proj = sub.sum(axis=1)
    thr = max(2, int(proj.max() * 0.04))
    bands, start = [], None
    for i, v in enumerate(proj):
        if v > thr and start is None:
            start = i
        elif v <= thr and start is not None:
            if i - start >= 8:
                bands.append((start, i))
            start = None
    if start is not None and len(proj) - start >= 8:
        bands.append((start, len(proj)))

    out = []
    for (b0, b1) in bands:
        yy0, yy1 = y_lo + b0, y_lo + b1
        cols = np.where(ink[yy0:yy1, :].sum(axis=0) > 0)[0]
        if len(cols) < 4:
            continue
        pad = max(2, int((yy1 - yy0) * 0.10))
        x0 = max(0, int(cols[0]) - pad)
        x1 = min(W, int(cols[-1]) + pad + 1)
        out.append((yy0, yy1, x0, x1))
    return out


def prepare(crop, max_w=None):
    """
    Grayscale crop -> the exact array the recogniser consumes.

    SCALES PROPORTIONALLY, THEN PADS. This is the fix for a bug that was
    quietly corrupting most of the handheld training data.

    The old code did `cv2.resize(crop, (crop.shape[1], 32))` and then truncated
    to MAX_W. Resizing to (width, 32) changes ONLY the height, so a 750x30 px
    line became 750x32 and was then chopped to 320x32 -- a 2.3x HORIZONTAL
    squash of text that was already only 30 px tall. Characters ended up about
    a third as wide as they were high, which is not a font, it is a smear.

    Worse, the label was the FULL transcript line while the pixels were only
    the first 320 px of it. That trains the model to hallucinate characters it
    cannot see -- actively harmful, and it would have shown up only as a model
    that never quite converges.

    The fix: fit the whole crop inside (max_w, CROP_H) preserving aspect
    ratio, then pad the remainder. A wide line becomes shorter AND its
    characters stay square, because both dimensions scale by the same factor.

    Returns (array, scaled_width) -- the caller needs the post-scale width
    because that is what determines the timestep count, not the padded width.
    """
    if max_w is None:
        max_w = R.MAX_W
    h, w = crop.shape
    if w <= 0 or h <= 0:
        return np.zeros((R.CROP_H, max_w), np.uint8), max_w

    # Uniform scale so the crop fits in BOTH dimensions.
    scale = min(max_w / w, R.CROP_H / h)
    nw = max(1, min(max_w, int(round(w * scale))))
    nh = max(1, min(R.CROP_H, int(round(h * scale))))
    c = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    out = np.zeros((R.CROP_H, nw), np.uint8)
    out[:nh, :nw] = c
    return out, nw


def legible(crop, min_contrast=28.0, min_ink_frac=0.01):
    """
    Reject bands that carry no readable text.

    A projection band is not automatically a line of text. On these photos the
    detector also fires on the plastic sleeve's edge, the shadow the receipt
    casts on the table, and the dark background above the first line. Those
    produce crops that are mostly dark with a little specular noise -- and if
    one is paired with a transcript line, it is a LABELLED sample teaching the
    model to read noise as 'GREEN FIELD'. Worse than dropping the page: it is
    actively harmful noise, and it is invisible in the loss curve.

    Two cheap tests, both of which a real text line passes comfortably:
      * contrast between the dark and light pixels in the crop -- text has
        some, a dark sleeve edge has almost none;
      * fraction of pixels that are ink -- text is a small but non-zero
        fraction, whereas an empty band is nearly zero.
    """
    c = crop.astype(np.int16)
    if c.size == 0:
        return False
    lo, hi = int(c.min()), int(c.max())
    if hi - lo < min_contrast:
        return False
    # Ink fraction against the crop's own midpoint, so this works on a bright
    # page and a dark one alike without assuming a polarity.
    mid = (lo + hi) / 2.0
    ink = float((c < mid).mean())
    return ink >= min_ink_frac


def fits(crop, max_w=None):
    """True if the crop keeps full resolution inside the width budget."""
    if max_w is None:
        max_w = R.MAX_W
    h, w = crop.shape
    return min(max_w / w, R.CROP_H / h) >= 1.0


def build(out_npz, pages_limit=None, verbose=True, strict_count=False):
    """
    Build artifacts/ocr_handheld.npz.

    Returns a dict of arrays plus the split counts, and PRINTS the pairing
    rate rather than silently training on a misaligned set. If the detector
    and the transcript disagree about where lines are, every label is wrong,
    and the only symptom would be a model that never converges -- so the
    pairing rate is reported loudly.

    strict_count=True keeps ONLY pages where the detector found exactly as
    many bands as the transcript has lines, so every positional label is
    trustworthy. Costs volume (measured: ~20/187 pages qualify) for label
    correctness -- the 8-epoch A/B run showed the noisier default cache
    teaches the model to blank out more lines (recall 60.7% vs 70.9%).
    """
    from PIL import Image

    pages = R.handheld_pages()
    if pages_limit:
        pages = pages[:pages_limit]

    imgs, labs, pgid, split = [], [], [], []
    widths = []              # post-scale ink width, for the timestep count
    stats = defaultdict(int)
    samples = []

    # Minimum pixels per character after scaling. CTC needs at least ~1.5
    # timesteps per character to separate adjacent glyphs, and timesteps =
    # width/8, so width/chars >= 12 is the hard floor. Below that the sample is
    # not merely hard, it is unlearnable, and keeping it only adds noise.
    MIN_PX_PER_CHAR = 12.0

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

        # How many bands does the transcript have? If they differ a lot, the
        # pairing below is guesswork and the labels are unreliable.
        stats["pages"] += 1
        stats["bands"] += len(boxes)
        stats["labels"] += len(want)
        ratio = len(boxes) / max(len(want), 1)
        if ratio < 0.5 or ratio > 2.0:
            stats["page_count_mismatch"] += 1
        if strict_count and len(boxes) != len(want):
            stats["page_count_mismatch_skipped"] += 1
            continue

        # Pair by POSITION, using align_lines' exact-match anchoring where it
        # helps. Positional is the honest default: the label file preserves
        # line breaks in order, so band k is line k unless the detector missed
        # or invented one. Where counts differ we still pair in order and let
        # the model see slightly misaligned pairs, because discarding them
        # would throw away most of the handheld data.
        n = min(len(boxes), len(want))
        for k in range(n):
            y0, y1, x0, x1 = boxes[k]
            lab = R.norm_label(want[k])
            if len(lab) < 2:
                stats["label_too_short"] += 1
                continue
            if len(lab) > 47:      # matches the training cache's observed max
                stats["label_too_long"] += 1
                continue
            t = len(lab)
            raw = a[y0:y1, x0:x1]
            if not legible(raw):
                stats["illegible_band"] += 1
                continue
            arr_i, sw = prepare(raw)
            # Only keep the crop if the text is still legible after scaling.
            # A 750 px line squeezed into 320 px keeps square characters but
            # is genuinely small; below this the CTC model has too few
            # timesteps per character to separate them, and the label is
            # longer than the image can justify. Better to drop the sample
            # than to train on an impossible one.
            per_char = sw / max(t, 1)
            if per_char < MIN_PX_PER_CHAR:
                stats["too_dense_to_read"] += 1
                continue
            imgs.append(arr_i)
            widths.append(sw)
            labs.append(lab)
            pgid.append(name)
            stats["crops"] += 1
            if len(samples) < 6:
                samples.append((name, k, want[k], lab, arr_i.shape, sw, t))

    if not imgs:
        print("HANDHELD BUILD PRODUCED NO CROPS")
        return None

    # Split by PAGE so lighting and perspective never leak across splits.
    uniq = sorted(set(pgid))
    n = len(uniq)
    n_tr = max(1, int(n * 0.70))
    n_va = max(1, int(n * 0.15))
    tr = set(uniq[:n_tr])
    va = set(uniq[n_tr:n_tr + n_va])
    te = set(uniq[n_tr + n_va:])
    for p in pgid:
        split.append(0 if p in tr else (1 if p in va else 2))

    H = R.CROP_H
    maxw = max(c.shape[1] for c in imgs)
    arr = np.zeros((len(imgs), H, maxw), np.uint8)
    for i, c in enumerate(imgs):
        arr[i, :, :c.shape[1]] = c

    np.savez_compressed(
        out_npz,
        images=arr,
        labels=np.array(labs, dtype=object),
        page=np.array(pgid, dtype=object),
        split=np.array(split, np.int8),
        # Real ink width per crop. The training loop MUST use this for the CTC
        # input length, not the padded array width -- telling CTC that
        # padding is signal is the same class of bug as the padded-width bug
        # in the SROIE loader, and it costs accuracy silently.
        widths=np.array(widths, np.int32),
    )

    if verbose:
        print(f"handheld crops: {len(imgs)} from {stats['pages']} pages "
              f"({len(uniq)} unique)")
        print(f"  bands detected {stats['bands']}, transcript lines "
              f"{stats['labels']}  -> {stats['bands']/max(stats['labels'],1):.2f} "
              f"bands per line")
        print(f"  pages with a >2x or <0.5x count mismatch: "
              f"{stats['page_count_mismatch']}")
        print(f"  dropped: {dict((k, v) for k, v in stats.items() if k.startswith(('label_', 'too_')))}")
        for s in range(3):
            c = sum(1 for x in split if x == s)
            u = len({p for p, x in zip(pgid, split) if x == s})
            print(f"  split {s}: {c} crops from {u} pages")
        w = np.array(widths)
        print(f"  scaled width: p5 {int(np.percentile(w,5))} p50 {int(np.percentile(w,50))} "
              f"p95 {int(np.percentile(w,95))} max {w.max()}")
        print(f"  px/char:     p5 {np.percentile(w/np.array([len(l) for l in labs]),5):.1f} "
              f"p50 {np.percentile(w/np.array([len(l) for l in labs]),50):.1f}")
        print(f"  crop array {arr.shape} ({arr.nbytes/1e6:.1f} MB)")
        print("  sample pairs (page, line, raw, normalised, shape, width, chars):")
        for s in samples:
            print(f"    {s[0]} #{s[1]}  {s[2]!r} -> {s[3]!r}  {s[4]} w={s[5]} n={s[6]}")
    return {"n": len(imgs), "pages": n, "split": np.array(split)}


def _selftest():
    """Prove the domain gap is real, using the CURRENT checkpoint."""
    import torch
    from ocr_model import build

    torch.set_num_threads(2)
    ck = torch.load(str(R.ROOT / "models" / "ocr" / "crnn.pt"),
                    map_location="cpu", weights_only=False)
    m = build(ck["num_classes"])
    m.load_state_dict(ck["model"])
    m.eval()
    from eval_ocr import eval_handheld
    r = eval_handheld(m, torch.device("cpu"), limit=6)
    print(f"\ncurrent checkpoint on handheld: CER {r['cer']*100:.1f}%, "
          f"line acc {r['line_acc']*100:.1f}%, "
          f"detection recall {r['detection_recall']*100:.1f}%")
    return True


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        out = R.ROOT / "artifacts" / "ocr_handheld.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        build(str(out))
