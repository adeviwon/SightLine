"""
Materialise the real-document training set to a cached .npz.

Streams the SROIE parquet row-group by row-group and writes one packed array of
word crops plus their labels and parent receipt ids. Everything downstream
(train_ocr.py, eval_ocr.py) reads only the cache, because decoding a 300 MB
parquet per epoch would dominate training time and the receipt images cannot
all be held in RAM.

CROP GEOMETRY
-------------
Height is fixed at 32 px. CTC requires a constant time axis, so every crop is
resized to the same height and keeps its own width (clamped, then bucketed at
collate time by width-padding into batches).

Width is preserved rather than normalised to a square: a 3-character price and
a 30-character company name have very different aspect ratios, and stretching
both to the same width destroys the glyph proportions the model must read.

WHY AUGMENT HERE AND NOT AT TRAIN TIME
--------------------------------------
Augmentation is applied once, at build time, and each augmented copy is stored.
Reasons: the augmentation pipeline costs real CPU (motion blur + JPEG per
sample), doing it once makes training epochs fast enough to iterate on, and it
makes the dataset byte-identical across runs, which is what lets the
reproducibility gate mean anything. The augmentation itself is deterministic
per (receipt, word, variant) via seedutil, so regenerating reproduces it.

AUGMENTATIONS — each maps to a real capture failure we observed:
  clean      the crop as-is (real thermal/scan appearance)
  jpeg       quality 25-60: WhatsApp/Instagram re-compression
  motion     directional 3-9px: hand shake during capture
  defocus    circular 1-3px: lens not focused at arm's length
  dark       multiply 0.35-0.7 + gamma: dim room, night capture
  bright     divide 1.3-1.9: glare / direct light
  lowres     downscale then upscale: a small photo upscaled by the OS
  shadow     one-sided linear gradient: hand shadow across the page
  composite  2-3 stacked: what a phone actually produces (motion + dark + jpeg)

The composite variants matter most: real captures are never singly degraded,
and a model trained only on single degradations tends to fail on the
combination, which is the only case that occurs in the field.

SPLIT BY DOCUMENT
-----------------
Emitted alongside the crops, and computed over receipt ids BEFORE any
augmentation, so no receipt can appear in two splits.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import realdata as R          # noqa: E402
import seedutil               # noqa: E402

CROP_H = R.CROP_H
MIN_W, MAX_W = R.MIN_W, R.MAX_W

VARIANTS = ["clean", "jpeg", "motion", "defocus", "dark", "bright",
            "lowres", "shadow", "composite", "perspective", "rotate"]

# Populated on first use by _load_homographies(). Declared here, above
# augment(), because referencing it before the assignment at the bottom of the
# file made it an unbound local at call time.
_HOM_CACHE = None


def crop_to_array(img: Image.Image) -> np.ndarray:
    """
    One word crop -> uint8 [CROP_H, W] grayscale, aspect preserved.

    Grayscale rather than RGB: every field this app extracts is a high-contrast
    dark-on-light token, so colour carries no signal the recogniser needs, and
    dropping a channel cuts the first conv's input by a third.
    """
    g = img.convert("L")
    w, h = g.size
    if w < 1 or h < 1:
        return np.zeros((CROP_H, MAX_W), np.uint8)
    scale = CROP_H / float(h)
    nw = max(MIN_W, min(MAX_W, int(round(w * scale))))
    return np.asarray(g.resize((nw, CROP_H), Image.BILINEAR), np.uint8)


def augment(arr: np.ndarray, kind: str, seed: int) -> np.ndarray:
    """Apply one named degradation to a uint8 crop. Deterministic in `seed`."""
    import cv2
    rng = np.random.RandomState(seed % (2 ** 31))
    a = arr
    if kind == "clean":
        return a

    if kind == "jpeg":
        import io
        q = int(rng.randint(25, 61))
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=q)
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("L"), np.uint8)

    if kind == "motion":
        length = int(rng.randint(3, 10))
        ang = float(rng.uniform(0, 180))
        k = np.zeros((length, length), np.float32)
        k[length // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), ang, 1.0)
        k = cv2.warpAffine(k, M, (length, length))
        k /= max(k.sum(), 1e-6)
        return cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REPLICATE)

    if kind == "defocus":
        r = int(rng.randint(1, 4)) | 1
        k = np.zeros((r, r), np.float32)
        cv2.circle(k, (r // 2, r // 2), max(1, r // 2 - 1), 1, -1)
        k /= max(k.sum(), 1e-6)
        return cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REPLICATE)

    if kind == "dark":
        f = float(rng.uniform(0.35, 0.70))
        g = float(rng.uniform(0.8, 1.4))
        x = (a.astype(np.float32) / 255.0) * f
        return np.clip((x ** g) * 255.0, 0, 255).astype(np.uint8)

    if kind == "bright":
        f = float(rng.uniform(1.3, 1.9))
        return np.clip((a.astype(np.float32) / 255.0) * f * 255.0,
                       0, 255).astype(np.uint8)

    if kind == "lowres":
        f = float(rng.uniform(0.35, 0.6))
        h, w = a.shape
        small = cv2.resize(a, (max(4, int(w * f)), max(4, int(h * f))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    if kind == "shadow":
        h, w = a.shape
        g = np.linspace(float(rng.uniform(0.25, 0.55)), 1.0, w,
                        dtype=np.float32)[None, :]
        if rng.rand() < 0.5:
            g = g[:, ::-1]
        return np.clip(a.astype(np.float32) * g, 0, 255).astype(np.uint8)

    if kind == "composite":
        # The realistic case: two or three stacked, in a plausible order.
        # Enhancement-style degradations last, matching how a capture pipeline
        # actually behaves (the sensor degrades, then the codec compresses).
        order = ["motion", "dark", "jpeg"] if rng.rand() < 0.5 else \
                ["defocus", "bright", "jpeg"]
        for i, k in enumerate(order):
            a = augment(a, k, seed + 7919 * (i + 1))
        return a

    if kind == "perspective":
        # A REAL handheld four-corner homography from the Zenodo set, applied
        # to a flat scan crop. This is the only geometric variant, and its
        # absence was a genuine gap: the eight others are all photometric
        # (blur, exposure, compression), so nothing in training ever taught
        # the model that a glyph can be a trapezoid. The hand-photographed
        # receipts are shot at an angle, and a model trained on axis-aligned
        # scans reads them far worse than its held-out scan score predicts.
        #
        # Using measured corner positions rather than a random shear matters:
        # the real set spans up to 4010 px on the long edge, and those are the
        # distortions a phone actually produces when held over a page.
        # `global` is required: assigning _HOM_CACHE here makes Python treat it
        # as a local for the WHOLE function, so reading it first raised
        # UnboundLocalError instead of returning the module-level None.
        global _HOM_CACHE
        homs = _HOM_CACHE
        if homs is None:
            homs = _HOM_CACHE = _load_homographies()
        if not homs:
            return a
        h, w = a.shape
        quad = homs[seedutil.name_hash(f"h{seed}") % len(homs)]
        pad = 3
        src = np.float32([[pad, pad], [w - 1 - pad, pad],
                          [w - 1 - pad, h - 1 - pad], [pad, h - 1 - pad]])
        # Map the crop's rectangle onto a subset of the measured quad, so the
        # distortion's magnitude matches the real capture instead of warping
        # the text to some arbitrary skew.
        # SIZE THE QUAD TO THE CROP, then let the homography's SHAPE skew it.
        #
        # Two mistakes here, both measured rather than reasoned about:
        #
        # 1. Scaling by the normalised [-1,1] coordinates put the destination
        #    in a ~28x6 px box in the middle of a 200x32 crop, so 97% of the
        #    output was border.
        # 2. borderValue=0 filled that border with BLACK. Measured 6182 of
        #    6400 pixels below the ink threshold -- the augmentation was
        #    replacing the word with a black rectangle, which would have taught
        #    the model to read noise while looking like a valid training
        #    sample. The border must be PAPER, which is the crop's own bright
        #    value, not zero.
        #
        # So: the quad's mean corner sits at the crop centre, and the corner
        # OFFSETS are scaled to the crop's half-extents in each axis. That keeps
        # the perspective's proportions while filling the frame.
        cx, cy = w / 2.0, h / 2.0
        f = float(rng.uniform(0.55, 0.95))   # fraction of the half-extent
        q = quad - quad.mean(axis=0)          # centred, roughly [-1,1]
        q = q * np.float32([w * 0.5 * f, h * 0.5 * f])
        if rng.rand() < 0.5:                  # the sheet can tilt either way
            q = q * np.float32([-1.0, 1.0])
        dst = (q + np.float32([cx, cy])).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        # Paper tone, from the crop's own bright pixels, so it adapts to a
        # dim or warm photo instead of assuming white.
        paper = int(np.percentile(a, 90))
        return cv2.warpPerspective(a, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=paper)

    if kind == "rotate":
        # Small in-plane rotation, the way a page sits when you do not line it
        # up with the camera. Kept modest on purpose: a word rotated 15 degrees
        # is a different recognition problem, not a harder version of this one.
        #
        # borderValue is the crop's own paper tone, not 0. Rotation exposes the
        # corners, and filling them black teaches the model to read a black
        # frame's worth of nothing as if it were a word.
        ang = float(rng.uniform(-7.0, 7.0))
        h, w = a.shape
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        paper = int(np.percentile(a, 90))
        return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=paper)

    raise ValueError(f"unknown augmentation {kind!r}")


def _load_homographies():
    """
    Read the Zenodo four-corner homographies, normalised to sit around the origin.

    The raw corners are in receipt-image pixel coordinates, spanning up to
    4010 px on the long edge. They are centred and divided by their own
    half-extent so every one lands in roughly [-1, 1]. That normalisation is
    what makes them reusable on a 32 px-tall word crop: the SHAPE of the
    distortion is preserved and only the scale, which is meaningless across
    receipts of different sizes, is discarded.

    Returns a list of float32 [4,2] arrays, empty when the handheld data is
    absent -- the perspective variant then becomes a no-op rather than an
    error, so the corpus still builds on a machine that has only SROIE.
    """
    try:
        raw = R.handheld_homography()
    except Exception:
        return []
    out = []
    for pts in raw.values():
        p = np.asarray(pts, np.float32)
        if p.shape != (4, 2) or p.max() <= 0:
            continue
        # SORT THE CORNERS into a ring before using them.
        #
        # The CSV stores them in the order TL, TR, BL, BR -- interleaved, not
        # cyclic. Read literally that quad is self-intersecting, and warping
        # with it folded the crop over itself: measured 680% of the original
        # ink pixels and mean luminance 79/255, i.e. the whole frame filled
        # dark. Sorting by angle around the centroid gives the true cyclic
        # order TL, TR, BR, BL, whose shoelace sign is the opposite of the raw
        # ordering's -- the same page, correctly wound.
        c = p.mean(axis=0)
        p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
        p = p - p.mean(axis=0)
        half = np.abs(p).max()
        if half <= 1e-6:
            continue                      # degenerate quad
        out.append((p / half).astype(np.float32))
    return out


def build(max_receipts=260, max_crops_per_receipt=40, seed=7):
    import pyarrow.parquet as pq

    files = sorted(R.SROIE_DIR.glob("train-*.parquet"))
    if not files:
        raise SystemExit("no SROIE parquet found; run the download step first")

    crops, labels, receipts, variants = [], [], [], []
    seen_receipts = set()

    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                rid = row.get("key") or ""
                if rid in seen_receipts:
                    continue
                if len(seen_receipts) >= max_receipts:
                    break
                words = row.get("words") or []
                boxes = row.get("bboxes") or []
                if not words:
                    continue
                try:
                    img = Image.open(
                        __import__("io").BytesIO(row["image"]["bytes"]))
                except Exception:
                    continue

                kept = 0
                for w, b in zip(words, boxes):
                    if kept >= max_crops_per_receipt:
                        break
                    txt = R.clean_for_training(w)
                    if len(txt) < 2 or not R.encode(txt):
                        continue
                    try:
                        x1, y1, x2, y2 = [int(v) for v in b]
                    except (TypeError, ValueError):
                        continue
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        continue
                    # Pad by 2px: tight boxes clip ascenders/descenders, and a
                    # clipped 'f' or '5' is a label the model cannot learn.
                    pad = 2
                    box = (max(0, x1 - pad), max(0, y1 - pad),
                           min(img.width, x2 + pad), min(img.height, y2 + pad))
                    if box[2] - box[0] < 6 or box[3] - box[1] < 6:
                        continue
                    base = crop_to_array(img.crop(box))
                    # 'clean' always, then DISTINCT degradation variants.
                    #
                    # The old picker was `VARIANTS[hash % len(VARIANTS)]` for
                    # i in range(2), which can draw "clean" twice or the same
                    # variant twice -- so a third of the augmentation budget
                    # silently did nothing, and no statistic revealed it. The
                    # clean sample is now always included exactly once and the
                    # two extra picks are guaranteed distinct and never clean.
                    others = [v for v in VARIANTS if v != "clean"]
                    h0 = seedutil.name_hash(f"{rid}|{kept}")
                    picks = ["clean"]
                    seen_k = {"clean"}
                    for i in range(2):
                        v = others[(h0 + i * 7919) % len(others)]
                        if v in seen_k:                 # hash collision
                            v = others[(h0 + i * 7919 + 1) % len(others)]
                        if v not in seen_k:
                            picks.append(v)
                            seen_k.add(v)
                    for kind in picks:
                        s = (seed * 1_000_003
                             + seedutil.name_hash(f"{rid}|{kept}|{kind}"))
                        crops.append(augment(base, kind, s))
                        labels.append(txt)
                        receipts.append(rid)
                        variants.append(kind)
                    kept += 1
                if kept:
                    seen_receipts.add(rid)
            if len(seen_receipts) >= max_receipts:
                break
        if len(seen_receipts) >= max_receipts:
            break

    # Split by DOCUMENT, before considering augmentation.
    idx = list(range(len(crops)))
    split_of = {}
    for k, v in R.split_by_document(
            [(i, receipts[i]) for i in idx], lambda p: p[1], seed=seed).items():
        for i, _ in v:
            split_of[i] = k

    maxw = max(c.shape[1] for c in crops)
    packed = np.zeros((len(crops), CROP_H, maxw), np.uint8)
    for i, c in enumerate(crops):
        packed[i, :, :c.shape[1]] = c

    R.OUT.mkdir(parents=True, exist_ok=True)
    out = R.OUT / "ocr_train.npz"
    np.savez_compressed(
        out,
        images=packed,
        widths=np.array([c.shape[1] for c in crops], np.int32),
        labels=np.array(labels, dtype=object),
        receipts=np.array(receipts, dtype=object),
        variants=np.array(variants, dtype=object),
        splits=np.array([split_of.get(i, "train") for i in idx], dtype=object),
    )
    return out, packed, labels, receipts, variants, split_of


def main():
    if "--selftest" in sys.argv:
        print("augmentation check on a synthetic 32x200 crop:")
        base = np.tile(np.linspace(20, 230, 200, dtype=np.uint8), (32, 1))
        for k in VARIANTS:
            a = augment(base, k, 12345)
            assert a.shape == base.shape, (k, a.shape)
            assert a.dtype == np.uint8
            print(f"  {k:10} shape={a.shape} mean={a.mean():6.1f} "
                  f"(base {base.mean():.1f})")
        # determinism
        assert np.array_equal(augment(base, "composite", 7),
                              augment(base, "composite", 7))
        print("  deterministic: OK")
        return 0

    limit = 260
    for i, a in enumerate(sys.argv):
        if a == "--receipts" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    print(f"building crops from up to {limit} SROIE receipts ...")
    out, packed, labels, receipts, variants, split_of = build(limit)
    n = len(labels)
    print(f"wrote {out}  ({out.stat().st_size/1024/1024:.1f} MB)")
    print(f"  crops     {n}")
    print(f"  width     max {packed.shape[2]}px, height {packed.shape[1]}px")
    print(f"  receipts  {len(set(receipts))}")
    from collections import Counter
    sc = Counter([split_of[i] for i in range(n)])
    for k in ("train", "val", "test"):
        ids = {receipts[i] for i in range(n) if split_of.get(i) == k}
        print(f"  {k:6} {sc.get(k,0):6} crops  {len(ids):4} receipts")
    print("  variants:", dict(Counter(variants).most_common()))
    lens = [len(R.encode(l)) for l in labels]
    print(f"  label length: min {min(lens)} median {int(np.median(lens))} "
          f"max {max(lens)}")


if __name__ == "__main__":
    sys.exit(main())
