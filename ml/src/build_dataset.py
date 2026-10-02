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
            "lowres", "shadow", "composite"]


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

    raise ValueError(f"unknown augmentation {kind!r}")


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
                    # 'clean' always, plus a deterministic subset of the
                    # degradation variants. Every word gets clean + one
                    # variant chosen by seed, so the corpus is not 9x larger
                    # but still covers every failure mode.
                    picks = ["clean"] + [
                        VARIANTS[seedutil.name_hash(f"{rid}|{kept}|{i}") % len(VARIANTS)]
                        for i in range(2)]
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
