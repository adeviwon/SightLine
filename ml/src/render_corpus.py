"""
render_corpus.py — train the recogniser on the documents it is actually served.

WHY THIS EXISTS
---------------
The shipped recogniser was trained on ICDAR SROIE receipt word crops and reads
28.0% of held-out crops exactly. The product's documents are not receipt words:
they are forms, statements and medicine labels rendered by
`capture.render_document` — pages, in our own fonts and layout. Train on receipts,
serve forms, and the accuracy that matters (fields spoken correctly) collapses:
measured through the app, 0.0% of ground-truth fields are recovered.

This builds the missing half of the training set from the renderer itself, so the
domain and the geometry match what the app feeds the model.

WHAT MAKES A CROP TRUSTWORTHY HERE
----------------------------------
* LABELS ARE DRAWN, NOT DETECTED. `capture.render_document` places line i at a
  computable y (`line_bands`), so no detector error can leak into a training
  target — unlike a corpus built by running the line finder over photographs.
* LABELS ARE CHARSET-CLEAN. The model emits "0-9A-Z " and space only, so
  `train_label` upper-cases and drops everything else. It is deliberately NOT
  `realdata.norm_label` (which maps O->0, for SCORING only) and not
  `clean_for_training` (which keeps case and punctuation for real photographs).
* SPLITS ARE BY PAGE, never by crop: two crops of the same rendered page never
  sit on opposite sides of train/test, the same leakage rule
  `realdata.split_by_document` enforces for receipts.

USE
---
    bash run.sh py ml/src/render_corpus.py --pages 400          # build
    bash run.sh eval-render                                    # score the model

    from render_corpus import build
    d = build(n_pages=400)      # -> data/render_corpus/{crops.npz,index.json}
"""

import argparse
import json
import random
import zlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import capture       # noqa: E402
import realdata as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "render_corpus"

# Profiles worth training on. worst_case is excluded because evaluate.SUB_HUMAN
# marks it unreadable to a human: a human cannot read it, so the model cannot be
# asked to, and training on it teaches guessing.
PROFILES = ["studio_clean", "handheld_light", "handheld_heavy", "low_light",
            "off_axis", "glossy_glare", "jpeg_social", "paper_texture",
            "hand_shadow"]

SPLIT_RATIOS = (0.70, 0.15, 0.15)


def train_label(s):
    """The string the model is asked to emit: charset-clean, upper-cased."""
    up = str(s).upper()
    kept = "".join(c for c in up if c in R.CHARSET)
    return " ".join(kept.split())


def line_bands(lines, scale=1.0):
    """The y-band of each rendered line, mirroring capture.render_document.

    The renderer's own layout: font size max(12, 19*scale), leading 30*scale,
    first baseline at 28*scale. The band is padded because a headless document
    still has ascenders and descenders, and a band that clips ink would produce a
    label for text that is partly outside the crop.
    """
    fs = max(12, int(19 * scale))
    lh = int(30 * scale)
    y0 = int(28 * scale)
    return [(y0 + i * lh - 3, y0 + i * lh + int(fs * 1.6)) for i in range(len(lines))]


def font_for_page(page_id):
    """The typeface for one page. Deterministic in the page id, so a rebuild is
    reproducible; a page is set in ONE family so its glyphs are consistent."""
    import capture as C
    return C.FONTS[zlib.crc32(str(page_id).encode()) % len(C.FONTS)]


def page_crops(gray, lines, scale=1.0, min_ink_px=3):
    """One tightly-cropped grayscale array + label per rendered line.

    The band comes from the layout; the TIGHT box comes from the ink inside that
    band, which is exactly what `build_dataset.crop_to_array` expects as input
    and what the app's crop rule produces at serving time.
    """
    out = []
    h, w = gray.shape
    for (y0, y1), line in zip(line_bands(lines, scale), lines):
        y0, y1 = max(0, y0), min(h, y1)
        if y1 - y0 < 5:
            continue
        band = gray[y0:y1]
        ink = band < 170
        cols = np.where(ink.sum(axis=0) > 0)[0]
        if len(cols) < min_ink_px:
            continue                       # a blank line carries no readable text
        strip = ink[:, cols[0]:cols[-1] + 1]
        rows = np.where(strip.sum(axis=1) > 0)[0]
        if len(rows) < 3:
            continue
        label = train_label(line)
        if not label:
            continue
        crop = band[y0 - y0 + rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        out.append((np.ascontiguousarray(crop, dtype=np.uint8), label))
    return out


def page_seed(page_id, salt=0):
    """A page's seed.

    zlib.crc32, NOT the builtin hash(): hash() is salted per process, so a repo
    that seeds a corpus with it produces a different corpus on every run (the bug
    ml/src/seedutil.py exists to fix).
    """
    import zlib
    return zlib.crc32((page_id + "|" + str(salt)).encode()) & 0xFFFFFF


def assign_splits(page_ids):
    """Deterministic, page-level splits that are non-empty whenever n >= 7."""
    ids = sorted(page_ids)
    n = len(ids)
    n_tr = max(1, int(round(SPLIT_RATIOS[0] * n)))
    n_va = max(1, int(round(SPLIT_RATIOS[1] * n)))
    out = {}
    for i, pid in enumerate(ids):
        out[pid] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return out


def build(n_pages=400, profiles=None, seed=0, out_dir=None, width=900,
          scales=(0.85, 1.0, 1.15), verbose=True):
    """Render pages, cut their lines, and write a split corpus.

    Returns the dict that was written: crops (object array of uint8 arrays),
    labels, doc (page id), profile, split.
    """
    profiles = list(profiles or PROFILES[:4])
    rng = random.Random(seed)
    page_ids = ["%s-%04d" % (dtype, i)
                for i in range(n_pages) for _, dtype in capture.DOCS]
    splits = assign_splits(page_ids)

    crops, labels, docs, profs, spl, fonts = [], [], [], [], [], []
    for pid in page_ids:
        dtype = pid.rsplit("-", 1)[0]
        lines = next(l for l, d in capture.DOCS if d == dtype)
        scale = rng.choice(scales)
        profile = profiles[rng.randrange(len(profiles))]
        page_font = font_for_page(pid)
        clean = capture.render_document(lines, width=width,
                                        seed=page_seed(pid), scale=scale,
                                        font=page_font)
        deg = capture.apply_profile(clean, seed=page_seed(pid, 1),
                                    **capture.CAPTURE_PROFILES[profile])
        gray = np.asarray(deg.convert("L"), dtype=np.uint8)
        for crop, label in page_crops(gray, lines, scale):
            crops.append(crop)
            labels.append(label)
            docs.append(pid)
            profs.append(profile)
            spl.append(splits[pid])
            fonts.append(page_font)
        if verbose and len(docs) and len(docs) % 500 < len(lines):
            print("  %d crops from %d pages" % (len(docs), page_ids.index(pid) + 1))

    crops_arr = np.empty(len(crops), dtype=object)
    for i, c in enumerate(crops):
        crops_arr[i] = c
    data = {
        "crops": crops_arr,
        "labels": np.array(labels, dtype=object),
        "doc": np.array(docs, dtype=object),
        "profile": np.array(profs, dtype=object),
        "split": np.array(spl, dtype=object),
        "font": np.array(fonts, dtype=object),
    }
    out_dir = Path(out_dir or OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "crops.npz", **data)
    (out_dir / "index.json").write_text(json.dumps({
        "n_crops": len(crops),
        "n_pages": len(page_ids),
        "profiles": profiles,
        "scales": list(scales),
        "seed": seed,
        "split_counts": {s: int((data["split"] == s).sum()) for s in ("train", "val", "test")},
        "fonts": sorted(set(fonts)),
        "font_counts": {Path(f).stem: int((data["font"] == f).sum()) for f in set(fonts)},
        "note": "labels are drawn from capture.render_document; splits are by page; "
                "one typeface per page (the alphabet-confusion lever)",
    }, indent=2) + "\n")
    if verbose:
        print("wrote %d crops from %d pages -> %s"
              % (len(crops), len(page_ids), out_dir / "crops.npz"))
    return data


def load(out_dir=None, split=None):
    """Read the corpus back; optionally one split only."""
    d = np.load(Path(out_dir or OUT) / "crops.npz", allow_pickle=True)
    keys = ["crops", "labels", "doc", "profile", "split"]
    if "font" in d.files:
        keys.append("font")
    data = {k: d[k] for k in keys}
    if split:
        m = data["split"] == split
        data = {k: v[m] for k, v in data.items()}
    return data


def write_cache(data=None, out_path=None, corpus_dir=None):
    """Pack the corpus into the TRAINER'S cache schema.

    `train_ocr.py --data a,b,c` already concatenates several caches and
    normalises the two schemas it has seen (`receipts` for receipts, `page` for
    pages). Emitting that same contract is why the rendered documents can join
    the training set without touching the trainer at all:

        bash run.sh py ml/src/train_ocr.py --data artifacts/ocr_train.npz,\
            data/render_corpus/crops_train.npz --epochs 40

    Crops are resized with `build_dataset.crop_to_array` — the SAME transform
    training uses on receipts — so the two sources are geometrically identical
    and the model cannot tell which one it is looking at.
    """
    import build_dataset as B
    from PIL import Image

    data = data if data is not None else load(corpus_dir)
    crops = [B.crop_to_array(Image.fromarray(np.asarray(c, np.uint8)))
             for c in data["crops"]]
    if not crops:
        raise SystemExit("empty corpus: build it first (render-corpus --pages N)")
    maxw = max(c.shape[1] for c in crops)
    packed = np.zeros((len(crops), crops[0].shape[0], maxw), np.uint8)
    for i, c in enumerate(crops):
        packed[i, :, :c.shape[1]] = c
    out_path = Path(out_path or (Path(corpus_dir or OUT) / "crops_train.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        images=packed,
        widths=np.array([c.shape[1] for c in crops], np.int32),
        labels=np.array([str(x) for x in data["labels"]], dtype=object),
        variants=np.array([str(x) for x in data["profile"]], dtype=object),
        splits=np.array([str(x) for x in data["split"]], dtype=object),
        page=np.array([str(x) for x in data["doc"]], dtype=object),
    )
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=400, help="pages PER document type")
    ap.add_argument("--profiles", default="", help="comma-separated; default the 4 mildest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cache-only", action="store_true",
                    help="skip rendering; pack the corpus already on disk into "
                         "the trainer's cache schema")
    a = ap.parse_args()
    if a.cache_only:
        p = write_cache(corpus_dir=a.out)
        z = np.load(p, allow_pickle=True)
        print("wrote %s  %d crops  maxW %d  splits %s"
              % (p, z["images"].shape[0], z["images"].shape[2],
                 {s: int((z["splits"] == s).sum()) for s in ("train", "val", "test")}))
        return
    profs = [p for p in a.profiles.split(",") if p] or None
    build(n_pages=a.pages, profiles=profs, seed=a.seed, out_dir=a.out)
    write_cache(corpus_dir=a.out)


if __name__ == "__main__":
    main()
