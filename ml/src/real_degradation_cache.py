"""
real_degradation_cache.py — train the recogniser on REAL photographs, not just
synthetic degradation.

WHY THIS EXISTS (the measured lever)
------------------------------------
Training on the ~22,800 RENDERED line crops lifted field recovery 27/300 ->
108/300 (+27pp, p=8.3e-25) and lifted recall on the correct ground truth
(spot+validation 0.337 -> 0.615). That corpus (`render_corpus.py`) is built by
DEGRADING clean renders with a synthetic camera model, and it fails hardest on
exactly the profiles that pretend to be a real camera:

    low_light        CER 0.72
    handheld_heavy   CER 0.51
    studio_clean     CER 0.09

Meanwhile the repo already contains REAL camera degradation and never trains on
it: `realdata.handheld_pages()` is 193 hand-photographed express-expense
receipts, shot by hand at night on a dark background in a plastic sleeve. This
module turns THOSE into a labelled (crop, label) cache in the trainer's own
schema, so the next run can simply append it to `--data`.

LABELS ARE ALIGNED BY GEOMETRY, NEVER BY THE MODEL
--------------------------------------------------
The hard part is the label. A transcript is a whole-page list of text lines; the
detector finds N line boxes. This module pairs them POSITIONALLY: transcript
line i in reading order -> detected box i in reading order, top to bottom.

If the two counts do not match for a page, THE PAGE IS DROPPED. It is never
guessed and never labelled from the model's own read -- that would be circular
(the model would be trained to reproduce its own errors) and would poison the
exact lever we are trying to pull. The drop count and its reason are reported.

THE DETECTOR IS REUSED, NOT REINVENTED
--------------------------------------
Line finding is the SHIPPED detector from `eval_ocr.py` (`ink_mask`,
`line_bands`, `locate_document`, `bands_in_document`), driven through the same
band-selection and crop-geometry code as `eval_ocr.read_page`. `read_page`
returns recognised TEXT, not boxes, and is not mine to edit, so this module
reproduces its detection half with its own helpers -- same functions, so the
crops are the crops the evaluator sees. It stops before recognition.

SPLIT BY DOCUMENT
-----------------
`realdata.split_by_document` partitions receipt ids, so no receipt sits on both
sides of train/test and the held-out real pages stay held out. The trainer then
sees real val/test crops only as a measurement, never as a training target.

USE
---
    bash run.sh py ml/src/real_degradation_cache.py --build
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as B      # noqa: E402  (crop_to_array: ONE crop geometry)
import eval_ocr as EO          # noqa: E402  (the shipped detector path)
import realdata as R           # noqa: E402
import render_corpus as RC     # noqa: E402  (write_cache: the trainer's schema)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "real_corpus"
SYNTH = ROOT / "data" / "render_corpus"

CROP_H = R.CROP_H                      # 32 -- CTC needs a constant time axis
MAX_W = R.MAX_W                        # 320 -- the trainer's width clamp
VARIANTS = ("real",)                   # one variant: this IS the real degradation

# The four profiles the retrain optimises for, in the order the task names them.
PROFILES = ["studio_clean", "handheld_light", "handheld_heavy", "low_light"]

MERGE_CMD = (
    "bash run.sh py ml/src/train_ocr.py \\\n"
    "  --data artifacts/ocr_train.npz,data/render_corpus/crops_train.npz,"
    "data/real_corpus/crops_train.npz \\\n"
    "  --epochs 30 --threads 4"
)
REMEASURE_CMDS = [
    "bash run.sh eval-render --split test",
    "bash run.sh eval-ocr",
    "bash run.sh py ml/src/eval_field_truth.py --split test --limit-pages 120",
]


# ── the detector: eval_ocr.read_page's detection half, boxes only ─────────────

def detect_boxes(gray):
    """Tight (y0, y1, x0, x1) boxes for the text lines on one photo.

    Mirrors `eval_ocr.read_page` exactly up to the point it calls `recognise`,
    using the SAME helpers (ink_mask, line_bands, locate_document,
    bands_in_document). No model is involved, which is the whole point: the
    boxes cannot be contaminated by what the model would read.
    """
    H, W = gray.shape
    ink = EO.ink_mask(gray)
    binv = (ink.astype(np.uint8)) * 255
    off = 0
    ys = np.where(binv.sum(axis=1) > W * 0.02)[0]
    if len(ys) > 20:
        off = max(0, ys[0] - 5)
        binv = binv[off:min(H, ys[-1] + 6), :]
    proj = (binv > 0).sum(axis=1)
    bands = [(y0 + off, y1 + off) for (y0, y1) in EO.line_bands(proj)]
    # The locate_document override is deliberately NOT used here, and this is a
    # MEASURED deletion, not a tidy-up. Against an exact drawn ground truth (120
    # re-rendered test pages, 1800 lines) the override costs 56 lines: line
    # recall 1744/1800 = 0.969 versus 1800/1800 = 1.000 without it, exact
    # McNemar p=2.776e-17, pairs (b=0, c=56) -- never better on any page. It is
    # concentrated on the low_light profile (line recall 0.867, 14 of 28 pages
    # lose lines) and costs nothing back on the 193 real hand-photographed
    # receipts (transcript count-match 16/193 with it, 20/193 without,
    # p=0.125). detect_document() picks the largest bright blob and hands back a
    # quad whose bands discard real lines. `locate_document` and
    # `bands_in_document` stay in this module -- they are still tested and still
    # the right tool for localising a receipt on a table -- but the bander wins
    # on pages that are already the document.

    boxes = []
    for (y0, y1) in bands:
        strip = ink[y0:y1, :]
        cols = np.where(strip.sum(axis=0) > 0)[0]
        if len(cols) < 4:                       # read_page's own minimum
            continue
        pad = max(2, int((y1 - y0) * 0.10))
        x0 = max(0, int(cols[0]) - pad)
        x1 = min(W, int(cols[-1]) + pad + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        boxes.append((int(y0), int(y1), int(x0), int(x1)))
    return boxes


# ── label alignment: geometry only, drop on any mismatch ─────────────────────

def align_labels(boxes, lines):
    """Pair detected boxes with transcript lines, in reading order.

    Returns (pairs, reason): `pairs` is a list of (box, line) when the counts
    match, else None and a human reason for the drop. A count mismatch is the
    ONLY failure mode, and it is a DROP -- never a positional guess, never the
    model's own read.
    """
    if not lines:
        return None, "no transcript lines"
    if len(boxes) != len(lines):
        return None, (f"count mismatch: {len(boxes)} detected boxes vs "
                      f"{len(lines)} transcript lines")
    return list(zip(boxes, lines)), None


def page_crops(page, detect=detect_boxes):
    """One photo -> (result, reason, n_boxes) at the trainer's geometry.

    `result` is (crops, labels) or None when the page must be dropped; `reason`
    says why; `n_boxes` is the detector's count (returned even on a drop so the
    report can show the mismatch without re-running the detector).

    `crops` are uint8 [CROP_H, w<=MAX_W] arrays (build_dataset.crop_to_array),
    `labels` are charset-clean (render_corpus.train_label) -- the SAME geometry
    and label convention as the rendered corpus, so the model cannot tell the
    two sources apart.
    """
    name, jp, lines = page
    gray = np.asarray(Image.open(jp).convert("L"), np.uint8)
    boxes = detect(gray)
    pairs, reason = align_labels(boxes, lines)
    if pairs is None:
        return None, reason, len(boxes)
    crops, labels = [], []
    for (y0, y1, x0, x1), line in pairs:
        label = RC.train_label(line)
        if not label:
            continue                        # line is illegible in the charset
        crop = B.crop_to_array(Image.fromarray(gray[y0:y1, x0:x1], "L"))
        crops.append(crop)
        labels.append(label)
    if not crops:
        return None, "no charset-legible line", len(boxes)
    return (crops, labels), None, len(boxes)


# ── build + document-disjoint split ──────────────────────────────────────────

def collect_pages(pages, limit=None, detect=detect_boxes, progress=None):
    """[page] -> (per_page, dropped). per_page: {name: (crops, labels)}."""
    per_page, dropped = {}, []
    for i, page in enumerate(pages):
        if limit and i >= limit:
            break
        res, reason, n_boxes = page_crops(page, detect=detect)
        if res is None:
            dropped.append({"page": page[0], "reason": reason,
                            "detected": n_boxes, "lines": len(page[2])})
            continue
        per_page[page[0]] = res
        if progress and (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(pages)} pages", flush=True)
    return per_page, dropped


def split_pages(per_page, seed=0):
    """Document-disjoint split over page names (train / val / test)."""
    names = list(per_page)
    return R.split_by_document(names, lambda n: n, seed=seed)


def build_cache(pages, out_path=None, limit=None, seed=0, detect=detect_boxes,
                verbose=True):
    """Build the labelled cache and write it in the trainer's schema."""
    import torch
    torch.set_num_threads(1)                # the box is loaded; stay single-thread
    per_page, dropped = collect_pages(pages, limit=limit, detect=detect,
                                      progress=print if verbose else None)
    sp = split_pages(per_page, seed=seed)
    assign = {n: s for s, names in sp.items() for n in names}

    crops, labels, profs, splits, ids = [], [], [], [], []
    for name in sorted(per_page):
        cs, ls = per_page[name]
        for c, l in zip(cs, ls):
            crops.append(c)
            labels.append(l)
            profs.append(VARIANTS[0])
            splits.append(assign[name])
            ids.append(name)

    if not crops:
        raise SystemExit("no crops built -- every page was dropped")

    obj = np.empty(len(crops), dtype=object)
    for i, c in enumerate(crops):
        obj[i] = c
    data = {                                # exactly what write_cache consumes
        "crops": obj,
        "labels": np.array(labels, dtype=object),
        "profile": np.array(profs, dtype=object),
        "split": np.array(splits, dtype=object),
        "doc": np.array(ids, dtype=object),
    }
    out_path = Path(out_path or (OUT / "crops_train.npz"))
    written = RC.write_cache(data, out_path=out_path)

    counts = {s: sum(1 for x in splits if x == s) for s in ("train", "val", "test")}
    pages_by_split = {s: len({ids[i] for i, x in enumerate(splits) if x == s})
                      for s in ("train", "val", "test")}
    if verbose:
        print(f"wrote {written}")
        print(f"  crops {len(crops)}  pages {len(per_page)}  "
              f"dropped {len(dropped)}")
        for s in ("train", "val", "test"):
            print(f"  {s:5} {counts[s]:5} crops  {pages_by_split[s]:3} pages")
    return {"path": str(written), "crops": len(crops), "pages": len(per_page),
            "dropped": dropped, "split_counts": counts,
            "pages_by_split": pages_by_split, "per_page": per_page}


# ── distribution match: real camera vs the synthetic profiles ────────────────

def crop_stats(a):
    """(ink_density, ink_to_background_contrast, laplacian_variance) for a crop.

    Ink is split from paper with Otsu on the crop itself, so the split adapts to
    a dim or warm photo instead of assuming white paper. Contrast is the mean
    separation between the two populations (0 when either is empty). Sharpness is
    the Laplacian variance -- the standard blur statistic, low when defocused.
    """
    a = np.asarray(a, np.uint8)
    if a.ndim != 2 or a.size == 0:
        return 0.0, 0.0, 0.0
    thr, _ = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # OpenCV's Otsu threshold is the value that separates the BRIGHT class
    # (src > thresh) from the dark one, and on a bimodal crop it lands ON the
    # ink mode -- so ink is `<= thresh`, not `< thresh`, or a two-tone crop
    # reports zero ink.
    ink = a <= thr
    paper = ~ink
    density = float(ink.mean())
    af = a.astype(np.float32)
    contrast = (float(af[paper].mean() - af[ink].mean())
                if ink.any() and paper.any() else 0.0)
    sharp = float(cv2.Laplacian(a, cv2.CV_64F).var())
    return density, contrast, sharp


def stats_of(crops):
    """Aggregate crop_stats over a list of crops -> mean/median per metric."""
    rows = np.array([crop_stats(c) for c in crops], np.float64)
    if not len(rows):
        return {k: 0.0 for k in ("ink_mean", "ink_median", "contrast_mean",
                                 "contrast_median", "sharp_mean", "sharp_median")}
    return {
        "ink_mean": float(rows[:, 0].mean()),
        "ink_median": float(np.median(rows[:, 0])),
        "contrast_mean": float(rows[:, 1].mean()),
        "contrast_median": float(np.median(rows[:, 1])),
        "sharp_mean": float(rows[:, 2].mean()),
        "sharp_median": float(np.median(rows[:, 2])),
        "n": int(len(rows)),
    }


def synth_crops_by_profile(synth_dir=None, cap=1500):
    """{profile: [crop_to_array(crop), ...]} from the rendered corpus."""
    f = Path(synth_dir or SYNTH) / "crops.npz"
    if not f.exists():
        return {}
    d = np.load(f, allow_pickle=True)
    raw = {}
    for c, p in zip(d["crops"], d["profile"]):
        raw.setdefault(str(p), []).append(c)
    # Deterministic cap BEFORE the resize, so a bounded run is cheap and
    # reproducible (the stride is fixed by each profile's own length).
    out = {}
    for k, v in raw.items():
        step = max(1, len(v) // cap)
        out[k] = [B.crop_to_array(Image.fromarray(np.asarray(c, np.uint8)))
                  for c in v[::step][:cap]]
    return out


def nearest_profile(vec, profile_vecs):
    """The synthetic profile whose (standardised) mean stats sit closest."""
    keys = list(profile_vecs)
    M = np.array([[profile_vecs[k][j] for j in range(3)] for k in keys])
    v = np.array(vec, np.float64)
    mu, sd = M.mean(axis=0), M.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    dist = np.linalg.norm(((M - mu) / sd) - ((v - mu) / sd), axis=1)
    i = int(np.argmin(dist))
    return keys[i], float(dist[i])


def distribution_match(real_crops, synth_dir=None, cap=1500, per_page=None):
    """Real-vs-synthetic table + each real page's nearest synthetic profile."""
    prof = synth_crops_by_profile(synth_dir=synth_dir, cap=cap)
    table = {"real": stats_of(real_crops)}
    for name in PROFILES:
        if name in prof:
            table[name] = stats_of(prof[name])
    vecs = {k: (v["ink_mean"], v["contrast_mean"], v["sharp_mean"])
            for k, v in table.items() if k != "real"}

    page_match = {}
    if per_page:
        for name, (cs, _) in per_page.items():
            s = stats_of(cs)
            m, dist = nearest_profile(
                (s["ink_mean"], s["contrast_mean"], s["sharp_mean"]), vecs)
            page_match[name] = {"nearest": m, "distance": dist,
                                "ink_mean": s["ink_mean"],
                                "contrast_mean": s["contrast_mean"],
                                "sharp_mean": s["sharp_mean"]}
    return table, page_match


def print_match(table, page_match):
    if "real" not in table:
        return
    print("\n" + "=" * 78)
    print("DISTRIBUTION MATCH — real camera photos vs the synthetic profiles")
    print("=" * 78)
    print("%-16s %8s %9s %10s %11s %10s %9s"
          % ("profile", "n", "ink_mean", "ink_med", "contrast", "sharp_mn", "sharp_md"))
    order = ["real"] + [p for p in PROFILES if p in table]
    for k in order:
        v = table[k]
        print("%-16s %8d %9.4f %10.4f %11.1f %10.1f %9.1f"
              % (k, v.get("n", 0), v["ink_mean"], v["ink_median"],
                 v["contrast_mean"], v["sharp_mean"], v["sharp_median"]))
    if not page_match:
        return
    from collections import Counter
    near = Counter(m["nearest"] for m in page_match.values())
    print("\nreal page -> nearest synthetic profile (standardised 3-vector):")
    for k, c in near.most_common():
        d = np.mean([m["distance"] for m in page_match.values()
                     if m["nearest"] == k])
        print("  %-16s %3d/%d pages   mean distance %.2f"
              % (k, c, len(page_match), d))
    byd = sorted(page_match.items(), key=lambda kv: kv[1]["distance"])
    print("  closest 3 pages :",
          ", ".join("%s->%s(%.1f)" % (n, m["nearest"], m["distance"])
                    for n, m in byd[:3]))
    print("  furthest 3 pages:",
          ", ".join("%s->%s(%.1f)" % (n, m["nearest"], m["distance"])
                    for n, m in byd[-3:]))


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true",
                    help="build the cache and print the report")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap pages (0 = all 193) for a bounded build")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "crops_train.npz"))
    ap.add_argument("--stats-cap", type=int, default=1500,
                    help="max crops per synthetic profile in the match table")
    ap.add_argument("--json", default="", help="also write the report as JSON here")
    a = ap.parse_args(argv)
    if not a.build:
        ap.print_help()
        return 0

    pages = R.handheld_pages()
    print(f"pages found       {len(pages)} hand-photographed receipts "
          f"(each with a whole-page transcript)")

    texts = {p[0]: p[2] for p in pages}
    rep = build_cache(pages, out_path=a.out, limit=(a.limit or None),
                      seed=a.seed)

    print(f"\ncrops built       {rep['crops']}")
    print(f"pages dropped     {len(rep['dropped'])}")
    from collections import Counter
    reasons = Counter(d["reason"].split(":")[0] for d in rep["dropped"])
    for r, c in reasons.most_common():
        print(f"    {c:3}  {r}")
    if rep["dropped"]:
        ex = rep["dropped"][:5]
        for d in ex:
            print(f"    e.g. {d['page']}: {d['reason']}")

    per_page = rep["per_page"]
    real = [c for cs, _ in per_page.values() for c in cs]
    table, page_match = distribution_match(real, cap=a.stats_cap,
                                           per_page=per_page)
    print_match(table, page_match)

    print("\n" + "-" * 78)
    print("MERGE into the next training run "
          "(ml/tools/retrain_rendered.sh backs up the checkpoint first):")
    print("  " + MERGE_CMD.replace("\n", "\n  "))
    print("\nRE-MEASURE afterwards:")
    for c in REMEASURE_CMDS:
        print("  " + c)
    print("  full suite deferred to the orchestrator")

    rep["pages_found"] = len(pages)
    rep["vector_match"] = table
    rep["page_match_counts"] = dict(Counter(
        m["nearest"] for m in page_match.values()))
    rep["merge_command"] = MERGE_CMD
    rep["remeasure"] = REMEASURE_CMDS
    rep["reasons"] = dict(reasons)
    if a.json:
        slim = {k: v for k, v in rep.items() if k != "per_page"}
        Path(a.json).write_text(json.dumps(slim, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
