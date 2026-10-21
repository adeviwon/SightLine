"""
rectify_real.py — LOCALISE and RECTIFY the real hand-photographed receipts with
the four-corner homographies already on disk, then re-run the SAME exact
transcript-count alignment rule.

THE MEASURED PROBLEM
--------------------
`real_degradation_cache.py` could only label 16-20 of the 193 real photographs:
the detector's line-box count did not equal the transcript line count on 177 of
them (138 under-detected, 39 over-detected), and the rule is "drop, never
guess". `line_detect.py` then proved no BANDER can fix it on the raw photos:
perspective gives every text line a different slope, so a single horizontal
projection smears one line into its neighbour and the valley between them never
closes -- the pitch is unmeasurable and the five ink-saturated pages collapse to
1-2 bands. Its own conclusion: the receipt must be LOCALISED and RECTIFIED
first.

THE TOOL WAS ALREADY HERE
-------------------------
`realdata.handheld_homography()` carries the four-corner annotation for every
one of the 193 pages (verified below: 199 entries, keys are the bare numeric
receipt id, all 193 pages covered). Each value is a (4, 2) float32 array of
[P1 P2 P3 P4] = [top-left, top-right, bottom-left, bottom-right] in the ORIGINAL
photograph's pixel coordinates. The lane that built the crop cache deliberately
left them out to stay single-purpose; this lane uses them.

WHAT THIS DOES
--------------
1. rectify() warps each photo to a top-down view with cv2.warpPerspective,
   mapping the four corners to the corners of a rectangle whose size is the
   quad's own mean edge length (see `rect_size` for the convention and why).
2. The SAME detector (`real_degradation_cache.detect_boxes`) runs on the
   rectified page, and the SAME exact-count rule (`align_labels`) decides
   usable/dropped. Nothing is guessed and nothing is trimmed.
3. The bander is swept ONLY where it is now meaningful -- with the perspective
   gone, two variants `line_detect.py` already implemented (`merge_thin` for the
   over-detected / fragmented class, `split_tall` for the under-detected /
   merged class) are compared against the shipped bander by count-match rate.
4. The rectified crops' statistics at the trainer's geometry are matched against
   BOTH the archived shipped synthetic profiles and the re-tuned ones.
5. The corpus is written in the trainer's schema (render_corpus.write_cache),
   with the same document-disjoint split, to a NEW path -- the existing
   data/real_corpus/crops_train.npz is never touched.

USE
---
    bash run.sh py ml/src/rectify_real.py --build
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as B           # noqa: E402  crop_to_array: ONE crop geometry
import line_detect as LD            # noqa: E402  variant_bands: the banders to sweep
import real_degradation_cache as RDC  # noqa: E402  detect_boxes + align_labels + stats
import realdata as R                # noqa: E402  handheld_pages / handheld_homography
import render_corpus as RC          # noqa: E402  write_cache / train_label

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "real_corpus"
CACHE = OUT / "crops_rectified.npz"
SYNTH = ROOT / "data" / "render_corpus"

CROP_H = R.CROP_H                      # 32
MAX_W = R.MAX_W                        # 320
VARIANT = "real"

# The two banders this lane sweeps, one per failure direction. Chosen, not
# searched: `merge_thin` folds the FRAGMENTS that make a page over-detected,
# `split_tall` splits the MERGED bands that make it under-detected. They are the
# paired opposites, they are already implemented in line_detect.py, and no new
# bander is invented here.
SWEEP = ("merge_thin", "split_tall")
PROFILES = ["studio_clean", "handheld_light", "handheld_heavy", "low_light"]


# ── localisation + rectification ─────────────────────────────────────────────

def key_of(name):
    """Page name (`1000-receipt`) -> homography.csv key (`1000`)."""
    return name[:-len("-receipt")] if name.endswith("-receipt") else name


def rect_size(quad):
    """(W, H) of the top-down rectangle the quad maps onto.

    Convention: W is the larger of the two horizontal edges (P1->P2, P3->P4) and
    H the larger of the two vertical edges (P1->P3, P2->P4), each rounded to the
    nearest pixel. Taking the LONGER edge keeps the mapping a scale-1 resample
    -- no axis is smoothly upsampled, which is what would blur the very glyphs
    the recogniser has to read -- and rounding to an int makes the output size a
    pure function of the quad, hence deterministic. This is the standard
    "rectify by the quad's own extent" rule; the alternative (fixing every page
    to one canonical size) would resample a wide receipt and a narrow one by
    different factors and put that difference into the training set.
    """
    p1, p2, p3, p4 = (np.asarray(p, np.float64) for p in quad)
    w = int(round(max(np.linalg.norm(p2 - p1), np.linalg.norm(p4 - p3))))
    h = int(round(max(np.linalg.norm(p3 - p1), np.linalg.norm(p4 - p2))))
    return max(w, 16), max(h, 16)


def rectify(gray, quad):
    """Top-down view of the receipt in `gray`, using the four-corner quad.

    P1..P4 (top-left, top-right, bottom-left, bottom-right) are mapped to the
    rectangle's corners in the same order, so the receipt's reading orientation
    is preserved. INTER_LINEAR with BORDER_CONSTANT(nan-free 0) is fixed, so the
    warp is deterministic; the destination is exactly the quad's rectangle, so
    no page content outside the receipt is carried into the crop.
    """
    gray = np.asarray(gray, np.uint8)
    w, h = rect_size(quad)
    src = np.asarray(quad, np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (w, h), flags=cv2.INTER_LINEAR)


def load_page(page, homographies):
    """(rectified_gray, quad_or_None) for one (name, path, lines) page.

    Falls back to the untouched photograph when a page has no homography, and
    says which happened via the returned quad, so the count of pages that could
    not be rectified is reportable rather than silent.
    """
    name, jp, _lines = page
    gray = np.asarray(Image.open(jp).convert("L"), np.uint8)
    quad = homographies.get(key_of(name))
    if quad is None:
        return gray, None
    return rectify(gray, quad), quad


# ── the bander sweep: shipped + the two variants, on the SAME box tail ───────

def boxes_from_bands(gray, bands):
    """`real_degradation_cache.detect_boxes`' box-construction tail.

    Identical x-extent padding and minimum-ink filter, so a bander's COUNTS are
    comparable with the shipped detector's -- the comparison isolates the bander
    change alone.
    """
    ink = RDC.EO.ink_mask(gray)
    H, W = gray.shape
    out = []
    for (y0, y1) in bands:
        y0, y1 = max(0, int(y0)), min(H, int(y1))
        if y1 - y0 < 2:
            continue
        strip = ink[y0:y1, :]
        cols = np.where(strip.sum(axis=0) > 0)[0]
        if len(cols) < 4:
            continue
        pad = max(2, int((y1 - y0) * 0.10))
        x0 = max(0, int(cols[0]) - pad)
        x1 = min(W, int(cols[-1]) + pad + 1)
        if x1 - x0 < 2:
            continue
        out.append((y0, y1, x0, x1))
    return out


def bander_boxes(gray, flags):
    """Boxes for a bander: shipped when `flags` is empty, else a line_detect
    variant on the very same ink mask and box tail."""
    if not flags:
        return RDC.detect_boxes(gray)
    return boxes_from_bands(gray, LD.variant_bands(gray, set(flags)))


def locate_bander(gray, flags):
    """The OTHER shipped path: eval_ocr.read_page, i.e. the same bander with the
    locate_document override ON. The earlier lane's own note quotes transcript
    count-match 16/193 with it against 20/193 without, so both are reported."""
    return boxes_from_bands(gray, LD.shipped_bands(gray, use_locate=True))


# ── exact-count rule, reused verbatim ────────────────────────────────────────

align_labels = RDC.align_labels        # drop on any mismatch; never trim/guess


def page_crops(page, homographies, bander=bander_boxes, flags=()):
    """One page -> (result, reason, n_boxes, rectified?).

    `result` is (crops, labels) or None when the page must be dropped, exactly
    the contract `real_degradation_cache.page_crops` has -- same crop geometry
    (build_dataset.crop_to_array), same label convention (render_corpus.
    train_label) -- so the model cannot tell this corpus from the rendered one.
    """
    name, _jp, lines = page
    gray, quad = load_page(page, homographies)
    boxes = bander(gray, flags)
    pairs, reason = align_labels(boxes, lines)
    if pairs is None:
        return None, reason, len(boxes), quad is not None
    crops, labels = [], []
    for (y0, y1, x0, x1), line in pairs:
        label = RC.train_label(line)
        if not label:
            continue
        crops.append(B.crop_to_array(Image.fromarray(gray[y0:y1, x0:x1], "L")))
        labels.append(label)
    if not crops:
        return None, "no charset-legible line", len(boxes), quad is not None
    return (crops, labels), None, len(boxes), quad is not None


# ── collection + report ──────────────────────────────────────────────────────

def collect(pages, homographies, bander=bander_boxes, flags=(), progress=False):
    """(per_page, dropped, n_rectified) over a page list."""
    per_page, dropped, n_rect = {}, [], 0
    for i, page in enumerate(pages):
        res, reason, n_boxes, was_rect = page_crops(page, homographies,
                                                    bander=bander, flags=flags)
        n_rect += int(was_rect)
        if res is None:
            dropped.append({"page": page[0], "reason": reason,
                            "detected": n_boxes, "lines": len(page[2])})
            continue
        per_page[page[0]] = res
        if progress and (i + 1) % 25 == 0:
            print("  ... %d/%d pages" % (i + 1, len(pages)), flush=True)
    return per_page, dropped, n_rect


def breakdown(dropped):
    """Still-dropped counts split by direction, which is the actionable part."""
    under = [d for d in dropped if d["detected"] < d["lines"]]
    over = [d for d in dropped if d["detected"] > d["lines"]]
    other = [d for d in dropped if d["detected"] == d["lines"]]
    return {"total": len(dropped), "under": len(under), "over": len(over),
            "other": len(other)}


def count_match(pages, homographies, bander=bander_boxes, flags=()):
    """(match, n, under, over) -- the exact-count rule applied per page."""
    m = u = o = 0
    for page in pages:
        gray, _q = load_page(page, homographies)
        nb = len(bander(gray, flags))
        nl = len(page[2])
        if nb == nl:
            m += 1
        elif nb < nl:
            u += 1
        else:
            o += 1
    return m, len(pages), u, o


def build_cache(per_page, out_path=None, seed=0, verbose=True):
    """Document-disjoint split -> the trainer's cache schema (render_corpus)."""
    sp = RDC.split_pages(per_page, seed=seed)
    assign = {n: s for s, names in sp.items() for n in names}
    crops, labels, profs, splits, ids = [], [], [], [], []
    for name in sorted(per_page):
        for c, l in zip(*per_page[name]):
            crops.append(c)
            labels.append(l)
            profs.append(VARIANT)
            splits.append(assign[name])
            ids.append(name)
    if not crops:
        raise SystemExit("no crops built -- every page was dropped")
    obj = np.empty(len(crops), dtype=object)
    for i, c in enumerate(crops):
        obj[i] = c
    data = {"crops": obj, "labels": np.array(labels, dtype=object),
            "profile": np.array(profs, dtype=object),
            "split": np.array(splits, dtype=object),
            "doc": np.array(ids, dtype=object)}
    out_path = Path(out_path or CACHE)
    written = RC.write_cache(data, out_path=out_path)
    counts = {s: sum(1 for x in splits if x == s) for s in ("train", "val", "test")}
    pages_by_split = {s: len({ids[i] for i, x in enumerate(splits) if x == s})
                      for s in ("train", "val", "test")}
    if verbose:
        print("wrote %s" % written)
        print("  crops %d  pages %d" % (len(crops), len(per_page)))
        for s in ("train", "val", "test"):
            print("  %-5s %5d crops  %3d pages" % (s, counts[s], pages_by_split[s]))
    return {"path": str(written), "crops": len(crops), "pages": len(per_page),
            "split_counts": counts, "pages_by_split": pages_by_split}


# ── distribution match vs the OLD and NEW synthetic profiles ─────────────────

def synth_table():
    """The synthetic (ink, contrast, sharp) rows at TRAINER geometry, old + new.

    Read from `retune_profiles.BEFORE_STATS / AFTER_STATS` -- the archived
    shipped values and the re-tuned ones -- so the comparison needs no rendering
    and is deterministic. `crop_stats` is the same statistic used on the real
    crops below, so the two sides of the table are apples-to-apples (this is the
    same discipline retune_profiles documents).
    """
    import retune_profiles as RP
    return {k: {"before": RP.BEFORE_STATS["trainer"][k],
                "after": RP.AFTER_STATS["trainer"][k]} for k in PROFILES}


def print_match(real_stats):
    tab = synth_table()
    print("\n" + "=" * 78)
    print("DISTRIBUTION MATCH — RECTIFIED real crops at the TRAINER's geometry")
    print("  vs the archived shipped synthetic profiles (before) and the")
    print("  re-tuned ones (after). Same crop_stats for every row.")
    print("=" * 78)
    print("%-16s %8s %9s %10s %11s %10s" %
          ("profile", "n", "ink_mean", "contrast", "sharp_mn", "source"))
    for tag, v in (("real(rectified)", real_stats),):
        print("%-16s %8d %9.4f %10.1f %11.1f %10s" %
              (tag, v["n"], v["ink_mean"], v["contrast_mean"],
               v["sharp_mean"], "measured"))
    print("-" * 78)
    print("%-16s %8s %9s %10s %11s %10s" %
          ("profile", "", "ink_mean", "contrast", "sharp_mn", "source"))
    for k in PROFILES:
        for when in ("before", "after"):
            s = tab[k][when]
            print("%-16s %8s %9.4f %10.1f %11.1f %10s" %
                  (k, "", s["ink"], s["contrast"], s["sharp"],
                   "old" if when == "before" else "NEW"))
    # Does each population bracket reality on all three metrics?
    print("\nbrackets the rectified real population on ALL of (ink, contrast, sharp):")
    for when, label in (("before", "old"), ("after", "NEW")):
        lo = [min(tab[k][when][m] for k in PROFILES)
              for m in ("ink", "contrast", "sharp")]
        hi = [max(tab[k][when][m] for k in PROFILES)
              for m in ("ink", "contrast", "sharp")]
        real = [real_stats["ink_mean"], real_stats["contrast_mean"],
                real_stats["sharp_mean"]]
        ok = all(l <= r <= h for l, r, h in zip(lo, hi, real))
        print("  %-4s  ink [%.3f,%.3f]  contrast [%.1f,%.1f]  sharp [%.1f,%.1f]"
              "  -> %s" % (label, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2],
                           "BRACKETS" if ok else "MISSES (real outside)"))
    return tab


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(CACHE))
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    if not a.build:
        ap.print_help()
        return 0
    import torch
    torch.set_num_threads(1)

    pages = R.handheld_pages()
    homographies = R.handheld_homography()
    if a.limit:
        pages = pages[:a.limit]
    covered = sum(1 for p in pages if key_of(p[0]) in homographies)
    print("pages                    %d hand-photographed receipts" % len(pages))
    print("homographies on disk     %d entries; %d/%d pages covered"
          % (len(homographies), covered, len(pages)))

    # ── BEFORE: the shipped pipeline, un-rectified ───────────────────────────
    before_pp, before_drop, _ = collect(pages, {})
    b_bd = breakdown(before_drop)
    b_crops = sum(len(cs) for cs, _ in before_pp.values())
    print("\nBEFORE (un-rectified, shipped detector)")
    print("  pages usable       %d/%d" % (len(before_pp), len(pages)))
    print("  crops buildable    %d" % b_crops)
    print("  pages dropped      %d  (under %d, over %d, other %d)"
          % (b_bd["total"], b_bd["under"], b_bd["over"], b_bd["other"]))
    loc_pp, loc_drop, _ = collect(pages, {}, bander=locate_bander)
    loc_crops = sum(len(cs) for cs, _ in loc_pp.values())
    loc_bd = breakdown(loc_drop)
    print("  [read_page path, locate_document ON: pages %d/%d, crops %d, "
          "dropped %d (under %d, over %d)]"
          % (len(loc_pp), len(pages), loc_crops, loc_bd["total"],
             loc_bd["under"], loc_bd["over"]))

    # ── AFTER: rectify, same rule, shipped bander ────────────────────────────
    after_pp, after_drop, n_rect = collect(pages, homographies)
    a_bd = breakdown(after_drop)
    a_crops = sum(len(cs) for cs, _ in after_pp.values())
    print("\nAFTER (rectified, shipped detector, SAME exact-count rule)")
    print("  pages rectified    %d/%d" % (n_rect, len(pages)))
    print("  pages usable       %d/%d" % (len(after_pp), len(pages)))
    print("  crops buildable    %d" % a_crops)
    print("  pages dropped      %d  (under %d, over %d, other %d)"
          % (a_bd["total"], a_bd["under"], a_bd["over"], a_bd["other"]))
    if after_drop[:4]:
        for d in after_drop[:4]:
            print("    e.g. %s: %s" % (d["page"], d["reason"]))

    # ── bander sweep, rectified pages only ───────────────────────────────────
    print("\nBANDER SWEEP (rectified pages; shipped bander vs two line_detect "
          "variants)\n  metric = pages whose detected box count == transcript "
          "line count")
    sweep = {}
    for tag, flags in [("shipped", ())] + [(v, (v,)) for v in SWEEP]:
        m, n, u, o = count_match(pages, homographies, flags=flags)
        sweep[tag] = {"match": m, "n": n, "under": u, "over": o,
                      "rate": m / n if n else 0.0}
        print("  %-14s %5d/%-4d %.3f   (under %d, over %d)"
              % (tag, m, n, m / n if n else 0.0, u, o))
    best = max(sweep, key=lambda k: (sweep[k]["match"], -len(k)))
    print("  best: %s  %d/%d  %s"
          % (best, sweep[best]["match"], sweep[best]["n"],
             "the shipped bander already wins" if best == "shipped"
             else "+%d pages over shipped" % (sweep[best]["match"]
                                              - sweep["shipped"]["match"])))

    # ── best bander -> the corpus that gets written ──────────────────────────
    best_flags = () if best == "shipped" else (best,)
    if best_flags:
        after_pp, after_drop, n_rect = collect(pages, homographies,
                                               flags=best_flags)
        a_bd = breakdown(after_drop)
        a_crops = sum(len(cs) for cs, _ in after_pp.values())
        print("\n  (rebuilding corpus with the best bander: %s)" % best)
        print("  pages usable %d/%d   crops %d   dropped %d (under %d, over %d)"
              % (len(after_pp), len(pages), a_crops, a_bd["total"],
                 a_bd["under"], a_bd["over"]))

    # ── distribution match ───────────────────────────────────────────────────
    real = [c for cs, _ in after_pp.values() for c in cs]
    real_stats = RDC.stats_of(real)
    tab = print_match(real_stats)
    from collections import Counter as C2
    reasons = C2(d["reason"].split(":")[0] for d in after_drop)
    print("\nstill-dropped reasons:")
    for r, c in reasons.most_common():
        print("    %3d  %s" % (c, r))

    rep = build_cache(after_pp, out_path=a.out, seed=a.seed,
                      verbose=True)

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"before": {"pages": len(before_pp), "crops": b_crops,
                        "dropped": b_bd},
             "after": {"pages": len(after_pp), "crops": a_crops,
                       "dropped": a_bd},
             "bander_sweep": sweep, "best_bander": best,
             "real_stats": real_stats, "synth": tab,
             "cache": rep}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
