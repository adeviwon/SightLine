"""
line_detect.py — measure the shipped LINE DETECTOR against an EXACT ground
truth, then try to fix the biggest measured loss in the project.

WHY THIS EXISTS
---------------
The previous lane built labelled crops from 193 real hand-photographed receipts
and had to DROP 177 of them (91.7%): the page's detected line-box count did not
match its transcript line count (138 under-detected, 39 over-detected). Every
downstream number — fields spoken, tokens read, whether the app spots anything
at all — is capped by whether the line detector found the line. This file
measures the detector honestly and tries the fixes, one flag at a time.

THE GROUND TRUTH, AND WHY IT IS NOT CIRCULAR
--------------------------------------------
A projection profile cannot tell you what it missed: it has no notion of a line
it never saw. The render corpus can. `render_corpus.py` DRAWS every line with
`capture.render_document`, whose layout (`render_corpus.line_bands`) places
line i at a computable y. That is a label nobody detected — it is where the
renderer PUT the ink. So for every rendered page we know the true line boxes and
can compute true recall AND precision at IoU 0.5, plus the count-mismatch rate
that did the damage.

The boxes are not stored in crops.npz (it carries crops/labels/doc/profile/
split only), so they are recovered by RE-RENDERING the same pages with the same
seed convention the corpus used: `corpus_test_pages` replays the exact RNG
stream of `render_corpus.build` (one `rng.choice(scales)` and one
`rng.randrange(profiles)` per page), so the scale and profile of every page are
the ones the corpus actually used, and `page_truth` reads the ink extent of each
rendered line from the CLEAN render.

    bash run.sh py ml/src/line_detect.py --split test --limit-pages 120

WHAT SHIPS HERE
---------------
  shipped_bands        the detector as it ships (eval_ocr.read_page's front end)
  VARIANTS             seven candidate fixes, each behind its own flag
  match_boxes          one-to-one greedy matching at IoU 0.5
  paired_verdict       compare_runs.compare's own IMPROVED/REGRESSED/INCONCLUSIVE
  field_recovery_proxy the detector's model-free contribution to field recovery:
                       did the detector find the line each expected field is on?

No model inference anywhere: this is pure geometry.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import capture as C            # noqa: E402
import compare_runs as CR      # noqa: E402
import eval_ocr as EO          # noqa: E402
import evaluate as E           # noqa: E402
import realdata as RD          # noqa: E402
import render_corpus as RC     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IoU_MIN = 0.5

# ── the shipped detector (ported, not re-implemented) ────────────────────────
#
# The lining path eval_handheld uses lives in eval_ocr.read_page: trim the page
# to its ink extent (remembering the offset), project, band the projection with
# line_bands, then REPLACE the bands with the ones inside a located document
# quad when locate_document finds one. We call those functions rather than
# copying them, so the shipped baseline can never drift from the shipped code.


def shipped_bands(gray, use_locate=True):
    """The shipped detector's line boxes (y0, y1), in absolute page rows.

    Faithful to eval_ocr.read_page. `use_locate=False` drops the
    locate_document override so the BANDING step can be compared one change at
    a time (the override bypasses banding entirely, which would make a
    band-level variant unmeasurable on any page where it fires). The caller
    reports both so the override's own effect size is visible.
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
    if use_locate:
        quad = EO.locate_document(gray)
        if quad is not None:
            located = EO.bands_in_document(gray, quad)
            if located:
                bands = located
    return bands


def _front_end(gray):
    """(proj, off, ink) — the trimmed row-ink profile the shipped bander sees."""
    H, W = gray.shape
    ink = EO.ink_mask(gray)
    binv = (ink.astype(np.uint8)) * 255
    off = 0
    ys = np.where(binv.sum(axis=1) > W * 0.02)[0]
    if len(ys) > 20:
        off = max(0, ys[0] - 5)
        binv = binv[off:min(H, ys[-1] + 6), :]
    proj = (binv > 0).sum(axis=1)
    return proj, off, ink


# ── helpers a bander needs ───────────────────────────────────────────────────

def _median_pitch(bands, fallback=30):
    """Median centre-to-centre spacing of a page's bands — the line pitch (px)."""
    if len(bands) < 2:
        return int(fallback)
    c = [(a + b) / 2.0 for a, b in bands]
    return int(round(float(np.median(np.diff(c))))) if len(c) > 1 else int(fallback)


def _autopitch(proj, lo=8, hi=80, fallback=30):
    """Line pitch from the autocorrelation of the detrended profile.

    Used where there are no bands to measure spacing from (the saturated page),
    because a periodic line grid still shows up as an autocorrelation peak even
    when the profile never drops to zero.
    """
    x = proj.astype(np.float64) - float(proj.mean())
    if not np.any(x):
        return fallback
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return fallback
    return int(lo + int(np.argmax(ac[lo:hi])))


def _local_minima(proj, y0, y1, n_want):
    """`n_want` split rows inside (y0, y1) at the profile's deepest valleys."""
    lo, hi_ = y0 + 2, y1 - 2
    if hi_ - lo < 3 or n_want < 1:
        return []
    seg = proj[lo:hi_]
    order = np.argsort(seg)[:max(1, int(n_want))]
    return sorted(int(lo + i) for i in order)


# ── the banders, one per candidate fix ───────────────────────────────────────

def _bands_hyst(proj, thr, gap_min, min_h):
    """Line_bands with GAP HYSTERESIS: a run only ends after `gap_min` quiet
    rows. Short dips inside a glyph no longer fragment one line into several."""
    bands, start, run = [], None, 0
    for i, v in enumerate(proj):
        if v > thr:
            if start is None:
                start = i
            run = 0
        elif start is not None:
            run += 1
            if run >= gap_min:
                end = i - run + 1
                if end - start >= min_h:
                    bands.append((start, end))
                start, run = None, 0
    if start is not None and len(proj) - start >= min_h:
        bands.append((start, len(proj)))
    return bands


def _shear_proj(ink, slope):
    """Row-ink profile of the ink mask SHEARED flat at `slope` (dy/dx).

    A tilted receipt drawn across a straight horizontal projection smears one
    line over extra rows and can touch its neighbour, so the valleys close up.
    Undoing the tilt first is what makes the valleys reappear.
    """
    H, W = ink.shape
    xc = W / 2.0
    xs = np.arange(W)
    dy = np.round(slope * (xs - xc)).astype(int)
    if not np.any(dy):
        return ink.sum(axis=1)
    shifted = np.zeros_like(ink)
    for x in range(W):
        d = int(dy[x])
        if d == 0:
            shifted[:, x] = ink[:, x]
        elif d > 0:
            shifted[d:, x] = ink[:H - d, x]
        else:
            shifted[:H + d, x] = ink[-d:, x]
    return shifted.sum(axis=1)


def _slope_of(ink):
    """Least-squares dy/dx over ink pixels — the page's tilt."""
    ys, xs = np.nonzero(ink)
    if len(xs) < 200:
        return 0.0
    # one centroid per occupied column keeps glyph shape from dominating
    cols = np.unique(xs)
    cent = np.array([ys[xs == c].mean() for c in cols])
    if len(cols) < 20:
        return 0.0
    a = np.polyfit(cols.astype(np.float64), cent, 1)
    return float(a[0])


def _bands_cc(ink, min_area=6, overlap=0.25):
    """Line boxes from CONNECTED COMPONENTS grouped by vertical overlap.

    A glyph is a component; components whose y-ranges overlap belong to the same
    line. This does not care about projection valleys at all, which is exactly
    the signal the saturated pages lack.
    """
    import cv2
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), 8)
    comps = [tuple(int(v) for v in stats[i])
             for i in range(1, n) if stats[i][4] >= min_area]
    comps.sort(key=lambda s: s[1])
    lines = []
    for x, y, w, h, area in comps:
        placed = False
        for ln in lines:
            lo = max(ln[0], y)
            hi = min(ln[1], y + h)
            if hi - lo > overlap * min(ln[1] - ln[0], h):
                ln[0], ln[1] = min(ln[0], y), max(ln[1], y + h)
                placed = True
                break
        if not placed:
            lines.append([y, y + h])
    return [(a, b) for a, b in lines if b - a >= 4]


def _merge_thin(bands, min_h):
    """Fold bands thinner than `min_h` into their nearest neighbour."""
    if not bands:
        return bands
    out = [list(b) for b in bands]
    i = 0
    while i < len(out):
        if out[i][1] - out[i][0] < min_h and len(out) > 1:
            if i + 1 < len(out) and (i == 0 or (out[i + 1][0] - out[i][1])
                                     <= (out[i][0] - out[i - 1][1])):
                out[i + 1][0] = min(out[i + 1][0], out[i][0])
                out.pop(i)
                continue
            if i > 0:
                out[i - 1][1] = max(out[i - 1][1], out[i][1])
                out.pop(i)
                continue
        i += 1
    return [(a, b) for a, b in out]


def _split_tall(proj, bands, max_h):
    """Split bands taller than `max_h` at their deepest interior valleys."""
    out = []
    for y0, y1 in bands:
        h = y1 - y0
        if h <= max_h:
            out.append((y0, y1))
            continue
        n = max(1, int(round(h / max_h)) - 1)
        cuts = _local_minima(proj, y0, y1, n)
        prev = y0
        for c in cuts + [y1]:
            if c - prev >= 4:
                out.append((prev, c))
            prev = c
    return out


def _sat_bands(proj, thr, pitch):
    """Periodic segmentation for the INK-SATURATED page.

    When the row profile never drops (no valley anywhere) no threshold splits
    it. But the line GRID is still periodic, so cut on the pitch: boundaries at
    the profile's minima nearest each multiple of the estimated pitch.
    """
    n = len(proj)
    on = np.where(proj > thr)[0]
    if len(on) < 2:
        return []
    start = int(on[0])
    bands, anchor = [], start
    while anchor < n - 4:
        target = anchor + pitch
        if target >= n - 2:
            bands.append((anchor, n))
            break
        w = max(2, int(pitch * 0.3))
        lo, hi_ = max(anchor + 3, target - w), min(n - 2, target + w)
        cut = int(lo + int(np.argmin(proj[lo:hi_]))) if hi_ > lo else target
        if cut - anchor >= 4:
            bands.append((anchor, cut))
        anchor = cut
    return bands


VARIANTS = {
    "adaptive_gap": "gap hysteresis sized to the median pitch (a short dip is "
                    "not a line break)",
    "slope_aware": "shear the ink projection flat before banding (a tilted "
                   "receipt smears line valleys closed)",
    "cc_group": "connected components grouped by vertical overlap (ignores "
                "projection valleys entirely)",
    "merge_thin": "merge bands thinner than half the pitch (fragments are not "
                  "lines)",
    "split_tall": "split bands taller than 1.6x the pitch at their deepest "
                  "valleys (merged lines)",
    "ink_floor": "reject bands whose ink density is below a floor "
                 "(texture / shadow, not text)",
    "sat_fix": "pitch-periodic banding when the profile has no valley "
               "(1002/1005/1009/1019 collapse)",
}


def variant_bands(gray, flags):
    """Bands for one or more variants, sharing the shipped front end.

    Every variant starts from the SAME trimmed profile and base bands the
    shipped bander produces, and changes exactly one thing, so a paired
    comparison isolates that change. `flags` is a set of VARIANTS keys.
    """
    proj, off, ink = _front_end(gray)
    base = [(a, b) for a, b in EO.line_bands(proj)]
    pitch = _median_pitch(base, _autopitch(proj))
    thr = max(2, int(proj.max() * 0.04))
    bands = base

    if "slope_aware" in flags:
        slope = _slope_of(ink)
        sproj = _shear_proj(ink, slope)
        bands = [(a, b) for a, b in EO.line_bands(sproj)]
        pitch = _median_pitch(bands, pitch)

    if "cc_group" in flags:
        bands = _bands_cc(ink)
        return [(a + off, b + off) for a, b in bands]

    if "adaptive_gap" in flags:
        gap_min = max(2, int(0.25 * pitch))
        bands = _bands_hyst(proj, thr, gap_min, 8)

    # Saturation is a property of the profile, not a flag request: the
    # documented failure is ink in 95-100% of rows (profile minimum 3 of 93 on
    # 1019-receipt), so the valley test is "almost every row has ink", not
    # "the minimum is high" — the minimum stays small even when it is
    # everywhere. Every bander collapses when there is no valley, so the fix
    # engages whenever it is asked for.
    if "sat_fix" in flags:
        on_frac = float((proj > thr).sum()) / max(len(proj), 1)
        if on_frac > 0.85:
            # The base bands are exactly what collapsed, so their spacing is a
            # useless pitch estimate (2 bands on 1019-receipt give 175px). The
            # autocorrelation pitch is the only independent estimate available.
            bands = _sat_bands(proj, thr, _autopitch(proj))

    if "merge_thin" in flags:
        bands = _merge_thin(bands, max(8, int(0.5 * pitch)))

    if "split_tall" in flags:
        bands = _split_tall(proj, bands, max(10, int(1.6 * pitch)))

    if "ink_floor" in flags:
        H, W = ink.shape
        keep = []
        for y0, y1 in bands:
            a, b = max(0, y0), min(H, y1)
            if b - a < 1:
                continue
            cov = ink[a:b].sum() / float((b - a) * W)
            if cov >= 0.005:
                keep.append((y0, y1))
        bands = keep

    return [(a + off, b + off) for a, b in bands]


# ── the exact ground truth ───────────────────────────────────────────────────

def page_truth(clean_gray, lines, scale):
    """Exact (y0, y1) of every rendered line, from the CLEAN render.

    The band comes from the renderer's own layout (`RC.line_bands`); the tight
    extent is the ink inside it (threshold 170, the same the corpus crop builder
    uses). This is a DRAWN label — no detector touched it.
    """
    H, _W = clean_gray.shape
    truth = []
    for i, (y0, y1) in enumerate(RC.line_bands(lines, scale)):
        y0, y1 = max(0, y0), min(H, y1)
        if y1 - y0 < 3:
            continue
        ink = clean_gray[y0:y1] < 170
        rows = np.where(ink.sum(axis=1) > 0)[0]
        if len(rows) < 3:
            continue
        truth.append({"line": i, "box": (y0 + int(rows[0]), y0 + int(rows[-1]) + 1),
                      "text": lines[i]})
    return truth


def corpus_test_pages(limit_pages=0, n_pages=1500, seed=0, width=900,
                      scales=(0.85, 1.0, 1.15), profiles=None):
    """Re-render the render_corpus TEST pages with their exact corpus geometry.

    Replays render_corpus.build's RNG stream so (scale, profile) per page match
    what the corpus on disk actually used, then re-renders each test page. The
    crop ordering/identity is not needed: the geometry is.
    """
    rng = random.Random(seed)
    page_ids = ["%s-%04d" % (dtype, i)
                for i in range(n_pages) for _, dtype in C.DOCS]
    splits = RC.assign_splits(page_ids)
    profs = list(profiles or RC.PROFILES[:4])
    geom = {}
    for pid in page_ids:
        scale = rng.choice(scales)
        prof = profs[rng.randrange(len(profs))]
        geom[pid] = (scale, prof)

    test_ids = [p for p in page_ids if splits[p] == "test"]
    if limit_pages:
        test_ids = test_ids[:limit_pages]

    pages = []
    for pid in test_ids:
        dtype = pid.rsplit("-", 1)[0]
        lines = next(l for l, d in C.DOCS if d == dtype)
        scale, profile = geom[pid]
        clean = C.render_document(lines, width=width, seed=RC.page_seed(pid),
                                  scale=scale, font=RC.font_for_page(pid))
        deg = C.apply_profile(clean, seed=RC.page_seed(pid, 1),
                              **C.CAPTURE_PROFILES[profile])
        gray = np.asarray(deg.convert("L"), dtype=np.uint8)
        clean_gray = np.asarray(clean.convert("L"), dtype=np.uint8)
        pages.append({"doc": pid, "doc_type": dtype, "profile": profile,
                      "gray": gray,
                      "truth": page_truth(clean_gray, lines, scale)})
    return pages


# ── matching + metrics ───────────────────────────────────────────────────────

def iou_y(a, b):
    """1-D IoU of two (y0, y1) intervals — the line-detection IoU, since every
    box spans the full page width."""
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(truth, pred, iou_min=IoU_MIN):
    """Greedy one-to-one matching at IoU >= iou_min. Returns (gt->pred, pred->gt,
    matched_count)."""
    cand = []
    for i, g in enumerate(truth):
        for j, p in enumerate(pred):
            v = iou_y(g, p)
            if v >= iou_min:
                cand.append((v, i, j))
    cand.sort(reverse=True)
    g2p, p2g = {}, {}
    for _v, i, j in cand:
        if i in g2p or j in p2g:
            continue
        g2p[i] = j
        p2g[j] = i
    return g2p, p2g, len(g2p)


def score_page(truth, pred):
    g2p, p2g, hit = match_boxes([t["box"] for t in truth], pred)
    return {"n_gt": len(truth), "n_pred": len(pred), "hit": hit,
            "matched_gt": g2p, "mismatch": len(truth) != len(pred)}


def wilson(k, n):
    return CR.wilson(k, n)


# ── paired comparison, judged by compare_runs' own vocabulary ─────────────────

def _rec(page, expect_ids, got_ids):
    return {"profile": page["profile"], "doc": page["doc"],
            "category": page["doc_type"], "expected_category": page["doc_type"],
            "expected_fields": [str(x) for x in expect_ids],
            "fields": [str(x) for x in got_ids]}


def _artifact(records):
    """A hackathon-shaped artifact whose field_recovery@pipeline items are one
    per (page, line or field). Feeding it to compare_runs.compare gives the
    repo's own discordant-pair-counted, exact-McNemar verdict."""
    n = sum(len(r["expected_fields"]) for r in records)
    k = sum(len(r["fields"]) for r in records)
    return {"classifier": {}, "missing": [],
            "pipeline": {"n": n, "field_recovery": (k / n) if n else 0.0,
                         "records": records}}


def paired_verdict(old_recs, new_recs, metric="field_recovery@pipeline"):
    """IMPROVED / REGRESSED / INCONCLUSIVE from compare_runs.compare itself —
    the repo's discordant-pair-counted, exact-McNemar verdict."""
    _so, _sn, rows = CR.compare(_artifact(old_recs), _artifact(new_recs))
    row = next(r for r in rows if r["metric"] == metric)
    return {"verdict": row["verdict"], "method": row["method"],
            "p": row["p"], "discordant": row["discordant"],
            "k_old": row["k_old"], "k_new": row["k_new"], "n": row["n"],
            "delta": row["delta"], "old_ci": row["old_ci"],
            "new_ci": row["new_ci"]}


# ── the harness ──────────────────────────────────────────────────────────────

def line_records(pages, detect):
    """Per (page, gt line) matched-or-not — the items a paired test needs."""
    out, per_page = [], []
    for pg in pages:
        pred = detect(pg["gray"])
        s = score_page(pg["truth"], pred)
        per_page.append((pg, pred, s))
        out.append(_rec(pg, [i for i in range(len(pg["truth"]))],
                        list(s["matched_gt"].keys())))
    return out, per_page


def field_records(pages, per_page):
    """Per (page, expected field) whether the field's LINE was matched.

    A model-free ceiling on field recovery: a field on a line the detector never
    found cannot be recovered by any recogniser or policy downstream. The line
    that carries a field is found by normalised substring match of the field
    token against the page's rendered line texts.
    """
    recs = []
    for pg, _pred, s in per_page:
        expect, got = [], []
        for f in E.FIELDS.get(pg["doc_type"], []):
            nf = E.norm(f)
            carrier = next((i for i, t in enumerate(pg["truth"])
                            if nf and nf in E.norm(t["text"])), None)
            if carrier is None:
                continue
            expect.append(f)
            if carrier in s["matched_gt"]:
                got.append(f)
        if expect:
            recs.append(_rec(pg, expect, got))
    return recs


def _agg(per_page):
    n_gt = sum(s["n_gt"] for _p, _pr, s in per_page)
    n_pred = sum(s["n_pred"] for _p, _pr, s in per_page)
    hit = sum(s["hit"] for _p, _pr, s in per_page)
    mism = sum(1 for _p, _pr, s in per_page if s["mismatch"])
    return {"pages": len(per_page), "n_gt": n_gt, "n_pred": n_pred,
            "hit": hit, "mismatch": mism}


def _pct(k, n):
    return "-" if not n else "%.3f" % (k / n)


def print_profile_table(title, per_page, detect_name):
    print("\n%s  [%s]" % (title, detect_name))
    print("%-16s %5s %7s %6s %7s %-16s %6s %7s %-16s %10s %-16s"
          % ("profile", "pages", "gt", "pred", "recall", "recall 95% CI",
             "prec", "prec 95% CI", "match@0.5", "mismatch", "mism 95% CI"))
    profiles = sorted({p["profile"] for p, _pr, _s in per_page})
    for prof in profiles + ["ALL"]:
        sub = per_page if prof == "ALL" else [t for t in per_page
                                              if t[0]["profile"] == prof]
        a = _agg(sub)
        rec_ci = wilson(a["hit"], a["n_gt"])
        pre_ci = wilson(a["hit"], a["n_pred"])
        mm_ci = wilson(a["mismatch"], a["pages"])
        print("%-16s %5d %7d %6d %7s %-16s %6s %7s %-16s %5d/%-4d %-16s"
              % (prof, a["pages"], a["n_gt"], a["n_pred"],
                 _pct(a["hit"], a["n_gt"]), "[%.3f,%.3f]" % rec_ci,
                 _pct(a["hit"], a["n_pred"]), "[%.3f,%.3f]" % pre_ci,
                 _pct(a["hit"], a["n_gt"]),
                 a["mismatch"], a["pages"], "[%.3f,%.3f]" % mm_ci))
    return None


def run_corpus(args):
    print("SightLine — LINE DETECTOR vs an EXACT (drawn) ground truth")
    print("split=%s  render test pages capped at %d  IoU match >= %.2f"
          % (args.split, args.limit_pages, IoU_MIN))
    pages = corpus_test_pages(limit_pages=args.limit_pages)
    n_gt = sum(len(p["truth"]) for p in pages)
    print("re-rendered %d corpus pages (same seed convention as "
          "render_corpus.build)  %d ground-truth lines\n" % (len(pages), n_gt))

    # baseline: the shipped detector, both the full read_page path and the
    # banding core (locate override off).
    base_recs, base_pp = line_records(
        pages, lambda g: shipped_bands(g, use_locate=False))
    loc_recs, loc_pp = line_records(
        pages, lambda g: shipped_bands(g, use_locate=True))
    print_profile_table("BASELINE — shipped detector (banding core)",
                        base_pp, "eval_ocr line_bands + ink_mask")
    print_profile_table("BASELINE — shipped detector (locate_document on)",
                        loc_pp, "eval_ocr.read_page, locate override")
    base_fields = field_records(pages, base_pp)
    loc_fields = field_records(pages, loc_pp)
    fa = _agg_fields(base_fields)
    print("\n  downstream PROXY (model-free): field-bearing lines matched "
          "(shipped) = %d/%d %s  %s"
          % (fa["k"], fa["n"], _pct(fa["k"], fa["n"]),
             "[%.3f,%.3f]" % wilson(fa["k"], fa["n"])))

    # The locate_document override is part of what ships, so a fix to it is a
    # fix to the shipped detector. Paired here: read_page (override ON, old) vs
    # the banding core (override OFF, new).
    loc_vs_core = paired_verdict(loc_recs, base_recs)
    loc_f = _agg_fields(loc_fields)
    print("\nPAIRED FIX CANDIDATE — drop the locate_document override:")
    print("  line recall core %d/%d vs read_page %d/%d -> %s   pairs %s"
          % (_agg(base_pp)["hit"], _agg(base_pp)["n_gt"],
             _agg(loc_pp)["hit"], _agg(loc_pp)["n_gt"],
             loc_vs_core["verdict"], loc_vs_core["discordant"]))
    print("  field-bearing lines %d/%d (locate on) -> %d/%d (locate off)  %s"
          % (loc_f["k"], loc_f["n"], fa["k"], fa["n"],
             paired_verdict(loc_fields, base_fields)["verdict"]))

    # variants, paired against the shipped banding core
    results = {}
    order = [v for v in VARIANTS if v in set(args.variants.split(","))] \
        if args.variants else list(VARIANTS)
    print("\nvariant comparison (paired, exact McNemar, same pages):")
    print("%-14s %8s %9s %9s %10s %16s %22s"
          % ("variant", "recallΔ", "precΔ", "hit old/new", "pairs b/c",
             "p", "verdict"))
    for v in order:
        recs, pp = line_records(pages, lambda g, _v=v: variant_bands(g, {_v}))
        pv = paired_verdict(base_recs, recs)
        vf = _agg_fields(field_records(pages, pp))
        a = _agg(pp)
        b = _agg(base_pp)
        results[v] = {"line": pv, "agg": a, "fields": vf,
                      "recall": a["hit"] / max(a["n_gt"], 1),
                      "base_recall": b["hit"] / max(b["n_gt"], 1),
                      "mismatch": a["mismatch"] / max(a["pages"], 1),
                      "base_mismatch": b["mismatch"] / max(b["pages"], 1)}
        print("%-14s %+8.3f %+9.3f %4d/%-4d %10s %16s %22s"
              % (v, results[v]["recall"] - results[v]["base_recall"],
                 a["hit"] / max(a["n_pred"], 1) - b["hit"] / max(b["n_pred"], 1),
                 a["hit"], b["hit"],
                 "%s/%s" % ((pv["discordant"] or ("-", "-"))[0],
                            (pv["discordant"] or ("-", "-"))[1]),
                 "p=%.4g" % pv["p"] if pv["p"] is not None else "-",
                 pv["verdict"]))
        print("%-14s   field-lines %d/%d %s   count-mismatch %.3f -> %.3f"
              % ("", vf["k"], vf["n"], _pct(vf["k"], vf["n"]),
                 results[v]["base_mismatch"], results[v]["mismatch"]))

    # best variant by line recall, then its downstream field effect, and a
    # combined variant of the winners.
    if results:
        best = max(results, key=lambda k: results[k]["recall"])
        print("\nBEST single variant by line recall: %s" % best)
        recs, pp = line_records(pages, lambda g, _v=best: variant_bands(g, {_v}))
        pv = paired_verdict(base_recs, recs)
        bf = _agg_fields(field_records(pages, pp))
        print("  line recall   %.3f -> %.3f   %s"
              % (_agg(base_pp)["hit"] / max(_agg(base_pp)["n_gt"], 1),
                 results[best]["recall"], pv["verdict"]))
        print("  field-lines   %d/%d -> %d/%d   %s"
              % (fa["k"], fa["n"], bf["k"], bf["n"],
                 paired_verdict(base_fields, field_records(pages, pp))["verdict"]))
        fpv = paired_verdict(base_fields, field_records(pages, pp))
        print("  downstream field-recovery proxy paired: %s/%s -> %s/%s  "
              "delta %+.3f  pairs %s/%s  p=%.4g  %s"
              % (fpv["k_old"], fpv["n"], fpv["k_new"], fpv["n"],
                 fpv["delta"] or 0.0,
                 (fpv["discordant"] or ("-", "-"))[0],
                 (fpv["discordant"] or ("-", "-"))[1],
                 fpv["p"] if fpv["p"] is not None else 1.0, fpv["verdict"]))

        winners = {best}
        if results[best]["fields"]["k"] > fa["k"]:
            winners.add("ink_floor")
        comb = set(winners)
        recs_c, pp_c = line_records(pages, lambda g: variant_bands(g, comb))
        cf = _agg_fields(field_records(pages, pp_c))
        ac = _agg(pp_c)
        print("\nCOMBINED variant %s:" % sorted(comb))
        print("  line recall %d/%d %s   precision %d/%d %s   mismatch %d/%d"
              % (ac["hit"], ac["n_gt"], _pct(ac["hit"], ac["n_gt"]),
                 ac["hit"], ac["n_pred"], _pct(ac["hit"], ac["n_pred"]),
                 ac["mismatch"], ac["pages"]))
        print("  field-lines %d/%d %s -> %d/%d   %s"
              % (fa["k"], fa["n"], _pct(fa["k"], fa["n"]), cf["k"], cf["n"],
                 paired_verdict(base_fields, field_records(pages, pp_c))["verdict"]))
    return results


def _agg_fields(field_recs):
    """k/n of expected fields whose carrying line the detector matched."""
    n = sum(len(r["expected_fields"]) for r in field_recs)
    k = sum(len([f for f in r["fields"] if f in r["expected_fields"]])
            for r in field_recs)
    return {"k": k, "n": n}


def _real_counts(pages, detect):
    """[(name, boxes, transcript_lines)] for one detector over the real pages."""
    from PIL import Image
    out = []
    for name, jp, lines in pages:
        gray = np.asarray(Image.open(jp).convert("L"), dtype=np.uint8)
        nl = len([w for w in lines if len(w.strip()) >= 3])
        out.append((name, len(detect(gray)), nl))
    return out


def _count_records(counts):
    """One item per page: did the detected box count equal the transcript line
    count? This is EXACTLY the rule that dropped 177/193 pages upstream, so it
    is the paired statistic to judge a real-page detector change by."""
    return [{"profile": "real", "doc": nm, "category": "receipt",
             "expected_category": "receipt", "expected_fields": ["count"],
             "fields": ["count"] if nb == nl else []}
            for nm, nb, nl in counts]


def run_real(args):
    """The 177/193 statistic itself, on the real hand-photographed receipts.

    Detection only — no model inference. Count-mismatch is exactly the rule that
    dropped 177 of 193 pages: detected box count != transcript line count. There
    is no per-line box ground truth for photographs (only the transcript count),
    so this is the honest metric available here, and it is paired per page.
    """
    pages = RD.handheld_pages()
    if not pages:
        print("\nreal handheld receipts not present; skipped")
        return
    pages = pages[:args.real_pages] if args.real_pages else pages
    print("\nREAL handheld receipts (ground truth = transcript line COUNT)")
    print("  (detected box count != transcript count is the rule that dropped "
          "177/193 pages upstream)")

    shipped_core = _real_counts(pages, lambda g: shipped_bands(g, use_locate=False))
    shipped_loc = _real_counts(pages, lambda g: shipped_bands(g, use_locate=True))

    def summarize(tag, counts):
        n = len(counts)
        mism = sum(1 for _nm, nb, nl in counts if nb != nl)
        under = sum(1 for _nm, nb, nl in counts if nb < nl)
        over = sum(1 for _nm, nb, nl in counts if nb > nl)
        print("  %-22s count-match %d/%d %s   mismatch %d/%d %.3f %s   "
              "(under %d, over %d)"
              % (tag, n - mism, n, _pct(n - mism, n), mism, n, mism / n,
                 "[%.3f,%.3f]" % wilson(mism, n), under, over))

    summarize("shipped (read_page)", shipped_loc)
    summarize("shipped (banding core)", shipped_core)
    pv = paired_verdict(_count_records(shipped_loc), _count_records(shipped_core))
    print("  paired: read_page (locate ON) -> banding core (locate OFF)  "
          "%s  pairs b/c %s" % (pv["verdict"], pv["discordant"]))

    print("\n  variant count-match vs shipped banding core (paired, 193 real pages):")
    print("  %-14s %10s %9s %-14s %16s %22s"
          % ("variant", "match k/n", "mismatch", "mismatch CI", "pairs b/c",
             "verdict"))
    for v in (list(VARIANTS) if not args.variants
              else [x for x in VARIANTS if x in set(args.variants.split(","))]):
        counts = _real_counts(pages, lambda g, _v=v: variant_bands(g, {_v}))
        n = len(counts)
        mism = sum(1 for _nm, nb, nl in counts if nb != nl)
        pv = paired_verdict(_count_records(shipped_core),
                            _count_records(counts))
        print("  %-14s %5d/%-4d %9.3f %-14s %16s %22s"
              % (v, n - mism, n, mism / n, "[%.3f,%.3f]" % wilson(mism, n),
                 "%s/%s" % ((pv["discordant"] or ("-", "-"))[0],
                            (pv["discordant"] or ("-", "-"))[1]),
                 pv["verdict"]))

    named = [(nm, nb, nl) for nm, nb, nl in shipped_loc if nb <= 2 and nl >= 10]
    if named:
        print("\n  INK-SATURATED COLLAPSE (<=2 boxes for a >=10-line page), "
              "shipped read_page:")
        for nm, nb, nl in named:
            print("    %-16s boxes=%d  transcript=%d" % (nm, nb, nl))


def main(argv=None):
    ap = argparse.ArgumentParser(description="line detector vs an exact GT")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-pages", type=int, default=120,
                    help="cap render test pages (0 = all)")
    ap.add_argument("--real-pages", type=int, default=193,
                    help="cap real handheld receipts (0 = all)")
    ap.add_argument("--variants", default="",
                    help="comma list to run (default all)")
    ap.add_argument("--no-real", action="store_true")
    a = ap.parse_args(argv)
    run_corpus(a)
    if not a.no_real and a.real_pages != -1:
        run_real(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
