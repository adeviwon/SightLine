"""
offaxis_detect — the Python HALF of Lane AE.

THE QUESTION THIS LANE ANSWERS
------------------------------
The app's end-to-end accuracy is 35.0% word accuracy, but the `off_axis` profile
reads at 1.1% word accuracy / CER 0.882.  The cause is the DETECTOR, not the
model: on the app's own render, `legal/off_axis` gives 2 detected bands where the
same document gives 15 in every other profile.  `off_axis` is
`dict(perspective=0.045, rot=4.0)` — a perspective warp plus a 4 degree rotation,
which is exactly what smears a row-projection flat and merges neighbouring lines
into one band.

`ml/src/line_detect.py` already ships a `slope_aware` bander (shear the ink
projection flat before banding) but it was measured REGRESSED and not adopted —
on a corpus of `PROFILES[:4]` = studio_clean, handheld_light, handheld_heavy,
low_light, i.e. a corpus with NO rotated pages.  So the shear was only ever
tested where there is no rotation to correct.

This file does two jobs:

  1. Runs the REAL `line_detect.variant_bands` — shipped front end and the
     shipped `slope_aware` shear, CALLED not re-implemented — on the app's own
     bench pages: the three `off_axis` pages and the un-rotated profiles as a
     control, and prints bands-vs-truth line counts and band heights.
  2. Recovers EXACT ground-truth line boxes on the off_axis page and writes them
     to /tmp/offaxis_gt_boxes.json, so app/tools/offaxis_detect.js can hand the
     recogniser ground-truth crops instead of detected ones (the prize: is the
     detector the ONLY problem?).  The boxes are exact because
     capture.render_document + capture.apply_profile reproduce the bench PNG
     byte-for-byte (verified: np.abs(rendered - saved).max() == 0 for off_axis),
     and a coordinate-map warp through the SAME perspective+rotation tells every
     degraded pixel which clean row/column it came from.

Run:   ./.venv/bin/python ml/tests/test_offaxis_detect.py            (table + asserts)
       ./.venv/bin/python -m pytest ml/tests/test_offaxis_detect.py -q
"""

import json
import sys
from pathlib import Path

import random

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import capture as C        # noqa: E402
import line_detect as LD   # noqa: E402

SAMPLES = Path("/tmp/sightline_samples")
GT_BOXES = Path("/tmp/offaxis_gt_boxes.json")

# render_samples.py's PROFILES order -> the profile index is half the seed.
PROFILES = ["studio_clean", "handheld_light", "off_axis", "jpeg_social"]
CONTROL = ["studio_clean", "handheld_light", "jpeg_social"]
# The shear's original corpus had these four; the two that render_samples.py does
# not write are available as capture.py __main__'s own single-document dumps.
EXTRA = {"handheld_heavy": "/tmp/sample_handheld_heavy.png",
         "low_light": "/tmp/sample_low_light.png"}


def seed_for(dtype, profile):
    """render_samples.py: seed = 123 * 7919 + p_i * 101 + dt_i."""
    dt_i = [d for _, d in C.DOCS].index(dtype)
    return 123 * 7919 + PROFILES.index(profile) * 101 + dt_i


def rendered(dtype, profile):
    """(clean, degraded) for a bench page — reproduces the PNG exactly."""
    lines = next(l for l, d in C.DOCS if d == dtype)
    seed = seed_for(dtype, profile)
    clean = C.render_document(lines, seed=seed)
    deg = C.apply_profile(clean, seed=seed, **C.CAPTURE_PROFILES[profile])
    return lines, clean, deg


def _perspective_nn(img, strength=0.045, seed=None):
    """`capture.perspective()` sampled NEAREST instead of BICUBIC.

    THIS HELPER WAS MISSING, so the test that calls it had never once run -- it raised
    NameError on a function that existed nowhere in the repo, and a workspace full of
    passing tests hid it because the failure lives in a test rather than in shipped code.

    Why not just call capture.perspective: this test encodes a pixel's (x,y) COORDINATE
    into its VALUE (low byte in one plane, high byte in another), warps, then decodes the
    coordinates back to learn exactly where each drawn line landed. capture.perspective
    hardcodes BICUBIC and interpolation BLENDS those byte values, so decoded coordinates
    smear and every box expands. Same corner jitter and same rng draw order as
    capture.perspective, so the geometry is identical; only the sampling differs.
    """
    rng = random.Random(seed)
    w, h = img.size
    s = strength
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(rng.uniform(0, s) * w, rng.uniform(0, s) * h),
           (w - rng.uniform(0, s) * w, rng.uniform(0, s) * h),
           (w - rng.uniform(0, s) * w, h - rng.uniform(0, s) * h),
           (rng.uniform(0, s) * w, h - rng.uniform(0, s) * h)]
    return img.transform((w, h), Image.PERSPECTIVE, C._perspective_coeffs(dst, src),
                         Image.NEAREST, fillcolor=(205, 205, 205))


def _decode(img16_lo, img16_hi):
    """Recombine two warped 8-bit images encoding (row % 256, row // 256)."""
    return img16_lo.astype(np.int32) + 256 * img16_hi.astype(np.int32)


def gt_boxes_on_degraded(dtype):
    """Exact GT line boxes ON THE off_axis PAGE, via the generator's own warp.

    A coordinate image carries the CLEAN row/column of every pixel through the
    profile's geometric ops (perspective then rotate — the same seed, so the same
    homography).  A degraded pixel belongs to line i when its clean row falls in
    line i's rendered band and its clean column in that line's ink columns.
    """
    lines, clean, deg = rendered(dtype, "off_axis")
    seed = seed_for(dtype, "off_axis")
    cg = np.asarray(clean.convert("L"), dtype=np.uint8)
    H, W = cg.shape

    truth = LD.page_truth(cg, lines, 1.0)          # {line, box:(y0,y1), text}

    rows = np.arange(H, dtype=np.int32)[:, None] * np.ones((1, W), np.int32)
    cols = np.ones((H, 1), np.int32) * np.arange(W, dtype=np.int32)[None, :]

    def warp(plane):
        """NEAREST perspective + rotation — capture.perspective hardcodes
        BICUBIC, which BLENDS the coordinate values and expands every box."""
        lo = Image.fromarray((plane % 256).astype(np.uint8), "L")
        hi = Image.fromarray((plane // 256).astype(np.uint8), "L")
        out = []
        for im in (lo, hi):
            im = _perspective_nn(im.convert("RGB"), 0.045, seed)
            im = im.rotate(4.0, fillcolor=(205, 205, 205), expand=False,
                           resample=0)                   # NEAREST: exact values
            out.append(np.asarray(im.convert("L"), dtype=np.uint8))
        return _decode(out[0], out[1])

    row_src, col_src = warp(rows), warp(cols)

    boxes = []
    for t in truth:
        y0, y1 = t["box"]
        band = cg[y0:y1] < 170
        cx = np.where(band.sum(axis=0) > 0)[0]
        if not len(cx):
            continue
        m = (row_src >= y0) & (row_src < y1) & (col_src >= int(cx[0])) & (col_src <= int(cx[-1]))
        ys, xs = np.where(m)
        if not len(ys):
            continue
        boxes.append({"line": t["line"], "text": t["text"],
                      "x0": int(xs.min()), "y0": int(ys.min()),
                      "x1": int(xs.max()) + 1, "y1": int(ys.max()) + 1})
    return boxes


# ── the measured table: shipped bands vs slope_aware bands vs truth ──────────

def _heights(bands):
    return [b[1] - b[0] for b in bands]


def sweep():
    """[(doc, profile, n_truth, base_bands, slope_bands)] over bench + controls."""
    out = []
    for dtype in [d for _, d in C.DOCS]:
        for profile in PROFILES:
            lines, clean, deg = rendered(dtype, profile)
            n_truth = len(lines)
            gray = np.asarray(deg.convert("L"), dtype=np.uint8)
            base = LD.variant_bands(gray, set())
            slope = LD.variant_bands(gray, {"slope_aware"})
            out.append((dtype, profile, n_truth, base, slope))
    return out


def print_table():
    print("\n=== line_detect.py (REAL, called not re-implemented): bands vs the "
          "drawn lines ===")
    print(f"{'doc/profile':<28}{'truth':>6}{'shipped':>9}{'slope_aware':>13}"
          f"   shipped bh med/max   slope bh med/max")
    print("-" * 96)
    for doc, profile, n, base, slope in sweep():
        bh, sh = _heights(base), _heights(slope)
        mk = lambda a: f"{int(np.median(a))}/{max(a)}" if a else "-/-"
        flag = "  <-- LOSES LINES" if len(base) < n else ""
        print(f"{doc + '/' + profile:<28}{n:>6}{len(base):>9}{len(slope):>13}"
              f"   {mk(bh):>16}   {mk(sh):>16}{flag}")
    print("\ncontrol profiles = the un-rotated bench profiles; the shear's original\n"
          "corpus (PROFILES[:4]) held only these, so it was never tested on a tilt.")


def write_boxes():
    out = {}
    for dtype in [d for _, d in C.DOCS]:
        out[dtype] = gt_boxes_on_degraded(dtype)
    GT_BOXES.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {GT_BOXES}  " +
          "  ".join(f"{k}:{len(v)}" for k, v in out.items()))
    return out


# ── the checks ───────────────────────────────────────────────────────────────

def _mask_of(bars, w, h):
    a = np.zeros((h, w), np.uint8)
    for y0, y1, x0, x1 in bars:
        a[y0:y1, x0:x1] = 1
    return a


def test_slope_of_recovers_a_known_tilt():
    """The shear's slope estimator must return the tilt it was given."""
    for slope in (0.05, 0.07, -0.07):
        w, h = 400, 300
        m = np.zeros((h, w), np.uint8)
        for x in range(40, 360):
            y = int(150 + slope * (x - 200))
            m[y:y + 12, x] = 1
        got = LD._slope_of(m)
        assert abs(got - slope) < 0.02, f"slope {slope} -> {got}"


def _runs(proj, thr):
    """[(y0,y1)] of runs where proj exceeds thr."""
    on = proj > thr
    out, s, i = [], None, 0
    while i <= len(on):
        v = i < len(on) and on[i]
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i))
            s = None
        i += 1
    return out


def test_shear_separates_lines_the_row_projection_merges():
    """Two tilted lines overlap in a flat projection and separate after shear."""
    slope, w, h = 0.15, 400, 260
    m = np.zeros((h, w), np.uint8)
    for x in range(40, 360):
        dy = int(slope * (x - 200))
        for y0 in (90, 104):
            m[y0 + dy:y0 + dy + 10, x] = 1
    thr = 4
    plain = _runs(m.sum(axis=1), thr)
    sheared = _runs(LD._shear_proj(m, LD._slope_of(m)), thr)
    print(f"    synthetic tilt {slope}: plain {len(plain)} bands, "
          f"slope_aware {len(sheared)} bands")
    assert len(plain) == 1, f"the two lines did not merge in the flat projection: {plain}"
    assert len(sheared) == 2, f"shear did not separate the lines: {sheared}"


def test_offaxis_gt_boxes_are_one_per_drawn_line():
    """Every drawn line gets exactly one GT box on the warped page."""
    boxes = write_boxes()
    for dtype, bs in boxes.items():
        n = len(next(l for l, d in C.DOCS if d == dtype))
        assert len(bs) == n, f"{dtype}: {len(bs)} GT boxes for {n} drawn lines"


if __name__ == "__main__":
    print_table()
    write_boxes()
    print("\n=== checks ===")
    for fn in (test_slope_of_recovers_a_known_tilt,
               test_shear_separates_lines_the_row_projection_merges):
        fn()
        print(f"  OK  {fn.__name__}")
    test_offaxis_gt_boxes_are_one_per_drawn_line()
    print("  OK  test_offaxis_gt_boxes_are_one_per_drawn_line")
    print("\nOFFAXIS-DETECT PY: OK")
