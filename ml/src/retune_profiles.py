"""
retune_profiles.py — re-tune the synthetic camera profiles against the REAL
hand-photographed statistics, so the simulator stops out-degrading reality.

THE PROBLEM (measured, and of exactly the shape that has paid twice before)
--------------------------------------------------------------------------
Training on rendered documents instead of only SROIE lifted field recovery
27/300 -> 108/300. The serve width being below the training clamp cost crop
exact-match 44.7% -> 0.0%. Both were TRAIN/SERVE DISTRIBUTION MISMATCH.

There is a third one, and it is in the SIMULATOR. Measured against the 193
hand-photographed Express-Expense receipts, the camera-like capture profiles in
`capture.CAPTURE_PROFILES` are not merely different from reality — they are far
harsher than any real photo:

    profile            ink_mean  contrast  sharpness (Laplacian var, trainer geometry)
    handheld_light        0.62       49        32    <- 30x blurrier than real
    handheld_heavy        0.63       36        19    <- 52x blurrier than real
    low_light             0.43       38        58    <- 17x blurrier than real

and those are exactly the profiles the recogniser is worst on (low_light CER
0.72, handheld_heavy 0.51, against studio_clean 0.09). The model is spending its
capacity learning to undo a degradation it will never see. The ink-density tell
is decisive: 0.62 ink on a dark-on-light document is sensor noise and shadow
being COUNTED as ink, three times the real 0.21.

WHAT THIS MODULE DOES
---------------------
1. Re-derives the real-crop statistics with the SAME three functions the earlier
   lane used (`real_degradation_cache.crop_stats` / `stats_of`) at BOTH the
   trainer's geometry (`build_dataset.crop_to_array`) and native scale.
2. Measures each synthetic profile the same way, before and after a re-tune.
3. Carries the re-tuned physical parameters (`TUNED`): blur radius/sigma, motion
   length, sensor ISO and lighting-gradient strength. NOTHING post-hoc rescales
   a statistic — matching a statistic by transforming the output would be
   fitting the metric, not the phenomenon.
4. Keeps the profiles ORDERED by severity and deterministic in the existing CRC32
   seed convention (`render_corpus.page_seed`, `seedutil.name_hash`).

USE
---
    bash run.sh py ml/src/retune_profiles.py --compare
    bash run.sh py ml/src/retune_profiles.py --fingerprint   # process-stable seed

The `--compare` table is the deliverable: the real row is the target, and every
camera profile must sit within a small multiple of it instead of 17-52x away.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as B           # noqa: E402  crop_to_array: ONE crop geometry
import capture                      # noqa: E402  apply_profile + CAPTURE_PROFILES
import real_degradation_cache as RDC  # noqa: E402  crop_stats / stats_of: SAME stats
import realdata as R                # noqa: E402  handheld_pages
import render_corpus as RC          # noqa: E402  page_crops / page_seed / font_for_page

ROOT = Path(__file__).resolve().parents[2]

# The profiles this lane is allowed to touch: the three that pretend to be a
# camera. `studio_clean` is a clean render (the sharpness ceiling, deliberately
# sharper than any photo) and the other six carry their own degradation families
# this lane did not measure, so they are left exactly as they were.
CAMERA = ("handheld_light", "handheld_heavy", "low_light")

# Table order for the report. studio_clean is shown but unchanged.
SHOWN = ("studio_clean", "handheld_light", "handheld_heavy", "low_light")

# ── BEFORE: the shipped capture profile table, verbatim ──────────────────────
BEFORE = {k: dict(v) for k, v in capture.CAPTURE_PROFILES.items()}

# ── AFTER: the three camera-like profiles, re-tuned ──────────────────────────
#
# What changed, and why (all three are PHYSICAL knobs, not statistic rescalers):
#
#   defocus   the circular bokeh-disc radius in PAGE pixels. The old 1.6-2.4 is
#             a 7x7 disc, i.e. a severe defocus on 19 px glyphs; 0.5-1.0 is a
#             3x3-5x5 disc, which is what a phone at arm's length actually does.
#   motion    the directional hand-shake kernel length in px. The old 7-11 px is
#             a ~1/10-page smear; 2-3 px matches real hand tremor at 1/50 s.
#   iso       the sensor-noise ISO fed to `sensor_noise` (Poisson-Gaussian). The
#             old 400-6400 with noise_iso is fine in magnitude, but at 6400 the
#             read noise + the strong lighting gradient flatten the histogram so
#             Otsu splits mid-grey and COUNT noise as ink (the 0.62 tell).
#   light     the lighting-gradient strength (multiplicative page falloff).
#             0.55 is a 55% falloff, far past a real room; 0.08-0.55 is bounded
#             so paper stays paper.
#
# Severity is preserved and now MONOTONE: light <= heavy <= low_light in ink,
# light >= heavy >= low_light in contrast and in sharpness. See `severity_ok`.
TUNED = {
    "handheld_light": dict(defocus=0.5, iso=400, light=0.08),
    "handheld_heavy": dict(defocus=0.7, motion=2, motion_angle=35.0,
                           iso=1600, light=0.25),
    "low_light":      dict(defocus=1.0, motion=3, motion_angle=35.0,
                           iso=3200, light=0.55, noise_iso=6400),
}

# The full table a rebuilt corpus should use: shipped values everywhere except
# the three camera-like profiles.
AFTER = {**BEFORE, **TUNED}

# Re-derived from the 193 hand-photographed receipts with `--compare` (n=394
# line crops). trainer = through build_dataset.crop_to_array; native = the raw
# detector box. NOTE: the earlier lane reported trainer sharp 1004.5 / native
# 1469 on n=290; ink and contrast agree, sharpness does not (different sample
# and detector limit). Ours are used, as the task requires.
REAL_STATS = {
    "native":  {"ink": 0.2044, "contrast": 89.6, "sharp": 2538.1, "n": 394},
    "trainer": {"ink": 0.2269, "contrast": 78.3, "sharp": 891.1,  "n": 394},
}

# Measured with --compare on 6 pages/doc-type per profile (n=282 crops each),
# embedded so the ordering and bracketing tests are deterministic and need no
# rendering or detector run. Regenerate with `--compare --json`.
BEFORE_STATS = {
    "native": {
        "studio_clean":   {"ink": 0.2376, "contrast": 172.9, "sharp": 26893.7},
        "handheld_light": {"ink": 0.6273, "contrast": 43.7,  "sharp": 226.3},
        "handheld_heavy": {"ink": 0.6434, "contrast": 33.1,  "sharp": 232.8},
        "low_light":      {"ink": 0.4576, "contrast": 35.5,  "sharp": 387.6},
    },
    "trainer": {
        "studio_clean":   {"ink": 0.2710, "contrast": 137.8, "sharp": 5249.5},
        "handheld_light": {"ink": 0.6293, "contrast": 43.1,  "sharp": 21.2},
        "handheld_heavy": {"ink": 0.6507, "contrast": 32.9,  "sharp": 13.3},
        "low_light":      {"ink": 0.4558, "contrast": 35.1,  "sharp": 53.1},
    },
}
AFTER_STATS = {
    "native": {
        "studio_clean":   {"ink": 0.2376, "contrast": 172.9, "sharp": 26893.7},
        "handheld_light": {"ink": 0.3181, "contrast": 107.1, "sharp": 5176.7},
        "handheld_heavy": {"ink": 0.3234, "contrast": 86.2,  "sharp": 2469.8},
        "low_light":      {"ink": 0.4092, "contrast": 44.9,  "sharp": 470.1},
    },
    "trainer": {
        "studio_clean":   {"ink": 0.2710, "contrast": 137.8, "sharp": 5249.5},
        "handheld_light": {"ink": 0.3325, "contrast": 96.5,  "sharp": 1469.7},
        "handheld_heavy": {"ink": 0.3553, "contrast": 78.6,  "sharp": 740.7},
        "low_light":      {"ink": 0.4170, "contrast": 43.1,  "sharp": 205.2},
    },
}

# Upper bound on how far a camera profile may sit from the real row on any
# metric, in either direction, before its statistics are "wrong" rather than
# "different". Chosen to admit the structural limit (the renderer + the
# crop_to_array resize put even a CLEAN render at ~1.2x the real ink density)
# while still failing the old 17-52x profiles by a mile.
BRACKET_FACTOR = 8.0


def embedded_table():
    """The measured before/after table in the predicates' nested shape."""
    t = {"real": {"before": {"native": REAL_STATS["native"],
                             "trainer": REAL_STATS["trainer"]}}}
    for k in SHOWN:
        t[k] = {"before": {g: BEFORE_STATS[g][k] for g in ("native", "trainer")},
                "after":  {g: AFTER_STATS[g][k] for g in ("native", "trainer")}}
    return t


# ── the statistic: reuse, never reimplement ──────────────────────────────────

def _t(s):
    """stats_of's dict -> the (ink, contrast, sharp, n) triple the report uses."""
    return {"ink": s["ink_mean"], "contrast": s["contrast_mean"],
            "sharp": s["sharp_mean"], "n": int(s.get("n", 0))}


def geometry_stats(native_crops):
    """(native, trainer) aggregate stats for a list of NATIVE uint8 crops.

    `native` is the detector/line box at its own resolution; `trainer` is the
    same crop through `build_dataset.crop_to_array` — the geometry the model
    actually sees. Both use `real_degradation_cache.crop_stats`, so this is
    apples-to-apples with the earlier lane's table.
    """
    if not native_crops:
        return _t(RDC.stats_of([])), _t(RDC.stats_of([]))
    trn = [B.crop_to_array(Image.fromarray(np.asarray(c, np.uint8)))
           for c in native_crops]
    return _t(RDC.stats_of(native_crops)), _t(RDC.stats_of(trn))


def synthetic_native(params_by_profile, pages_per_doc=6, seed=0,
                     doc_types=None):
    """{profile: [native line crop, ...]} from bounded, seeded renders.

    Every profile sees the SAME clean page, so a before/after difference is the
    parameters and nothing else. Seeding is `render_corpus.page_seed` (zlib.crc32
    of the page id) and `seedutil`-style name hashing — never `hash()`, whose
    per-process salt once produced three different corpora from one seed.
    """
    import capture as C
    out = {k: [] for k in params_by_profile}
    doc_types = doc_types or [d for _, d in C.DOCS]
    page_ids = ["%s-%04d" % (dt, i) for i in range(pages_per_doc)
                for dt in doc_types]
    for pid in page_ids:
        dtype = pid.rsplit("-", 1)[0]
        lines = next(l for l, d in C.DOCS if d == dtype)
        clean = C.render_document(lines, width=900, seed=RC.page_seed(pid),
                                  scale=1.0, font=RC.font_for_page(pid))
        # One salt for every profile on this page: the clean input is shared.
        salt = RC.page_seed(pid, 1)
        for prof, params in params_by_profile.items():
            deg = C.apply_profile(clean, seed=salt, **params)
            gray = np.asarray(deg.convert("L"), np.uint8)
            for crop, label in RC.page_crops(gray, lines, 1.0):
                out[prof].append(crop)
    return out


def real_native(limit=None, progress=False):
    """Native tight line crops from the hand-photographed receipts.

    Re-derived here, not read from the cached corpus, because the cache is
    trainer geometry only. Detection is `real_degradation_cache.detect_boxes`
    (the shipped detector, no model), labelling is count-match or DROP.
    """
    crops = []
    pages = R.handheld_pages()
    if limit:
        pages = pages[:limit]
    for i, (name, jp, lines) in enumerate(pages):
        gray = np.asarray(Image.open(jp).convert("L"), np.uint8)
        boxes = RDC.detect_boxes(gray)
        pairs, _ = RDC.align_labels(boxes, lines)
        if pairs is None:
            continue
        for (y0, y1, x0, x1), line in pairs:
            if not RC.train_label(line):
                continue
            crops.append(gray[y0:y1, x0:x1])
        if progress and (i + 1) % 50 == 0:
            print("  ... %d/%d pages" % (i + 1, len(pages)), flush=True)
    return crops


# ── bracketing + ordering, as pure predicates (the tests use these) ──────────

def bracket_report(table, geo="trainer", when="after", factor=BRACKET_FACTOR):
    """Per camera profile: the worst real-relative deviation on any metric.

    `table[profile][when][geo] = {"ink", "contrast", "sharp", ...}`. Returns
    {profile: {"worst_ratio", "ok", "detail"}} where a ratio > 1 means the
    profile is N x away from real on that metric.
    """
    real = table["real"]["before"][geo]
    out = {}
    for prof in CAMERA:
        s = table[prof][when][geo]
        ratios = {}
        for m in ("ink", "contrast", "sharp"):
            r0 = real[m]
            r = s[m] / r0 if r0 > 1e-9 else float("inf")
            ratios[m] = r if r >= 1 else 1.0 / max(r, 1e-9)
        worst = max(ratios.values())
        out[prof] = {"worst_ratio": worst, "ok": worst <= factor,
                     "detail": {m: round(ratios[m], 2) for m in ratios}}
    return out


def severity_ok(table, geo="trainer", when="after"):
    """True when ink rises and contrast/sharpness fall, light -> heavy -> low.

    The ladder the re-tune must preserve, and now makes fully monotone.
    `studio_clean` is excluded: it is the clean ceiling, not a severity rung.
    """
    order = ["handheld_light", "handheld_heavy", "low_light"]
    s = [table[p][when][geo] for p in order]
    ink_up = all(s[i]["ink"] <= s[i + 1]["ink"] for i in range(len(s) - 1))
    con_down = all(s[i]["contrast"] >= s[i + 1]["contrast"]
                   for i in range(len(s) - 1))
    shp_down = all(s[i]["sharp"] >= s[i + 1]["sharp"]
                   for i in range(len(s) - 1))
    return ink_up and con_down and shp_down


# ── report ───────────────────────────────────────────────────────────────────

def print_table(table, factor=BRACKET_FACTOR):
    """The before/after table, real row as the target, both geometries."""
    print("\n" + "=" * 96)
    print("SYNTHETIC CAMERA PROFILES vs REAL HAND-PHOTOGRAPHED RECEIPTS")
    print("ink = Otsu ink fraction (lower is paper-like); contrast = mean ink/paper")
    print("separation (higher is cleaner); sharp = Laplacian variance (higher is sharper)")
    print("real = %d hand-photographed receipts, re-derived here with the SAME functions"
          % table["real"]["before"]["trainer"]["n"])
    print("=" * 96)
    for geo in ("native", "trainer"):
        print("\n  --- %s geometry ---  (trainer = through build_dataset.crop_to_array)"
              % geo)
        print("  %-16s %10s %10s %11s %12s %10s %11s"
              % ("profile", "B ink", "B contrast", "B sharp",
                 "A ink", "A contrast", "A sharp"))
        for k in ("real",) + SHOWN:
            b = table.get(k, {}).get("before", {}).get(geo)
            a = table.get(k, {}).get("after", {}).get(geo)
            tag = k
            if k == "real":
                print("  %-16s %10.4f %10.1f %11.1f %12s %10s %11s"
                      % (tag, b["ink"], b["contrast"], b["sharp"], "-", "-", "-"))
            elif k == "studio_clean":
                print("  %-16s %10.4f %10.1f %11.1f %12.4f %10.1f %11.1f   (unchanged)"
                      % (tag, b["ink"], b["contrast"], b["sharp"],
                         a["ink"], a["contrast"], a["sharp"]))
            else:
                print("  %-16s %10.4f %10.1f %11.1f %12.4f %10.1f %11.1f"
                      % (tag, b["ink"], b["contrast"], b["sharp"],
                         a["ink"], a["contrast"], a["sharp"]))

    real_t = table["real"]["before"]["trainer"]
    print("\n  real-relative deviation (trainer geometry, worst metric; 1.0 = identical)")
    print("  %-16s %12s %12s %12s %10s" % ("profile", "before(x)", "after(x)",
                                           "ink  a/b", "brackets"))
    rep_b = bracket_report(table, "trainer", when="before", factor=1e9)
    rep_a = bracket_report(table, "trainer", when="after", factor=factor)
    for k in CAMERA:
        print("  %-16s %12.1f %12.1f %12.2f %10s"
              % (k, rep_b[k]["worst_ratio"], rep_a[k]["worst_ratio"],
                 table[k]["after"]["trainer"]["ink"] / max(real_t["ink"], 1e-9),
                 "yes" if rep_a[k]["ok"] else "NO"))
    print("\n  severity order (light <= heavy <= low in ink, >= in contrast/sharp):")
    for geo in ("native", "trainer"):
        print("    %-8s %s" % (geo, "OK" if severity_ok(table, geo)
                               else "BROKEN"))
    print("  bracket factor: a camera profile is 'bracketing reality' when its worst\n"
          "  real-relative deviation is <= %.0fx on every metric." % factor)


# ── the CLI ──────────────────────────────────────────────────────────────────

def fingerprint(pages_per_doc=1):
    """A hash of the generated sample. Identical across processes iff seeding is.

    This is the regression guard for the `hash()`-seeded-corpus bug: run it twice
    with different PYTHONHASHSEED and the two digests must match.
    """
    prof = {"handheld_light": TUNED["handheld_light"]}
    crops = synthetic_native(prof, pages_per_doc=pages_per_doc)
    h = hashlib.sha256()
    for c in crops["handheld_light"]:
        h.update(np.ascontiguousarray(c, np.uint8).tobytes())
        h.update(b"|")
    return {"n": len(crops["handheld_light"]), "sha256": h.hexdigest(),
            "seeder": "render_corpus.page_seed (zlib.crc32)"}


def compare(pages_per_doc=6, real_limit=None, as_json=""):
    table = {"real": {}, "studio_clean": {}, "handheld_light": {},
             "handheld_heavy": {}, "low_light": {}}

    print("measuring real hand-photographed receipts ...", flush=True)
    rn, rt = geometry_stats(real_native(limit=real_limit))
    table["real"]["before"] = {"native": rn, "trainer": rt}

    print("rendering the before/after synthetic sample "
          "(%d page(s)/doc-type per profile) ..." % pages_per_doc, flush=True)
    before = synthetic_native({k: BEFORE[k] for k in SHOWN},
                              pages_per_doc=pages_per_doc)
    after = synthetic_native({k: AFTER[k] for k in SHOWN},
                             pages_per_doc=pages_per_doc)
    for k in SHOWN:
        bn, bt = geometry_stats([np.asarray(c, np.uint8) for c in before[k]])
        an, at = geometry_stats([np.asarray(c, np.uint8) for c in after[k]])
        table[k]["before"] = {"native": bn, "trainer": bt}
        table[k]["after"] = {"native": an, "trainer": at}

    print_table(table)

    if as_json:
        Path(as_json).write_text(json.dumps(_slim(table), indent=2))
        print("\nwrote %s" % as_json)
    return table


def _slim(table):
    out = {}
    for k, whens in table.items():
        out[k] = {w: {g: {m: round(float(s[m]), 4)
                          for m in ("ink", "contrast", "sharp", "n")}
                      for g, s in geos.items()}
                  for w, geos in whens.items()}
    return out


RETRAIN_COMMANDS = [
    "# 0. back up the checkpoint training will overwrite",
    "cp models/ocr/crnn.pt models/ocr/crnn_before_retune.pt",
    "# 1. rebuild the rendered corpus with the re-tuned camera profiles",
    "bash run.sh render-corpus --pages 500 --seed 0 "
    "--profiles studio_clean,handheld_light,handheld_heavy,low_light",
    "#    (this writes BOTH data/render_corpus/crops.npz and crops_train.npz;",
    "#     'bash run.sh render-corpus --cache-only' re-packs without re-rendering)",
    "# 2. merge the SROIE cache + the rendered cache + the real-photo cache and train",
    "bash run.sh py ml/src/train_ocr.py "
    "--data artifacts/ocr_train.npz,data/render_corpus/crops_train.npz,"
    "data/real_corpus/crops_train.npz --epochs 30 --threads 4",
    "# 3. re-measure against the before numbers, in this order",
    "bash run.sh eval-render --split test",
    "bash run.sh eval-ocr",
    "bash run.sh py ml/src/eval_field_truth.py --split test --limit-pages 120",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--compare", action="store_true",
                    help="the before/after table, real row as the target")
    ap.add_argument("--fingerprint", action="store_true",
                    help="print a hash of a fixed sample (process-stable seed)")
    ap.add_argument("--pages", type=int, default=6,
                    help="pages per doc-type per profile in the sample")
    ap.add_argument("--real-limit", type=int, default=0,
                    help="cap the hand-photographed pages (0 = all)")
    ap.add_argument("--json", default="", help="also write the table here")
    a = ap.parse_args(argv)

    if a.fingerprint:
        print(json.dumps(fingerprint(), indent=2))
        return 0
    if a.compare:
        compare(pages_per_doc=a.pages, real_limit=(a.real_limit or None),
                as_json=a.json)
        print("\n" + "-" * 96)
        print("RE-TUNED PARAMETERS (capture.CAPTURE_PROFILES -> this lane's TUNED):")
        for k in CAMERA:
            print("  %-16s %s" % (k, TUNED[k]))
        print("\nORDERED COMMANDS FOR THE NEXT RETRAIN (do NOT run the training here):")
        for c in RETRAIN_COMMANDS:
            print("  " + c)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
