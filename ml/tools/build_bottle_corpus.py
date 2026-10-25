"""
build_bottle_corpus.py — a corpus of DOSE PANELS ON CURVED GLOSSY BOTTLES.

    bash run.sh py ml/tools/build_bottle_corpus.py --pages 500
    bash run.sh py ml/tools/build_bottle_corpus.py --pages 3 --probe   # look at it

WHY THIS EXISTS: the shipped recogniser reads NOTHING on a real medicine bottle.
Measured, not assumed — a clean 1506x89 crop of a bold "Methylphenidate HCl" line
from a real photograph decodes to 'TB'; whole real pages give 'F'/'H' per band and
0 of 2 pair with their transcript. Five hypotheses were tested and killed first
(squash-to-320, native aspect, tiling, five preprocessings, and an un-warp SEARCH
scored by the shipped word LM): none of them read the text, so the cause is the
training distribution, not geometry, not preprocessing, not the decoder.

Two things are missing from every other corpus in this repo:
  * TEXT the world actually prints — `fetch_dose_labels.py` brings 15,501 panel
    lines from 1,239 real FDA labels (dosage_and_administration, active_ingredient,
    purpose, warnings...), instead of our renderer's idea of a dose panel.
  * GEOMETRY — a label wrapped round a bottle is compressed toward its edges, and
    our whole corpus is flat pages. `capture.cylinder_wrap` adds that, on top of the
    existing glare/lighting/noise/jpeg profiles, so curvature is IN the data rather
    than hoped for.

Everyone is reused, not re-implemented: `render_corpus.build` does layout, band
extraction, label derivation, splits and packing, and this file supplies only the
two things it cannot invent (the text, and the curl).

WHAT THIS DOES NOT PROVE: a synthetic bottle is still synthetic. Its numbers are a
PREDICTION — they say the model can learn curvature, not that a real bottle got
read. The real metric is the human-transcribed photographs in
data/real_bottles/truth/, and that is what a retrain has to move.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

import capture as C          # noqa: E402
import render_corpus as RC   # noqa: E402

PANELS = ROOT / "data" / "dose_panels.txt"
OUT = ROOT / "data" / "bottle_corpus"

# Curvature strengths, as fractions of a half-turn. A pill bottle's label is
# usually a partial wrap (0.2-0.4); the deep end exists so the model does not learn
# one curvature the way it learned one font.
CURLS = (0.15, 0.25, 0.35, 0.45, 0.55)
# The capture profiles a REAL phone photo of glossy packaging lands in. Reusing the
# existing profile parameters matters: they were re-tuned against measured
# photographs (retune_profiles.py), so this corpus inherits that calibration
# instead of inventing a second, uncalibrated idea of "glossy".
PROFILES = ["glossy_glare", "off_axis", "jpeg_social", "handheld_light",
            "handheld_heavy", "low_light", "worst_case"]


def curl_transform(img, seed):
    """The one geometry the flat renderer cannot express: the panel is CURVED."""
    rng = random.Random(seed)
    curl = rng.choice(CURLS)
    return C.cylinder_wrap(img, strength=curl), "bottle_curl%.2f" % curl


def wrap(line, width=38):
    """Break a leaflet paragraph into the short physical lines a label actually
    prints. openFDA returns sections as paragraphs (some 400+ chars) and drawing one
    as a single unbroken line teaches the model a line length no bottle has; it also
    puts the label outside the crop width the app can serve."""
    import textwrap
    return textwrap.wrap(line, width=width) or [line]


def panels_from(lines, per_page, seed=0):
    """Real panel lines -> pages of (page_id, lines), lines shuffled across pages.

    Shuffling is deliberate: the real files arrive in leaflet order, and keeping
    that order would teach the model that 'DOSAGE' is always the third line.
    """
    rng = random.Random(seed)
    wrapped = [w for l in lines for w in wrap(l)]
    pool = [l for l in wrapped if len(l) >= 4]
    rng.shuffle(pool)
    out = []
    for i in range(0, len(pool), per_page):
        chunk = pool[i:i + per_page]
        if len(chunk) < 3:
            continue
        out.append(("dose-%04d" % (i // per_page), chunk))
    return out


def probe(data, n=4):
    """Write a few crops out so a person can LOOK at what was built."""
    from PIL import Image
    out = Path("/tmp/bottle_corpus_probe")
    out.mkdir(exist_ok=True)
    idx = np.linspace(0, len(data["crops"]) - 1, n).astype(int)
    for k, i in enumerate(idx):
        crop = np.asarray(data["crops"][i], np.uint8)
        Image.fromarray(crop).resize((crop.shape[1] * 2, crop.shape[0] * 2),
                                     Image.LANCZOS).save(out / ("crop%02d.png" % k))
        print("   %s  %dx%d  label=%r" % (out / ("crop%02d.png" % k),
                                          crop.shape[1], crop.shape[0],
                                          data["labels"][i][:60]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=500)
    ap.add_argument("--lines-per-page", type=int, default=9)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--min-contrast", type=int, default=55,
                    help="p95-p5 pixel separation a crop must have to be readable")
    a = ap.parse_args()

    if not PANELS.exists():
        print("no real panel text at %s\nrun: bash run.sh py ml/tools/"
              "fetch_dose_labels.py --n 1200" % PANELS, file=sys.stderr)
        return 1
    lines = [l.rstrip("\n") for l in PANELS.read_text().splitlines() if l.strip()]
    pl = panels_from(lines, a.lines_per_page, seed=a.seed)[:a.pages]
    if not pl:
        print("no panels built from %d lines" % len(lines), file=sys.stderr)
        return 1
    print("%d real panel lines -> %d bottle pages of ~%d lines"
          % (len(lines), len(pl), a.lines_per_page))

    data = RC.build(n_pages=0, profiles=PROFILES, seed=a.seed, out_dir=a.out,
                    panels=pl, page_transform=curl_transform, verbose=True)

    # CONTRAST FLOOR. A crop whose ink and paper are the same value teaches nothing
    # and actively harms: it asks the model to emit a full string from a smear. The
    # worst_case profile (defocus 3.0, hand shadow, light 0.5) produced crops that a
    # person could not read at 2x either, so the floor is measured, not assumed:
    # p95-p5 of the pixels, the separation between paper and ink.
    keep, dropped = [], 0
    for i, c in enumerate(data["crops"]):
        arr = np.asarray(c, np.float32)
        if float(np.percentile(arr, 95) - np.percentile(arr, 5)) < a.min_contrast:
            dropped += 1
            continue
        keep.append(i)
    if dropped:
        print("dropped %d/%d crops below the contrast floor (%d): unreadable by a "
              "person, so unlabelled in practice" % (dropped, len(data["crops"]),
                                                     a.min_contrast))
    for k in ("crops", "labels", "doc", "profile", "split", "font"):
        if k in data:
            data[k] = np.array([data[k][i] for i in keep], dtype=object) if k != "crops" \
                else np.array([data[k][i] for i in keep], dtype=object)
    cache = RC.write_cache(data=data, out_path=Path(a.out) / "crops_train.npz")

    z = np.load(cache, allow_pickle=True)
    splits = {s: int((z["splits"] == s).sum()) for s in ("train", "val", "test")}
    variants = sorted(set(str(v) for v in z["variants"]))
    print("wrote %s  %s  %d crops  maxW %d" % (cache, splits, z["images"].shape[0],
                                               z["images"].shape[2]))
    print("curvatures present: " + ", ".join(variants))

    # Self-check: what makes this corpus correct is that a LABEL IS THE TEXT THAT
    # WAS DRAWN, and that the curl actually reached the pixels. Both are cheap to
    # assert and both have failed silently elsewhere in this project.
    assert z["images"].shape[0] == len(data["labels"]), "packed crop count mismatch"
    assert all(str(l).strip() == str(l) for l in z["labels"][:200])
    assert all(str(l) == str(l).upper() for l in z["labels"][:200]), "labels must be upper"
    assert len(variants) >= 3, "expected several curvatures, got %s" % variants
    widths = z["widths"][z["widths"] > 0]
    assert widths.min() >= 8, "a cropped line narrower than 8px is not text"
    if a.probe:
        print("probe crops:")
        probe(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
