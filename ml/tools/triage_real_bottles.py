"""
triage_real_bottles.py — which of the fetched REAL photographs actually contain a
readable dosage panel, and can our own detector even see it?

    bash run.sh py ml/tools/triage_real_bottles.py --out data/real_bottles/triage.csv

WHY TRIAGE BEFORE TRANSCRIBING
------------------------------
Wikimedia's medicine categories are mostly antique bottles, museum group shots and
pharmacy shelves — real photographs, but not a doses panel you could transcribe.
Transcribing is the expensive step (a person reads the pixels), so the filter runs
first, and it reports TWO independent things because they fail differently:

* `bands`  — lines our page detector finds. Zero bands means the product could not
  read this photo at all, whatever the model inside it does.
* `ink`, `sharpness` — how much of the frame is text-ish and how crisp it is. A
  glossy curved bottle gives a plausible band count off a highlight or a wall of
  shrink-wrap markings, and those are exactly the photos whose transcript would be
  invented rather than read.

A file is a transcription CANDIDATE when the detector finds at least
`--min-bands` lines; the report also prints the rows ordered by sharpness so a
human can pick a subset that is genuinely legible.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eval_ocr as EO          # noqa: E402
import realdata as R           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "real_bottles" / "raw"
OUT = ROOT / "data" / "real_bottles" / "triage.csv"


def stats(path):
    from PIL import Image
    img = Image.open(path).convert("L")
    w0, h0 = img.size
    # the detector works at a page scale; cap the long side so a 6000px photo does
    # not cost seconds per file
    if max(w0, h0) > 1600:
        s = 1600.0 / max(w0, h0)
        img = img.resize((max(1, int(w0 * s)), max(1, int(h0 * s))))
    a = np.asarray(img, np.uint8)
    proj = np.asarray(EO.ink_mask(a), float).sum(1)
    bands = EO.line_bands(proj, min_h=max(4, int(0.008 * a.shape[0])))
    lap = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], float)
    from numpy.lib.stride_tricks import sliding_window_view
    if min(a.shape) > 8:
        win = sliding_window_view(a.astype(float), (3, 3))
        sharp = float(np.abs((win * lap).sum(-1)).var())
    else:
        sharp = 0.0
    ink = float(np.asarray(EO.ink_mask(a), float).mean())
    return {"w": w0, "h": h0, "bands": len(bands), "ink": round(ink, 4),
            "sharpness": round(sharp, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-bands", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted(p for p in RAW.glob("*") if p.suffix.lower() in
                   (".jpg", ".jpeg", ".png"))
    if a.limit:
        files = files[:a.limit]
    if not files:
        print("no photographs in %s — run ml/tools/fetch_real_bottles.py" % RAW)
        return 1

    rows = []
    for i, p in enumerate(files):
        try:
            s = stats(p)
        except Exception as e:                                    # pragma: no cover
            print("  skip %s (%s)" % (p.name[:40], e), file=sys.stderr)
            continue
        rows.append(dict(file=p.name, **s))
        if (i + 1) % 25 == 0:
            print("  triaged %d/%d" % (i + 1, len(files)), flush=True)

    rows.sort(key=lambda r: (-r["bands"], -r["sharpness"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(a.out).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "w", "h", "bands", "ink", "sharpness"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    cand = [r for r in rows if r["bands"] >= a.min_bands]
    print("%d photographs triaged, %d with >=%d detected lines (transcription candidates)"
          % (len(rows), len(cand), a.min_bands))
    print("bands distribution: " + ", ".join(
        "%d bands:%d" % (b, sum(1 for r in rows if r["bands"] == b))
        for b in range(0, 6)) + (", 6+:%d" % sum(1 for r in rows if r["bands"] >= 6)
                                 if any(r["bands"] >= 6 for r in rows) else ""))
    print("\ntop candidates by sharpness (the ones a person can actually read):")
    for r in sorted(cand, key=lambda r: -r["sharpness"])[:15]:
        print("  %-42s %4dx%-5d bands %2d ink %.3f sharp %8.1f"
              % (r["file"][:42], r["w"], r["h"], r["bands"], r["ink"], r["sharpness"]))
    print("\nwrote %s" % a.out)
    print("NOTE: a candidate is not a label. The transcript is read off the pixels by a "
          "person and is the ONLY ground truth this set ever has.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
