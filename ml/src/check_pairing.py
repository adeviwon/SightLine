"""
Is the band->label pairing trustworthy, or systematically shifted?

The handheld build pairs detected band k with transcript line k. That is only
valid if the detector finds exactly the text lines, in order. The build reports
0.71 bands per transcript line, which means it is NOT finding them all -- so
the question is whether it is missing lines uniformly (pairing degrades but
stays roughly right) or missing them from the top (every label is shifted and
the training data is garbage).

This decides whether artifacts/ocr_handheld.npz is usable at all. Training on
mis-paired data is worse than not training on it.
"""
import sys
from collections import Counter

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/ubuntu/Projects/sightline/ml/src")
import build_handheld as BH   # noqa: E402
import realdata as R          # noqa: E402

pages = R.handheld_pages()
ratios = []
for (name, jp, lines) in pages:
    want = [l.strip() for l in lines if len(l.strip()) >= 3]
    if not want:
        continue
    a = np.asarray(Image.open(jp).convert("L"), np.uint8)
    boxes = BH.detect_lines(a)
    if not boxes:
        continue
    ratios.append((name, len(boxes), len(want), len(boxes) / max(len(want), 1)))

r = np.array([x[3] for x in ratios])
print(f"pages analysed: {len(ratios)}")
print(f"bands/line ratio: p10 {np.percentile(r,10):.2f}  p50 {np.percentile(r,50):.2f}"
      f"  p90 {np.percentile(r,90):.2f}")
close = sum(1 for x in r if 0.85 <= x <= 1.15)
print(f"pages with ratio in [0.85, 1.15]: {close}  ({close/len(r)*100:.0f}%)")
print(f"  -> crops from those pages would be ~{close/len(r)*2301:.0f}")
print()

# The decisive test: WHERE does the detector disagree with the transcript?
# Take pages where the counts match and look at the first band's height
# against the first line. If band 0 is the sleeve edge rather than the first
# text line, the first band will be much TALLER than a text line (it spans the
# dark background above the receipt) or its ink fraction will be tiny.
name, jp, lines = pages[0][0], pages[0][1], pages[0][2]
want = [l.strip() for l in lines if len(l.strip()) >= 3]
a = np.asarray(Image.open(jp).convert("L"), np.uint8)
boxes = BH.detect_lines(a)
print(f"page {name}: {len(boxes)} bands, {len(want)} transcript lines")
print(f"{'#':>3} {'band h':>7} {'ink%':>6} {'contrast':>9}  label")
for k, (y0, y1, x0, x1) in enumerate(boxes[:10]):
    crop = a[y0:y1, x0:x1]
    c = crop.astype(np.int16)
    lo, hi = int(c.min()), int(c.max())
    mid = (lo + hi) / 2.0
    ink = float((c < mid).mean()) * 100
    lab = want[k] if k < len(want) else "(no label)"
    flag = "" if BH.legible(crop) else "  <- REJECTED"
    print(f"{k:>3} {y1-y0:>7} {ink:>5.1f}% {hi-lo:>9}  {lab[:34]}{flag}")

print()
print("If band 0's label is not what band 0 actually shows, the pairing is")
print("shifted and the npz is unusable without an alignment step.")
