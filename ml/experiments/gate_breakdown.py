"""Why do 2,480 real-photo band/line pairs produce only 76 training crops?

The pipeline works. The question is which gate eats the yield, because:
  * a gate that is a CONFIG VALUE (e.g. the 47-char label cap) is nearly free to
    raise -- and raising it needs no new data collection;
  * a gate that is protecting us from real label noise (illegible bands) must
    not be relaxed, because training on noise teaches the model to read noise.
Prints the breakdown, and separates those two cases explicitly.
"""
import sys
import collections

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/ubuntu/Projects/sightline/ml/src")
import build_handheld as BH   # noqa: E402
import eval_ocr as E          # noqa: E402
import realdata as R          # noqa: E402

MAX_LABEL = 47


def main():
    pages = R.handheld_pages()
    counts = collections.Counter()
    kept = 0
    dens = []
    long_labels = []

    for name, jp, lines in pages:
        want = [l.strip() for l in lines if len(l.strip()) >= 3]
        if len(want) < 2:
            counts["page_no_label"] += 1
            continue
        gray = np.asarray(Image.open(jp).convert("L"), np.uint8)
        # detect_lines returns (y0,y1,x0,x1) -- the builder's own contract.
        # bands_in_document returns (y0,y1) only, so using it here would need
        # the x-extent recomputed. The builder's detection is what the training
        # cache actually came from, so measure against THAT, not the eval path.
        bands = BH.detect_lines(gray)
        if len(bands) < 2:
            counts["page_no_bands"] += 1
            continue
        counts["pages"] += 1
        n = min(len(bands), len(want))
        counts["pairs_available"] += n
        if abs(len(bands) - len(want)) > 0:
            counts["page_count_mismatch"] += 1
        for k in range(n):
            y0, y1, x0, x1 = bands[k]
            lab = R.norm_label(want[k])
            if len(lab) < 2:
                counts["label_too_short"] += 1
                continue
            if len(lab) > MAX_LABEL:
                counts["label_too_long"] += 1
                long_labels.append((name, lab))
                continue
            raw = gray[y0:y1, x0:x1]
            if not BH.legible(raw):
                counts["illegible_band"] += 1
                continue
            arr_i, sw = BH.prepare(raw)
            # The real density gate: timesteps = width/8, and CTC needs >=1.5
            # timesteps per glyph, i.e. width/chars >= 12. This is what actually
            # kills the yield -- my first version checked BH.fits() instead and
            # reported 2,656 keepable, which was the wrong gate.
            if sw / max(len(lab), 1) < 12.0:
                counts["too_dense_to_read"] += 1
                continue
            kept += 1
            dens.append(sw / max(len(lab), 1))

    total = sum(counts.values())
    print(f"pages processed           {counts['pages']:>6}")
    print(f"band/line pairs available {counts['pairs_available']:>6}")
    print(f"KEEPABLE                  {kept:>6}")
    if dens:
        d = sorted(dens)
        print(f"  px/char among kept:  p10 {d[len(d)//10]:.1f}  median "
              f"{d[len(d)//2]:.1f}  p90 {d[9*len(d)//10]:.1f}")
    print(f"\nrejections (what eats the yield):")
    for k in ("label_too_long", "illegible_band", "too_dense_to_read",
              "label_too_short"):
        print(f"  {k:<20}{counts[k]:>6}   "
              f"{100*counts[k]/max(1,counts['pairs_available']):5.1f}% of pairs")

    print("\n--- label_too_long is a CONFIG VALUE ---")
    print(f"  cap is {MAX_LABEL} chars; {counts['label_too_long']} pairs rejected")
    if long_labels:
        lens = [len(l) for _, l in long_labels]
        print(f"  rejected label lengths: min {min(lens)} median "
              f"{sorted(lens)[len(lens)//2]} max {max(lens)}")
        print(f"  of those, <=60 chars: {sum(1 for l in lens if l <= 60)} "
              f"(raisable without touching the model)")
    print("\n--- illegible_band is PROTECTING label quality ---")
    print("  relaxing it trains the model to read noise as text. Do not.")

    print("\n--- page_count_mismatch ---")
    print(f"  {counts['page_count_mismatch']}/{counts['pages']} pages have a "
          f"band/line count mismatch; those pairs are positional guesses.")


if __name__ == "__main__":
    main()