"""
Why does ml/experiments/eval_realphoto.py measure the shipped 40-epoch model at
101.6% CER when eval_ocr.py measures the SAME model at 56.5%? A CER above 100%
means the output is longer than the truth and almost entirely wrong -- i.e. the
quick bench is feeding the model something it never saw in training.

Three suspects, one script:

1. cache dtype: if images are stored as float 0..1 but the bench divides by 255
   again, every pixel collapses toward black and the model sees mud.
2. padding value: the bench pads with uint8 zeros AFTER slicing -- training may
   pad with something else (e.g. white 255).
3. normalization: train_ocr.py's exact pixel transform vs the bench's
   (x/255 - 0.5)/0.5.

Run:  bash run.sh py ml/experiments/check_norm.py
"""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/Projects/sightline")


def main():
    print("=== 1. cache dtype / range ===")
    for p in ("artifacts/ocr_train.npz", "artifacts/handheld_rebuilt.npz"):
        f = ROOT / p
        if not f.exists():
            print(f"  {p}: MISSING (fetch-data / rebuild needed)")
            continue
        d = np.load(f, allow_pickle=True)
        im = d["images"]
        print(f"  {p.split('/')[-1]}: dtype={im.dtype} min={im.min()} "
              f"max={im.max()} mean={im.mean():.2f}")

    print("\n=== 2. how train_ocr.py transforms pixels ===")
    src = (ROOT / "ml/src/train_ocr.py").read_text()
    hits = [ln.rstrip() for ln in src.splitlines()
            if re.search(r"255|astype|/ 2|0\.5|normalize|float", ln)
            and not ln.strip().startswith("#")]
    for ln in hits[:12]:
        print("  " + ln)

    print("\n=== 3. how eval_ocr.py prepares a crop before the model ===")
    src = (ROOT / "ml/src/eval_ocr.py").read_text()
    hits = [ln.rstrip() for ln in src.splitlines()
            if re.search(r"255|astype|0\.5|prepare|float32", ln)
            and not ln.strip().startswith("#")]
    for ln in hits[:12]:
        print("  " + ln)

    print("\n=== verdict hints ===")
    print("  If cache is float 0..1 and bench divides by 255 -> that's the bug.")
    print("  If train normalizes differently -> copy ITS transform verbatim.")


if __name__ == "__main__":
    main()