"""
Does training on real photos help on real photos? Measured through the OFFICIAL
pipeline (eval_ocr.eval_handheld -- detection + recognition + page-level
alignment), not the quick cache bench.

Why not the cache bench: its band-to-label pairing is positional and 167/187
pages have a count mismatch, so a model reading the pixels CORRECTLY still
scores huge CER against labels that belong to a different line. The official
path aligns predicted lines to transcript lines per page and reports detection
recall separately, which is the honest product measurement.

Models compared (all on the SAME first-20 handheld pages, so the shipped row
should reproduce the historical 56.5% CER -- that reproduction validates the
harness before we trust the A/B rows):

  shipped 40ep SROIE only      (the current model in the repo)
  A        8ep  SROIE only     (fresh control)
  B        8ep  SROIE + 2,301 real-photo crops

Run:  bash run.sh py ml/experiments/eval_official_ab.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path("/home/ubuntu/Projects/sightline")
sys.path.insert(0, str(ROOT / "ml/src"))
import eval_ocr as E                        # noqa: E402
import realdata as R                        # noqa: E402
from ocr_model import build as build_model  # noqa: E402

MODELS = [
    ("shipped 40ep SROIE only    ", "/tmp/crnn_before_realphoto.pt"),
    ("A       8ep  SROIE only    ", "/tmp/crnn_A.pt"),
    ("B       8ep  SROIE+2,301noisy", "/tmp/crnn_B.pt"),
    ("B2      8ep  SROIE+362strict", "/tmp/crnn_B2.pt"),
]
LIMIT = 20   # matches the historical 20-page handheld evaluation


def main():
    torch.set_num_threads(4)
    device = torch.device("cpu")
    print(f"official eval_handheld, first {LIMIT} handheld pages, all models\n")
    print(f"{'model':<29}{'recall':>8}{'CER':>8}{'line-exact':>12}"
          f"{'lines':>7}{'missed':>8}")
    print("-" * 74)
    for label, path in MODELS:
        if not Path(path).exists():
            print(f"{label:<29}  (checkpoint missing: {path})")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = build_model(ck.get("num_classes", R.NUM_CLASSES))
        m.load_state_dict(ck["model"])
        m.eval()
        r = E.eval_handheld(m, device, limit=LIMIT)
        print(f"{label:<29}{r['detection_recall']*100:>7.1f}%"
              f"{r['cer']*100:>7.1f}%{r['line_acc']*100:>11.1f}%"
              f"{r['n']:>7}{r['missed']:>8}")


if __name__ == "__main__":
    main()