"""
Side-by-side: the quick bench's feed path vs the project's natural-width path,
on the same cache crops, same model. The divergence points at the bug.

  path BENCH   : pad crop to width 256, decode timesteps w//8
  path NATURAL : feed crop at its natural width, decode ALL timesteps

If NATURAL reads labels and BENCH emits garbage, the pad/slice logic is the
bug. If both are garbage, the bug is upstream (charset, checkpoint, cache).

Run:  bash run.sh py ml/experiments/bench_compare.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/ubuntu/Projects/sightline")
sys.path.insert(0, str(ROOT / "ml/src"))
import realdata as R                        # noqa: E402
from ocr_model import build as build_model  # noqa: E402

CKPT = "/tmp/crnn_before_realphoto.pt"      # shipped 40-epoch model
CACHE = "/tmp/handheld_rebuilt.npz"


def feed_natural(model, img_u8, w):
    x = img_u8[:, :w].astype(np.float32)
    x = (x / 255.0 - 0.5) / 0.5
    with torch.no_grad():
        pred = model(torch.from_numpy(x[None, None]))
    idx = pred.argmax(-1).numpy().reshape(-1)
    return R.decode_greedy(idx)


def feed_bench(model, img_u8, w):
    x = np.zeros((R.CROP_H, 256), np.float32)
    x[:, :w] = img_u8[:, :w]
    x = (x / 255.0 - 0.5) / 0.5
    with torch.no_grad():
        pred = model(torch.from_numpy(x[None, None]))
    idx = pred.argmax(-1).numpy().reshape(-1)
    return R.decode_greedy(idx[: max(1, w // 8)])


def main():
    d = np.load(CACHE, allow_pickle=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print("ckpt epoch", ck.get("epoch"), "| charset match:",
          "".join(ck["charset"]) == R.CHARSET)
    m = build_model(ck["num_classes"])
    m.load_state_dict(ck["model"])
    m.eval()

    test_idx = [i for i, s in enumerate(d["split"]) if int(s) == 2][:8]
    print(f"\n{'label':<34}{'NATURAL':<26}{'BENCH':<26}")
    print("-" * 86)
    for i in test_idx:
        w = min(int(d["widths"][i]), 256)
        img = d["images"][i]
        want = str(d["labels"][i])
        nat = feed_natural(m, img, w)
        ben = feed_bench(m, img, w)
        print(f"{want[:33]:<34}{nat[:25]:<26}{ben[:25]:<26}")


if __name__ == "__main__":
    main()