"""
Diagnostic: what PSNR is even achievable?

Before tuning a restorer you need to know the floor. This script measures:

  identity    PSNR(degraded, clean)          <- what a do-nothing model scores
  oracle      PSNR(clean, clean) = inf        <- sanity check on the metric
  per-profile identity PSNR                   <- which profiles are hopeless

If a trained model's val PSNR is below the identity baseline for the same
patches, the model is actively destroying information and something is wrong
with the training, not the task.

Also reports how much headroom exists: a profile where identity already scores
28 dB has nothing to gain; one at 9 dB is mostly unrecoverable.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import capture  # noqa: E402
from model import SightLineNet, psnr  # noqa: E402


def mse(a, b):
    return float(np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2))


def psnr_np(a, b):
    m = max(mse(a, b), 1e-10)
    return 10.0 * np.log10(1.0 / m)


def main():
    z = np.load("artifacts/restorer_data.npz", allow_pickle=True)
    deg, cln, seeds = z["deg"], z["clean"], z["seeds"]
    print(f"{len(deg)} patches cached\n")

    id_psnr = np.array([psnr_np(d, c) for d, c in zip(deg, cln)])
    print(f"IDENTITY (no restoration at all):")
    print(f"  mean PSNR {id_psnr.mean():.2f} dB   min {id_psnr.min():.2f}   "
          f"max {id_psnr.max():.2f}\n")

    ck = Path("models/restorer/restorer.pt")
    if ck.exists():
        m = SightLineNet()
        m.load_state_dict(torch.load(ck, map_location="cpu",
                                     weights_only=True)["state_dict"])
        m.eval()
        model_psnr = []
        with torch.no_grad():
            for d, c in zip(deg, cln):
                x = torch.from_numpy(d)[None, None]
                model_psnr.append(psnr(m(x), torch.from_numpy(c)[None, None]))
        model_psnr = np.array(model_psnr)
        delta = model_psnr - id_psnr
        print(f"TRAINED MODEL (checkpoint epoch info below):")
        print(f"  mean PSNR {model_psnr.mean():.2f} dB")
        print(f"  vs identity: {delta.mean():+.2f} dB  "
              f"(better on {100*(delta>0).mean():.0f}% of patches, "
              f"worse on {100*(delta<0).mean():.0f}%)")
        ck_info = torch.load(ck, map_location="cpu", weights_only=True)
        print(f"  checkpoint: epoch {ck_info.get('epoch')}, "
              f"logged val {ck_info.get('val_psnr', 0):.2f} dB")
    else:
        model_psnr = None
        print("no checkpoint yet\n")

    print(f"\nHEADROOM BY PATCH (sorted, worst first):")
    print(f"  {'idx':>5} {'identity':>9} {'model':>8} {'delta':>8}")
    order = np.argsort(id_psnr)
    step = max(1, len(order) // 14)
    for i in order[::step]:
        m_s = f"{model_psnr[i]:8.2f}" if model_psnr is not None else "       —"
        d_s = (f"{model_psnr[i]-id_psnr[i]:+8.2f}"
               if model_psnr is not None else "       —")
        print(f"  {i:>5} {id_psnr[i]:9.2f} {m_s} {d_s}")
    print(f"\nInterpretation: identity mean {id_psnr.mean():.2f} dB is the floor.")
    print(f"A trained model must EXCEED it. If it does not, lower the learning")
    print(f"rate — a model that scores below identity is deleting information.")


if __name__ == "__main__":
    main()
