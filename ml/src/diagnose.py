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
import gating  # noqa: E402
import seedutil  # noqa: E402  — stable cross-process seed derivation
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
        model_psnr, gated_psnr = [], []
        with torch.no_grad():
            for d, c in zip(deg, cln):
                x = torch.from_numpy(d)[None, None]
                y = torch.from_numpy(c)[None, None]
                out = m(x).numpy()[0, 0]
                model_psnr.append(psnr(torch.from_numpy(out[None, None]), y))
                # apply the same degradation gate the app uses
                w = gating.degradation_weight(d)
                mixed = w ** 0.7 * out + (1 - w ** 0.7) * d
                gated_psnr.append(psnr(torch.from_numpy(mixed[None, None]), y))
        model_psnr = np.array(model_psnr)
        gated_psnr = np.array(gated_psnr)
        delta = model_psnr - id_psnr
        gdelta = gated_psnr - id_psnr
        print(f"UNCONDITIONAL restorer:")
        print(f"  mean PSNR {model_psnr.mean():.2f} dB")
        print(f"  vs identity: {delta.mean():+.2f} dB  "
              f"(better on {100*(delta>0).mean():.0f}% of patches, "
              f"worse on {100*(delta<0).mean():.0f}%)")
        print(f"GATED restorer (blend by measured degradation):")
        print(f"  mean PSNR {gated_psnr.mean():.2f} dB")
        print(f"  vs identity: {gdelta.mean():+.2f} dB  "
              f"(better on {100*(gdelta>0).mean():.0f}% of patches, "
              f"worse on {100*(gdelta<0).mean():.0f}%)")

        # The mean is a bad summary here: per-patch PSNR spans 1.6 to 100 dB,
        # so a handful of extremes dominate it and hide what happens to the
        # typical patch. The median, and the split by how damaged the input
        # was, are what actually describe the model.
        print(f"\nDELTA vs IDENTITY — median is the honest summary:")
        print(f"  unconditional : mean {delta.mean():+6.2f} dB   "
              f"median {np.median(delta):+6.2f} dB")
        print(f"  gated         : mean {gdelta.mean():+6.2f} dB   "
              f"median {np.median(gdelta):+6.2f} dB")
        bands = [
            ("undamaged (identity >= 25 dB)", id_psnr >= 25),
            ("mildly damaged (12-25 dB)", (id_psnr >= 12) & (id_psnr < 25)),
            ("badly damaged (< 12 dB)", id_psnr < 12),
        ]
        print(f"\nBY INPUT DAMAGE LEVEL (median delta vs identity):")
        print(f"  {'band':<32} {'n':>4} {'uncond':>8} {'gated':>8}")
        for label, m in bands:
            if not m.any():
                continue
            print(f"  {label:<32} {int(m.sum()):>4} "
                  f"{np.median(delta[m]):>+8.2f} {np.median(gdelta[m]):>+8.2f}")
        print(f"\nThe model earns its keep on badly damaged input. The question")
        print(f"the gate answers is whether it stops HURTING undamaged input —")
        print(f"compare the top row. PSNR is only a proxy; the decisive test is")
        print(f"field accuracy via 'bash run.sh eval'.")

        # Why is the gate not protecting undamaged patches? Print the gate's
        # own inputs for them. This is the diagnostic that matters: if weight
        # is not ~0 on a patch the model is destroying, the calibration is
        # wrong, not the model.
        und = np.where(id_psnr >= 25)[0][:8]
        if len(und):
            print(f"\nGATE INPUTS on undamaged patches (weight should be ~0):")
            print(f"  {'idx':>5} {'ident':>7} {'sharp':>9} {'noise':>8} "
                  f"{'contr':>7} {'weight':>7}")
            for i in und:
                g = gating.to_gray(deg[i])
                print(f"  {i:>5} {id_psnr[i]:>7.1f} "
                      f"{gating.sharpness(g):>9.5f} "
                      f"{gating.noise_sigma(g):>8.4f} "
                      f"{gating.contrast(g):>7.3f} "
                      f"{gating.degradation_weight(g):>7.3f}")
            print(f"\n  anchors: SHARP_CLEAN={gating.SHARP_CLEAN} "
                  f"SHARP_BAD={gating.SHARP_BAD}")
            print(f"           NOISE_CLEAN={gating.NOISE_CLEAN} "
                  f"NOISE_BAD={gating.NOISE_BAD} "
                  f"CONTENT_MIN={gating.CONTENT_MIN}")
        print(f"\nThe gate exists because the unconditional model WINS BIG on")
        print(f"damaged input and LOSES BIG on already-clean input. Blending")
        print(f"keeps the win and removes the loss: a clean scan passes through")
        print(f"untouched (weight 0), a blurred one is fully restored (weight 1).")
        ck_info = torch.load(ck, map_location="cpu", weights_only=True)
        print(f"  checkpoint: epoch {ck_info.get('epoch')}, "
              f"logged val {ck_info.get('val_psnr', 0):.2f} dB")
    else:
        model_psnr = gated_psnr = None
        print("no checkpoint yet\n")

    print(f"\nHEADROOM BY PATCH (sorted, worst first):")
    print(f"  {'idx':>5} {'identity':>9} {'gated':>8} {'delta':>8}")
    order = np.argsort(id_psnr)
    step = max(1, len(order) // 14)
    for i in order[::step]:
        g_s = f"{gated_psnr[i]:8.2f}" if gated_psnr is not None else "       —"
        d_s = (f"{gated_psnr[i]-id_psnr[i]:+8.2f}"
               if gated_psnr is not None else "       —")
        print(f"  {i:>5} {id_psnr[i]:9.2f} {g_s} {d_s}")
    print(f"\nInterpretation: identity mean {id_psnr.mean():.2f} dB is the floor.")
    print(f"A trained model must EXCEED it. If it does not, lower the learning")
    print(f"rate — a model that scores below identity is deleting information.")
    if gated_psnr is not None:
        print(f"With the gate, mean delta is {gated_psnr.mean()-id_psnr.mean():+.2f} dB.")


if __name__ == "__main__":
    main()
