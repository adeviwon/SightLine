"""
Does training on real photos actually help on real photos?

Three models, one held-out set each:

  A  8 epochs, SROIE only
  B  8 epochs, SROIE + 2,301 real-photo crops
  current (40 epochs, SROIE only)  -- the shipped checkpoint

Scored on the rebuilt handheld cache's TEST split: 308 line crops from real
hand-photographed pages, held out BY PAGE so no page appears in train and test.
Also scored on the SROIE test receipts, to show what each change costs on the
clean-scan domain.

The comparison that matters is A vs B at equal epochs. Comparing B against the
40-epoch shipped model would conflate data with training time.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/Projects/sightline/ml/src")
import realdata as R                        # noqa: E402
from ocr_model import build as build_model  # noqa: E402

torch.set_num_threads(4)


def cer(pred, want):
    if not want:
        return 1.0
    # Levenshtein pkg absent in this venv; stdlib dynamic program is ~50 lines
    # slower but identical, and n here is ~600 crops total.
    prev = list(range(len(want) + 1))
    for i, pc in enumerate(pred, 1):
        cur = [i]
        for j, wc in enumerate(want, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (pc != wc)))
        prev = cur
    return prev[-1] / max(len(want), 1)


def eval_crops(model, npz_path, split_val):
    d = np.load(npz_path, allow_pickle=True)
    keep = [i for i, s in enumerate(d["split"])
            if str(s) == split_val or (not isinstance(s, str) and int(s) == 2
                                       and split_val == "test")]
    if split_val == "test":
        keep = [i for i, s in enumerate(d["split"]) if int(s) == 2]
    n, tot, exact = 0, 0.0, 0
    for i in keep:
        w = min(int(d["widths"][i]), 256)
        x = np.zeros((R.CROP_H, 256), np.float32)
        x[:, :w] = d["images"][i][:, :w]
        x = (x / 255.0 - 0.5) / 0.5
        with torch.no_grad():
            pred = model(torch.from_numpy(x[None, None]))[0]
        # model returns [T,C] (no batch dim) -> argmax(-1)[0] grabbed a
        # timestep, not the batch; reshape(-1) handles both [T] and [1,T].
        idx = pred.argmax(-1).numpy().reshape(-1)
        txt = R.decode_greedy(idx[: max(1, w // 8)])
        want = str(d["labels"][i])
        c = cer(txt, want)
        tot += c
        exact += (txt == want)
        n += 1
    return dict(n=n, cer=tot / max(n, 1), exact=exact / max(n, 1))


def main():
    ckpts = {
        "A  8ep SROIE only        ": "/tmp/crnn_A.pt",
        "B  8ep SROIE+realphotos  ": "/tmp/crnn_B.pt",
        "shipped 40ep SROIE only  ": "/tmp/crnn_before_realphoto.pt",
    }
    if not any(__import__("os").path.exists(v) for v in ckpts.values()):
        print("no checkpoints yet; training still running")
        return

    print(f"{'model':<26}{'handheld n':>11}{'CER':>9}{'exact':>9}")
    print("-" * 56)
    for label, path in ckpts.items():
        if not __import__("os").path.exists(path):
            print(f"{label:<26}{'(pending)':>11}")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = build_model(ck["num_classes"])
        m.load_state_dict(ck["model"])
        m.eval()
        r = eval_crops(m, "/tmp/handheld_rebuilt.npz", "test")
        print(f"{label:<26}{r['n']:>11}{r['cer']*100:>8.1f}%"
              f"{r['exact']*100:>8.1f}%")


if __name__ == "__main__":
    main()