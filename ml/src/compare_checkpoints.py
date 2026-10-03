"""
Which checkpoint is actually better?

Two runs used different val splits, so their numbers cannot be compared:

    epoch 20,  30,021 crops / 260 receipts / 39 val receipts  -> 29.7% word acc
    epoch 26,  52,554 crops / 420 receipts / 63 val receipts  -> 28.1% word acc

29.7 > 28.1 on the face of it, but they were scored on DIFFERENT held-out sets,
so the comparison is meaningless. This evaluates both on the SAME data: the
347 SROIE test receipts, which neither run trained on and neither used for
validation.

That is the only comparison that answers the question. If the smaller corpus
wins on the same set, the 75% data increase plus perspective augmentation did
not help and should not be shipped as an improvement.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_ocr as E   # noqa: E402
import realdata as R   # noqa: E402
from ocr_model import build as build_model  # noqa: E402


def load(path):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    m = build_model(ck["num_classes"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def main():
    torch.set_num_threads(2)
    cands = [
        ("ep20 small corpus (260 rec)", Path("/tmp/crnn_ep22_best.pt")),
        ("ep26 large corpus (420 rec)", R.ROOT / "models" / "ocr" / "crnn.pt"),
    ]

    print("Evaluating on the SROIE TEST receipts -- the same 347 pages for")
    print("every candidate. Neither run trained or validated on these.\n")
    print(f"{'checkpoint':32} {'word acc':>9} {'CER':>7}  n_words")
    print("-" * 62)

    for name, p in cands:
        if not p.exists():
            print(f"{name:32}  MISSING ({p})")
            continue
        m, ck = load(p)
        r = E.eval_sroie_test(m, torch.device("cpu"), limit=400)
        if r is None:
            print(f"{name:32}  no test parquet")
            continue
        print(f"{name:32} {r['word_acc']*100:8.2f}% {r['cer']*100:6.2f}%  "
              f"{r.get('n_words', r.get('n', 0))}")
        # per-category so a regression can be attributed
        for k in sorted(r.get("per", {})):
            v = r["per"][k]
            print(f"    {k:28} {v[1]/max(v[0],1)*100:6.2f}%  ({v[0]} crops)")


if __name__ == "__main__":
    main()