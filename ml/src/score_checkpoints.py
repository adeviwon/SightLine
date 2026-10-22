"""Paired, same-crops comparison of two checkpoints on the held-out splits.

Why: retrain3 finished and has never been scored, and the punctuation experiment needs a
frozen baseline to compare against. Both models are run on the SAME crops with the SAME
metric, so the only variable is the weights.

  env -u PYTHONPATH -u VIRTUAL_ENV ./.venv/bin/python ml/src/score_checkpoints.py \
      models/ocr/crnn_epoch29_final.pt models/ocr/crnn.pt
"""
import sys, re, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))
import realdata as R                      # noqa: E402
import ctc_decode as CD                   # noqa: E402
from ocr_model import build               # noqa: E402

norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s or "").lower())
torch.set_num_threads(1)
DEV = torch.device("cpu")

SETS = [("SROIE test (scanned receipts)", "artifacts/ocr_train.npz", "splits", "test"),
        ("render corpus test (synthetic)", "data/render_corpus/crops_train.npz", "splits", "test"),
        ("real photos rectified (test)", "data/real_corpus/crops_rectified.npz", "splits", "test")]


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build(ck.get("num_classes", R.NUM_CLASSES))
    m.load_state_dict(ck["model"]); m.eval()
    return ck, m


def run(m, imgs, widths):
    out = []
    with torch.no_grad():
        for im, w in zip(imgs, widths):
            c = im[:, :int(w)]
            x = (c.astype(np.float32) / 255.0 - 0.5) / 0.5
            out.append(R.decode_greedy(m(torch.from_numpy(x[None, None]))[0].argmax(-1).numpy()))
    return out


def main():
    paths = sys.argv[1:] or ["models/ocr/crnn_epoch29_final.pt", "models/ocr/crnn.pt"]
    models = {}
    for p in paths:
        ck, m = load(p)
        models[p] = (ck, m)
        print("loaded %-38s epoch=%-4s val_word_acc=%s params=%s"
              % (Path(p).name, ck.get("epoch"), round(ck.get("val_word_acc") or 0, 4), ck.get("params")))
    print()
    for name, npz, skey, want in SETS:
        fp = ROOT / npz
        if not fp.exists():
            print("%-34s MISSING %s" % (name, npz)); continue
        d = np.load(fp, allow_pickle=True)
        images, widths, labels = d["images"], d["widths"], d["labels"]
        splits = np.asarray(d[skey])
        idx = np.where(splits == want)[0]
        rng = np.random.default_rng(0)
        if len(idx) > 1500:
            idx = rng.choice(idx, 1500, replace=False)
        imgs = [images[i] for i in idx]; ws = [widths[i] for i in idx]
        labs = [str(labels[i]) for i in idx]
        print("%-34s n=%-5d" % (name, len(idx)), end="", flush=True)
        for p in paths:
            t = time.time()
            hyp = run(models[p][1], imgs, ws)
            ex = sum(1 for h, l in zip(hyp, labs) if norm(h) == norm(l) and norm(l))
            cer = sum(CD.lev(norm(h), norm(l)) / max(1, len(norm(l))) for h, l in zip(hyp, labs)) / len(hyp)
            # token recall, the app's word-accuracy definition
            wr = []
            for h, l in zip(hyp, labs):
                gt = [w for w in norm(l).split()] or [norm(l)]
                wr.append(1.0)
            print("  | %s: exact %.1f%%  CER %.3f  (%.0fs)" % (
                Path(p).name[:22], 100*ex/len(hyp), cer, time.time()-t), end="", flush=True)
        print()


if __name__ == "__main__":
    main()
