"""
gen_wordlm_vectors.py — golden vectors that pin the JS word-LM decoder to the
Python one, on REAL crops of the shipped recogniser.

    bash run.sh py ml/src/gen_wordlm_vectors.py --n 60

WHY REAL CROPS AND NOT SYNTHETIC LOGITS
---------------------------------------
Synthetic logit matrices prove the arithmetic; they do not prove the COMPOSITION
(which transform is applied in which order, what shape the caller passes). These
vectors are the model's own output on held-out crops, so the JS side is asked the
exact question the app asks it — log-softmax then beam+word-LM — and the expected
strings were produced by the Python decoder over the SAME logits.

TWO CHECKS, STATED SEPARATELY
-----------------------------
* `logprobs` + `expected`: the DECODER must return the identical string. Exact,
  no tolerance, because both sides get identical input.
* `logits` + `logprobs`: the log-softmax transform must agree within 1e-5
  (float32 torch on one side, float64 Math.log on the other). Tolerance is the
  honest claim here — an exact-match demand would fail on rounding, not on a bug.

Each row records the crop's ground-truth label so a divergence can be read as
"which crop, which word", not "some array differs".
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as B      # noqa: E402
import ctc_decode as CD        # noqa: E402
import eval_decode_ab as AB    # noqa: E402
import realdata as R           # noqa: E402
import render_corpus as RC     # noqa: E402
import word_lm as WL           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "app" / "tools" / "fixtures" / "wordlm_vectors.json"
BEAM, ALPHA = 4, 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--beam", type=int, default=BEAM)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    import torch
    from PIL import Image
    data = RC.load(split=a.split)
    n = min(a.n, data["crops"].shape[0])
    arrays = [B.crop_to_array(Image.fromarray(np.asarray(c, np.uint8)))
              for c in data["crops"][:n]]
    labels = [str(x) for x in data["labels"][:n]]
    model, ckpt = AB.load_model(a.ckpt or None)

    logits = []
    for arr in arrays:
        x = (np.asarray(arr, np.float32)[None, None] / 255.0 - 0.5) / 0.5
        with torch.no_grad():
            lg = model(torch.from_numpy(x)).cpu().numpy()[0]     # [T, C]
        logits.append(lg)

    trie_path = ROOT / "app" / "data" / "word_trie.json"
    lm = WL.load_trie_json(trie_path)
    rows = []
    for i, lg in enumerate(logits):
        lps = torch.log_softmax(torch.from_numpy(lg), dim=-1).numpy().astype(np.float64)
        text = CD.beam_decode(lps, beam=a.beam, lm=lm, alpha=a.alpha)
        greedy = R.decode_greedy(lg.argmax(-1))
        rows.append({
            "label": labels[i],
            "greedy": greedy,
            "expected": text,
            "changed": text != greedy,
            "T": int(lg.shape[0]), "C": int(lg.shape[1]),
            "logits": base64.b64encode(lg.astype(np.float32).tobytes()).decode(),
            "logprobs": base64.b64encode(lps.astype(np.float64).tobytes()).decode(),
        })

    out = {"meta": {"ckpt": ckpt, "split": a.split, "n": n, "beam": a.beam,
                    "alpha": a.alpha, "trie": str(trie_path),
                    "trie_words": json.loads(trie_path.read_text())["meta"]["words"],
                    "note": "expected = Python ctc_decode.beam_decode over these "
                            "logprobs with the shipped word trie; logits are the "
                            "recogniser's raw output for the log-softmax check"},
           "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    nb = Path(a.out).stat().st_size
    print("wrote %s (%d rows, %.1f KB)  beam=%d alpha=%.2f"
          % (a.out, n, nb / 1024.0, a.beam, a.alpha))
    print("expected != greedy on %d/%d crops (the LM changing the read)"
          % (sum(1 for r in rows if r["changed"]), n))
    print("checkpoint %s  trie %s" % (Path(ckpt).name, trie_path.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
