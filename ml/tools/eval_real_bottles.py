"""
eval_real_bottles.py — the number that decides whether any of this worked.

    bash run.sh py ml/tools/eval_real_bottles.py                  # real photographs
    bash run.sh py ml/tools/eval_real_bottles.py --synthetic      # the bottle corpus
    bash run.sh py ml/tools/eval_real_bottles.py --ckpt models/ocr/crnn.pre-bottle.pt

WHY IT EXISTS SEPARATELY FROM eval-render
-----------------------------------------
`eval-render` scores the synthetic renders the model was trained on, so it can only
tell you the model learned its own renderer. The doses question is about REAL
photographs of curved, glossy packaging, and every published number for that path so
far has had no ground truth at all. `data/real_bottles/truth/*.txt` is the only real
truth in the repo (transcribed off the pixels by a person, `[?]` where unreadable),
so this harness is deliberately small, slow and explicit rather than clever.

WHAT IT PRINTS, AND WHAT IT REFUSES TO PRINT
--------------------------------------------
* lines detected vs lines transcribed — the reader may simply not FIND the panel, and
  that is a different failure from misreading it;
* paired exact match and CER ONLY where the two counts agree. Positional pairing
  across different counts scores the pairing, not the reading (the same rule
  `sightline-ml-validation-gates` §18 applies to receipt pages);
* a page with no transcript is not scored, and a transcript that is mostly `[?]` is
  capped, because a guessed label silently becomes a ceiling.

Small n is stated in the output. Two pages are a smoke test, not a rate.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

import ctc_decode as CD      # noqa: E402
import eval_ocr as EO        # noqa: E402
import ocr_model as OM       # noqa: E402
import realdata as R         # noqa: E402

REAL = ROOT / "data" / "real_bottles"


def words(s):
    """Tokens for an ORDER-FREE comparison.

    NOT `realdata.norm_label`: that keeps a string comparable as a whole and strips
    whitespace on purpose, so using it on a joined page collapses 60 words into one
    token and every recall comes out 0 (measured, twice).
    """
    import re
    return [w for w in (re.sub(r"[^A-Z0-9]", "", t)
                        for t in str(s).upper().split()) if w]


def read_page(model, gray):
    """Every line the product's own reader gets off one photograph."""
    got = EO.read_page(model, "cpu", gray)
    return [g[0] for g in got]


def score_real(model, verbose=True):
    pages, scored, exact, cer_sum = 0, 0, 0, 0.0
    lines_tot, lines_ok = 0, 0
    pages_wo, words_hit, words_truth, words_read = 0, 0, 0, 0
    for img in sorted((REAL / "raw").glob("*")):
        t = REAL / "truth" / (img.stem + ".txt")
        if not t.exists():
            continue
        truth = [l.strip() for l in t.read_text().splitlines()
                 if l.strip() and l.strip() != "[?]"]
        if not truth:
            continue
        pages += 1
        from PIL import Image
        gray = np.asarray(Image.open(img).convert("L"), np.uint8)
        got = read_page(model, gray)
        print("\n== %s" % img.name[:60])
        print("   detected %d lines, transcribed %d" % (len(got), len(truth)))
        if verbose:
            for g in got[:10]:
                print("     read: %r" % g[:70])
        if len(got) != len(truth):
            print("   NOT PAIRED (counts differ) — no positional score; a positional "
                  "pairing here would measure the pairing")
        if len(got) != len(truth):
            # Order-free word recall: of the words a person read off this bottle,
            # how many does the reader produce ANYWHERE on the page? This is the
            # metric that survives a count mismatch, and the mismatch is inherent —
            # the detector also fires on bottle ridges, reflections and shrink-wrap,
            # which carry no text at all. Precision is reported too, because a reader
            # that emits every word it knows would score a perfect recall.
            # Per-word, case-folded: the first version compared raw tokens, so the
            # uppercase read '1300 GOULD DRIVE' scored zero against the transcript
            # line "1300 Gould Drive, Gainesville, GA 30504, USA" on case alone.
            tw, rw = words(" ".join(truth)), words(" ".join(got))
            from collections import Counter
            hit = sum((Counter(tw) & Counter(rw)).values())
            print("   word recall %d/%d = %.1f%%   |  read precision %d/%d = %.1f%%"
                  % (hit, len(tw), 100.0 * hit / max(1, len(tw)), hit, len(rw),
                     100.0 * hit / max(1, len(rw))))
            pages_wo += 1
            words_hit += hit
            words_truth += len(tw)
            words_read += len(rw)
            continue
        e = sum(1 for a, b in zip(got, truth) if R.norm_label(a) == R.norm_label(b))
        c = CD.cer_aggregate(got, truth)
        scored += 1
        exact += e
        cer_sum += c
        lines_tot += len(truth)
        lines_ok += e
        print("   PAIRED: exact %d/%d lines   CER %.3f" % (e, len(truth), c))
    print("\n--- REAL photographs: %d with a transcript, %d positionally scored, "
          "%d scored order-free" % (pages, scored, pages_wo))
    if pages_wo:
        print("    ORDER-FREE word recall %d/%d = %.1f%%   read precision %.1f%% "
              "(n=%d pages, counts did not match so no positional score was possible)"
              % (words_hit, words_truth, 100.0 * words_hit / max(1, words_truth),
                 100.0 * words_hit / max(1, words_read), pages_wo))
    if scored:
        print("    line exact %d/%d = %.1f%%   mean page CER %.3f   n=%d pages"
              % (lines_ok, lines_tot, 100.0 * lines_ok / max(1, lines_tot),
                 cer_sum / scored, scored))
        if scored < 5:
            print("    n<5: a smoke test, NOT a rate. Report it with n attached.")
    return scored, lines_ok, lines_tot


def score_synthetic(model, verbose=True):
    """Crop-level exact/CER on the bottle corpus's own held-out split."""
    import torch
    import build_dataset as B
    from PIL import Image
    z = np.load(ROOT / "data" / "bottle_corpus" / "crops_train.npz", allow_pickle=True)
    m = np.array([str(s) == "test" for s in z["splits"]])
    imgs, widths, labels = z["images"][m], z["widths"][m], [str(x) for x in z["labels"][m]]
    ok = 0
    preds = []
    for i in range(len(labels)):
        arr = imgs[i][:, :max(8, int(widths[i]))]
        x = (arr.astype(np.float32)[None, None] / 255.0 - 0.5) / 0.5
        with torch.no_grad():
            lg = model(torch.from_numpy(x)).cpu().numpy()[0]
        p = R.decode_greedy(lg.argmax(-1))
        preds.append(p)
        ok += int(R.norm_label(p) == R.norm_label(labels[i]))
        if verbose and i < 8:
            print("   %-34r vs %r" % (p[:34], labels[i][:34]))
    cer = CD.cer_aggregate(preds, labels)
    print("\n--- SYNTHETIC bottle split: %d crops, exact %d (%.1f%%), CER %.3f"
          % (len(labels), ok, 100.0 * ok / len(labels), cer))
    print("    (synthetic: this is a PREDICTION that curvature was learned, not "
          "evidence a real bottle is read)")
    return ok, len(labels)


def load_trained(ckpt):
    """Load the SHIPPED checkpoint and refuse to measure with random weights.

    `ocr_model.build()` is the ARCHITECTURE, not the trained model. Calling it and
    forgetting the checkpoint decodes with a random initialisation and reports 0%
    exact everywhere — indistinguishable from a model that cannot read the data, and
    it cost this project an afternoon before the canary caught it. `--synthetic`
    numbers from before this guard existed are VOID.
    """
    import torch
    path = Path(ckpt or (ROOT / "models" / "ocr" / "crnn.pt"))
    assert path.exists(), "no checkpoint at %s" % path
    model = OM.build()
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    model.eval()
    hit = 0
    for c, l in zip(_CANARY_CROPS, _CANARY_LABELS):
        a = np.asarray(c, np.uint8)
        x = (a.astype(np.float32)[None, None] / 255.0 - 0.5) / 0.5
        with torch.no_grad():
            lg = model(torch.from_numpy(x)).cpu().numpy()[0]
        hit += int(R.norm_label(R.decode_greedy(lg.argmax(-1))) == R.norm_label(l))
    assert hit >= 2, ("CANARY FAILED (%d/6 known crops read exactly) — the model in "
                      "this process is not the shipped one" % hit)
    print("checkpoint: %s — canary %d/6 known render crops exact" % (path.name, hit))
    return model


def _canaries():
    """Six known crops from the render test split, loaded once for the tripwire."""
    import build_dataset as B
    import render_corpus as RC
    from PIL import Image
    d = RC.load(split="test")
    crops, labels = [], []
    for c, l in list(zip(d["crops"], d["labels"]))[:6]:
        crops.append(B.crop_to_array(Image.fromarray(np.asarray(c, np.uint8))))
        labels.append(str(l))
    return crops, labels


_CANARY_CROPS, _CANARY_LABELS = [], []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    global _CANARY_CROPS, _CANARY_LABELS
    _CANARY_CROPS, _CANARY_LABELS = _canaries()
    model = load_trained(a.ckpt)
    if a.synthetic:
        return 0 if score_synthetic(model, not a.quiet)[0] >= 0 else 1
    score_real(model, not a.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
