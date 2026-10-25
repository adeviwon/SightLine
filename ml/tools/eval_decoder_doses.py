"""
eval_decoder_doses.py — which WORD LM actually decodes medication doses best.

    bash run.sh py ml/tools/eval_decoder_doses.py                       # both splits
    bash run.sh py ml/tools/eval_decoder_doses.py --corpus bottle
    bash run.sh py ml/tools/eval_decoder_doses.py --configs shipped,dose,union

WHY THIS EXISTS, AND WHY IT IS PAIRED
-------------------------------------
The decoder's own levers are the only ones left that cost nothing at inference: a
better lexicon and a better weight cost ZERO extra forward passes. But this repo has
already been burned twice by decoder claims that did not survive a proper test:

  * a sweep that ranks every cell against GREEDY cannot rank cells against each
    other ('bigram40' looked +10 crops over the shipped config and was 10 crops NET
    WORSE when tested paired);
  * a lexicon's coverage table looks like a quality argument and is not one — a word
    the trie lacks is still decodable, it just pays the penalty.

So every config here is decoded from the SAME logits, on held-out crops, and every
comparison is PAIRED exact McNemar against both greedy and the shipped config. The
product metric is exact match; CER is reported as an instrument.

CORPORA
-------
* `bottle` — data/bottle_corpus test split: real FDA panel text, wrapped on a
  cylinder, photographed. This is the doses-domain set.
* `render` — data/render_corpus test split: the documents the product already
  serves, i.e. the no-regression set. A doses lexicon that wins on bottles while
  breaking prescriptions is not a win.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

import ctc_decode as CD        # noqa: E402
import ocr_model as OM         # noqa: E402
import realdata as R           # noqa: E402
import word_lm as WL           # noqa: E402

SHIPPED_TRIE = ROOT / "app" / "data" / "word_trie.json"
DOSE_TRIE = ROOT / "app" / "data" / "dose_trie.json"
DOSE_SMALL = ROOT / "app" / "data" / "dose_trie_small.json"
# The trie that shipped BEFORE the doses LM replaced it. Kept as a named artifact because
# once the app's word_trie.json becomes the union, "shipped" in a table silently means the
# UNION and every paired comparison loses the baseline it was supposed to be against.
DOCS_ONLY = ROOT / "app" / "data" / "word_trie_docs_only.json"
DOSE_UNION = ROOT / "app" / "data" / "dose_trie_union.json"
DOSE_BG40 = ROOT / "app" / "data" / "dose_trie_bg40.json"
DOSE_BG80 = ROOT / "app" / "data" / "dose_trie_bg80.json"
UNION_BG40 = ROOT / "app" / "data" / "dose_trie_union_bg40.json"
UNION_S05 = ROOT / "app" / "data" / "dose_trie_union_s05.json"
UNION_S20 = ROOT / "app" / "data" / "dose_trie_union_s20.json"
UNION_MC2 = ROOT / "app" / "data" / "dose_trie_union_mc2_a35.json"

CONFIGS = {
    # name -> (lexicon/trie source, beam, alpha)
    "greedy": (None, 4, 0.0),
    "shipped": (SHIPPED_TRIE, 4, 0.25),
    "docs": (DOCS_ONLY, 4, 0.25),        # the pre-doses baseline, always comparable
    "dose": (DOSE_TRIE, 4, 0.25),
    "dose-beam8": (DOSE_TRIE, 8, 0.25),
    # The trie is PRECACHED, so its size is a real cost, not a detail: the shipped one
    # is 222 KB and this one is 1.55 MB. `small` keeps count>=5 plus the unit/direction
    # vocabulary that dose language is made of; `union` adds the shipped words back so
    # the product's document path cannot lose vocabulary.
    "dose-small": (DOSE_SMALL, 4, 0.25),
    "dose-union": (DOSE_UNION, 4, 0.25),
    # The same doses lexicon with REAL word bigrams shallow-fused at word boundaries:
    # the actual word-LM part of a word-LM beam decoder.
    "dose-bg40": (DOSE_BG40, 4, 0.25),
    "dose-bg80": (DOSE_BG80, 4, 0.25),
    # alpha is the LM's weight and the lexicon it weights just changed size by 2.2x,
    # so the value tuned for the old lexicon is not automatically right for this one.
    "small-a15": (DOSE_SMALL, 4, 0.15),
    "small-a35": (DOSE_SMALL, 4, 0.35),
    "small-a45": (DOSE_SMALL, 4, 0.45),
    "union-a45": (DOSE_UNION, 4, 0.45),
    # min-count 5 pruned real drug names out of the vocabulary (METHYLPHENIDATE x3,
    # ANASTROZOLE x2, ATORVASTATIN x2 in the mined panels) — a doses LM that cannot
    # spell the drug on the bottle is missing the point of the artifact.
    "mc2-a35": (UNION_MC2, 4, 0.35),
    "union-s05-a35": (UNION_S05, 4, 0.35),
    "union-s20-a35": (UNION_S20, 4, 0.35),
    "union-bg40-a35": (UNION_BG40, 4, 0.35),
    # The union keeps the shipped documents' vocabulary, so the product's existing
    # path cannot lose words to a doses-specific change.
    "dose-union2": (DOSE_UNION, 4, 0.25),
    "dose-union-bg40": (UNION_BG40, 4, 0.25),
    "union-s05": (UNION_S05, 4, 0.25),
    "union-s20": (UNION_S20, 4, 0.25),
    "union-a35": (DOSE_UNION, 4, 0.35),
}


def load_corpus(kind, limit=0):
    """(images, widths, labels) for a held-out split, in trainer-cache geometry."""
    if kind in ("bottle", "flat"):
        cache = "flat_doses_corpus" if kind == "flat" else "bottle_corpus"
        z = np.load(ROOT / "data" / cache / "crops_train.npz", allow_pickle=True)
        m = np.array([str(s) == "test" for s in z["splits"]])
        imgs, widths = z["images"][m], z["widths"][m]
        labels = [str(x) for x in z["labels"][m]]
        # the cache is right-padded to maxW; crop each row back to its true width
        rows = [imgs[i][:, :max(8, int(widths[i]))] for i in range(len(labels))]
    else:
        import build_dataset as B
        import render_corpus as RC
        from PIL import Image
        d = RC.load(split="test")
        rows, labels = [], []
        for c, l in zip(d["crops"], d["labels"]):
            rows.append(B.crop_to_array(Image.fromarray(np.asarray(c, np.uint8))))
            labels.append(str(l))
    if limit:
        rows, labels = rows[:limit], labels[:limit]
    return rows, labels


def load_trained(ckpt):
    """Load the SHIPPED checkpoint, and refuse to measure with random weights.

    `ocr_model.build()` returns the ARCHITECTURE — no weights. A harness that calls
    it and forgets the checkpoint silently decodes with a random initialisation and
    reports 0% exact on every configuration, which reads exactly like a model that
    cannot read the data. That is not hypothetical: it is how the first version of
    this file "proved" the doses corpora unreadable, and how an afternoon of
    hypothesis-killing was run against noise. The canary below is the tripwire.
    """
    import torch
    path = Path(ckpt or (ROOT / "models" / "ocr" / "crnn.pt"))
    assert path.exists(), "no checkpoint at %s" % path
    model = OM.build()
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    model.eval()
    return model, path


def canary(model):
    """One known crop the shipped model must read exactly, or we are not measuring."""
    import build_dataset as B
    import render_corpus as RC
    from PIL import Image
    import torch
    d = RC.load(split="test")
    hit = 0
    for c, l in list(zip(d["crops"], d["labels"]))[:6]:
        a = B.crop_to_array(Image.fromarray(np.asarray(c, np.uint8)))
        x = (a.astype(np.float32)[None, None] / 255.0 - 0.5) / 0.5
        with torch.no_grad():
            lg = model(torch.from_numpy(x)).cpu().numpy()[0]
        hit += int(R.norm_label(R.decode_greedy(lg.argmax(-1))) == R.norm_label(str(l)))
    assert hit >= 2, ("CANARY FAILED (%d/6 known crops read exactly): the model in this "
                      "process is not the shipped one — check --ckpt before believing "
                      "ANY number below" % hit)
    return hit


def logits_for(model, rows):
    import torch
    out = []
    with torch.no_grad():
        for arr in rows:
            x = (np.asarray(arr, np.float32)[None, None] / 255.0 - 0.5) / 0.5
            out.append(model(torch.from_numpy(x)).cpu().numpy()[0])
    return out


def trie_for(path):
    if path is None:
        return None
    return WL.load_trie_json(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="bottle,render")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "decoder_doses.json"))
    a = ap.parse_args()

    import torch
    torch.set_num_threads(max(1, a.threads))
    model, ckpt = load_trained(a.ckpt)
    print("checkpoint: %s — canary %d/6 known render crops exact" % (ckpt.name, canary(model)))

    asked = [c.strip() for c in a.configs.split(",") if c.strip()]
    # A mistyped config must not be SILENTLY DROPPED: this filtered unknown names out,
    # so a run that measured six configs looked like it measured seven and the missing
    # one simply never appeared in the table.
    unknown = [c for c in asked if c not in CONFIGS]
    assert not unknown, "unknown config(s) %s; known: %s" % (unknown, sorted(CONFIGS))
    # `greedy` is the reference every comparison is printed against, so it is always
    # measured even when the caller forgets to ask for it (asking for it used to be
    # load-bearing and undocumented).
    names = asked if "greedy" in asked else ["greedy"] + asked
    report = {}
    for kind in a.corpus.split(","):
        rows, labels = load_corpus(kind, a.limit)
        t0 = time.time()
        lps = logits_for(model, rows)
        n = len(labels)
        print("\n=== %s split: %d held-out crops (logits in %.0fs)"
              % (kind, n, time.time() - t0), flush=True)
        bits, res = {}, {}
        for name in names:
            src, beam, alpha = CONFIGS[name]
            lm = trie_for(src) if alpha > 0 else None
            preds = [CD.beam_decode(lp, beam=beam, lm=lm, alpha=alpha) for lp in lps] \
                if alpha > 0 else [R.decode_greedy(lp.argmax(-1)) for lp in lps]
            bits[name] = CD.exact_match_bits(preds, labels)
            res[name] = {"exact": int(sum(bits[name])), "rate": float(sum(bits[name])) / n,
                         "cer": float(CD.cer_aggregate(preds, labels)),
                         "trie": (src.name if src else None), "beam": beam, "alpha": alpha}
        base = "greedy"
        print("  %-11s %7s %7s %7s   %s" % ("config", "exact", "rate", "CER", "vs greedy / vs shipped"))
        for name in names:
            vg = CD.pair_counts(bits[base], bits[name])
            pg = CD.mcnemar_exact(vg["variant_only"], vg["greedy_only"])
            line = "  %-11s %5d/%d %6.2f%% %7.4f   +%d/-%d p=%.4f" % (
                name, res[name]["exact"], n, 100.0 * res[name]["rate"], res[name]["cer"],
                vg["variant_only"], vg["greedy_only"], pg)
            if name != "shipped" and "shipped" in bits:
                vs = CD.pair_counts(bits["shipped"], bits[name])
                ps = CD.mcnemar_exact(vs["variant_only"], vs["greedy_only"])
                line += "  |  vs shipped +%d/-%d p=%.4f" % (
                    vs["variant_only"], vs["greedy_only"], ps)
                res[name]["vs_shipped"] = {"fixes": vs["variant_only"],
                                           "breaks": vs["greedy_only"], "p": ps}
            print(line, flush=True)
            res[name]["vs_greedy"] = {"fixes": vg["variant_only"],
                                      "breaks": vg["greedy_only"], "p": pg}
        for name in names:
            if res[name]["trie"]:
                p = ROOT / "app" / "data" / res[name]["trie"]
                res[name]["trie_kb"] = round(p.stat().st_size / 1024, 1)
        report[kind] = {"n": n, "results": res}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("\nwrote %s" % a.out)
    print("READ IT LIKE THIS: a config is only a candidate if it beats `shipped` "
          "PAIRED (p<0.05, fixes > breaks) on the corpus it is FOR, and does not lose "
          "on the other one. Weights and lexicons are free at inference; a bigger trie "
          "is not free in a precache budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
