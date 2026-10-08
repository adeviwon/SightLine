"""
Step 3 metrics for the hackathon brief: accuracy AND F1, done honestly.

The brief asks for "at least two performance metrics -- accuracy and F1 score".
The stored eval already contains both, and they read:

    test_accuracy  1.0        macro_f1  1.0        n = 24
    5-fold CV      mean 1.0   std 0.0

A perfect score on 24 samples is NOT a result worth submitting. It means the
classification task is SATURATED -- the four categories use distinctive
vocabulary (sort code / dose / whereas / invoice), so a frozen MiniLM embedding
plus a linear head separates them perfectly. Reporting "100% accuracy" to
judges invites the obvious question and invites it for the wrong reason: the
answer is "our task is too easy", not "our model is excellent".

So this script:
  1. re-runs the classifier across several seeds so the spread is visible
     instead of a single lucky draw,
  2. reports accuracy, macro-F1, and PER-CLASS F1 (a saturated macro-F1 hides
     which class actually carries the result),
  3. reports bootstrap confidence intervals, because n=24 means a point
     estimate on its own is close to meaningless,
  4. and states plainly what the number is and is not evidence of.

The interesting engineering here is the OCR, not this. The classifier is
reported as an end-to-end wiring check, and the document should say so.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/ubuntu/Projects/sightline/ml/src")

import train_classifier as T   # noqa: E402
from corpus import CATEGORIES  # noqa: E402


def bootstrap_ci(correct, n_boot=4000, seed=123):
    rng = np.random.default_rng(seed)
    c = np.asarray(correct, float)
    n = len(c)
    if n == 0:
        return (0.0, 0.0)
    draws = rng.choice(c, size=(n_boot, n), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def _train(T, E_tr, y_tr, E_va, y_va, seed):
    """train_head's return arity has changed across revisions; adapt to it."""
    out = T.train_head(E_tr, y_tr, E_va, y_va, epochs=60, lr=0.002,
                       batch=32, seed=seed, verbose=False)
    if isinstance(out, tuple):
        head = out[0]
        # Some revisions return (head, history); later ones (head, history, best)
        return head, (out[1] if len(out) > 1 else None), out[-1]
    return out, None, None


def main():
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score

    torch.set_num_threads(4)
    seeds = [123, 121, 122, 124, 125]
    rows = []

    for sd in seeds:
        tr_t, tr_l, va_t, va_l, te_t, te_l = T.three_way_split(seed=sd)
        tok, model = T.load_encoder()
        # train_head indexes with torch.randperm and passes labels straight into
        # nn.CrossEntropyLoss, so BOTH the embeddings and the labels must be
        # tensors. Feeding numpy labels raises "argument 'target' must be
        # Tensor, not numpy.ndarray" -- which is what happened first.
        E_tr = torch.from_numpy(np.asarray(T.embed_corpus(tr_t, tok, model)))
        E_va = torch.from_numpy(np.asarray(T.embed_corpus(va_t, tok, model)))
        E_te = torch.from_numpy(np.asarray(T.embed_corpus(te_t, tok, model)))
        y_tr = torch.tensor([int(v) for v in tr_l])
        y_va = torch.tensor([int(v) for v in va_l])
        y_te_np = np.asarray([int(v) for v in te_l])

        # Retrain per seed so the spread reflects the whole pipeline, not one
        # fixed weight matrix.
        head, hist, _ = _train(T, E_tr, y_tr, E_va, y_va, sd)

        with torch.no_grad():
            probs = torch.softmax(head(E_te), dim=1).numpy()
        pred = probs.argmax(1)
        y_te = y_te_np

        m = T.metrics_from(probs, y_te)
        rows.append(dict(
            seed=sd,
            acc=m["accuracy"],
            macro_f1=m["macro_f1"],
            ci=m["accuracy_ci95_bootstrap"],
            per_class_f1={
                CATEGORIES[i]: round(float(f1_score(
                    y_te, pred, average=None, zero_division=0,
                    labels=list(range(len(CATEGORIES))))[i]), 4)
                for i in range(len(CATEGORIES))},
            per_class_p={
                CATEGORIES[i]: round(float(precision_score(
                    y_te, pred, average=None, zero_division=0,
                    labels=list(range(len(CATEGORIES))))[i]), 4)
                for i in range(len(CATEGORIES))},
            per_class_r={
                CATEGORIES[i]: round(float(recall_score(
                    y_te, pred, average=None, zero_division=0,
                    labels=list(range(len(CATEGORIES))))[i]), 4)
                for i in range(len(CATEGORIES))},
            n=m["n"],
            n_per_class=m["n_per_category"],
            confusion=m["confusion_matrix"]["rows_true_cols_pred"],
        ))
        print(f"  seed {sd}: acc={m['accuracy']:.4f} "
              f"macroF1={m['macro_f1']:.4f}  n={m['n']}", flush=True)

    accs = [r["acc"] for r in rows]
    f1s = [r["macro_f1"] for r in rows]
    n = rows[0]["n"]

    print("\n" + "=" * 68)
    print(f"CLASSIFIER, {len(seeds)} seeds, n={n} held-out templates")
    print("=" * 68)
    print(f"  accuracy    mean {np.mean(accs)*100:.1f}%   "
          f"range {min(accs)*100:.1f}-{max(accs)*100:.1f}%   "
          f"sd {np.std(accs)*100:.2f}")
    print(f"  macro-F1    mean {np.mean(f1s)*100:.1f}%   "
          f"range {min(f1s)*100:.1f}-{max(f1s)*100:.1f}%   "
          f"sd {np.std(f1s)*100:.2f}")

    print(f"\n  per class (seed {rows[0]['seed']}), n per class shown:")
    print(f"  {'class':<12}{'n':>4}{'F1':>8}{'prec':>8}{'recall':>8}")
    for c in CATEGORIES:
        print(f"  {c:<12}{rows[0]['n_per_class'][c]:>4}"
              f"{rows[0]['per_class_f1'][c]*100:>7.1f}%"
              f"{rows[0]['per_class_p'][c]*100:>7.1f}%"
              f"{rows[0]['per_class_r'][c]*100:>7.1f}%")

    print(f"\n  confusion matrix (rows = true, cols = predicted):")
    print(f"  {'':<12}" + "".join(f"{c[:7]:>9}" for c in CATEGORIES))
    for i, c in enumerate(CATEGORIES):
        print(f"  {c:<12}" + "".join(f"{v:>9}" for v in rows[0]["confusion"][i]))

    allsat = all(a == 1.0 for a in accs)
    print("\n" + "-" * 68)
    if allsat:
        print("VERDICT: saturated across every seed.")
        print("  A perfect score here means the four categories are lexically")
        print("  distinct, NOT that the classifier is unusually strong. The")
        print("  honest claim is 'the text-classification stage works end to")
        print("  end on held-out templates'. It is NOT evidence about the OCR,")
        print("  which is the hard part and is measured separately in")
        print("  docs/03_RESULTS.md.")
        print(f"  n={n} is also too small to support any accuracy claim on its")
        print("  own -- at n=24 the 95% interval on 100% runs from about")
        print(f"  {85.7:.1f}% to 100%.")
    else:
        print("VERDICT: not saturated; variation across seeds is informative.")

    out = Path("/home/ubuntu/Projects/sightline/artifacts/step3_classifier_metrics.json")
    out.write_text(json.dumps({
        "seeds": seeds,
        "per_seed": rows,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_sd": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_sd": float(np.std(f1s)),
        "n_test": n,
        "saturated": allsat,
        "caveat": ("A perfect score on n=24 indicates a saturated task, not an "
                   "unusually strong model. Do not present as 100% accuracy."),
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()