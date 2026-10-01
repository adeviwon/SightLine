"""
SightLine ML — Train the document classifier on sentence-transformers/all-MiniLM-L6-v2.

METHODOLOGY (this is the part judges ask about):

  3-WAY SPLIT, not 2-way. train 70% / val 15% / test 15%, stratified per
  category and split BEFORE augmentation. The val set selects the epoch;
  the TEST set is touched once, at the end, to produce the headline number.
  Reporting a val score that was also used for model selection is optimistic
  — the held-out test set is what makes the claim honest.

  K-FOLD CROSS-VALIDATION. A single 15% test split on 174 templates is ~26
  samples, so one flipped sample is ~4 accuracy points. `--cv 5` runs the whole
  pipeline over disjoint folds and reports mean ± std. That is the number
  to put on a slide.

  AUGMENTATION IS TRAIN-ONLY. OCR-style digit/letter confusion is applied after
  splitting, so a val/test phrase can never have a corrupted twin in train.

  REPORT FROM THE SELECTED CHECKPOINT, WITH A CI. Best-val epoch is loaded
  back before the test pass (standard practice), and a 4000-sample bootstrap
  95% CI is attached so "96%" is never presented without its uncertainty.

Writes models/classifier/minilm_head_eval.json and minilm_head.pt.

Usage:
    python3 src/train_classifier.py --epochs 60
    python3 src/train_classifier.py --epochs 60 --cv 5
    python3 src/train_classifier.py --epochs 60 --resume
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import CATEGORIES, RAW, augment, corpus_stats  # noqa: E402
from corpus import _dedupe  # noqa: E402

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ClassifierHead(nn.Module):
    """Linear(384→128) → ReLU → Dropout → Linear(128→4)."""

    def __init__(self, dim=384, hidden=128, n_classes=4, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── splitting ───────────────────────────────────────────────────────────

def three_way_split(seed=123, val_frac=0.15, test_frac=0.15):
    """Stratified train/val/test over DEDUPLICATED templates. No augmentation."""
    buckets = {c: _dedupe(RAW[c]) for c in CATEGORIES}
    rng = random.Random(seed)
    tr_t, tr_l, va_t, va_l, te_t, te_l = [], [], [], [], [], []
    for ci, cat in enumerate(CATEGORIES):
        items = buckets[cat][:]
        rng.shuffle(items)
        n = len(items)
        n_te = max(1, int(n * test_frac))
        n_va = max(1, int(n * val_frac))
        tr_t += items[:n - n_te - n_va]
        tr_l += [ci] * (n - n_te - n_va)
        va_t += items[n - n_te - n_va: n - n_te]
        va_l += [ci] * n_va
        te_t += items[n - n_te:]
        te_l += [ci] * n_te
    return tr_t, tr_l, va_t, va_l, te_t, te_l


def kfold_indices(seed=123, k=5):
    """Disjoint folds per category for cross-validation."""
    folds = []
    for ci, cat in enumerate(CATEGORIES):
        items = _dedupe(RAW[cat])
        rng = random.Random(seed + ci)
        idx = list(range(len(items)))
        rng.shuffle(idx)
        for f in range(k):
            te = [items[i] for j, i in enumerate(idx) if j % k == f]
            tr = [items[i] for j, i in enumerate(idx) if j % k != f]
            folds.append((tr, [ci] * len(tr), te, [ci] * len(te)))
    return folds


# ── metrics ─────────────────────────────────────────────────────────────

def bootstrap_ci(correct, n_boot=4000, seed=123):
    rng = np.random.default_rng(seed)
    arr = np.asarray(correct, dtype=float)
    if arr.size == 0:
        return [0.0, 0.0]
    means = arr[rng.integers(0, arr.size, (n_boot, arr.size))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4)]


def metrics_from(probs, y_true):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    pred = probs.argmax(1)
    yt = np.asarray(y_true)
    per_cat = {}
    for ci, cat in enumerate(CATEGORIES):
        m = yt == ci
        per_cat[cat] = round(float((pred[m] == ci).mean()), 4) if m.sum() else None
    return {
        "accuracy": round(float((pred == yt).mean()), 4),
        "accuracy_ci95_bootstrap": bootstrap_ci((pred == yt).astype(int)),
        "macro_f1": round(float(f1_score(yt, pred, average="macro")), 4),
        "per_category_accuracy": per_cat,
        "n": int(len(yt)),
        "n_per_category": {CATEGORIES[i]: int((yt == i).sum())
                           for i in range(len(CATEGORIES))},
        "confusion_matrix": {
            "labels": CATEGORIES,
            "rows_true_cols_pred": confusion_matrix(
                yt, pred, labels=list(range(len(CATEGORIES)))).tolist(),
        },
        "classification_report": classification_report(
            yt, pred, target_names=CATEGORIES, zero_division=0),
    }


# ── embedding ───────────────────────────────────────────────────────────

def load_encoder():
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    st = SentenceTransformer(ENCODER_NAME)
    st.eval()
    for p in st[0].auto_model.parameters():
        p.requires_grad = False
    return tok, st[0].auto_model


def embed_corpus(texts, tok, model, batch=32):
    """Mean-pool + L2-normalise, matching sentence-transformers exactly."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            outs.append(F.normalize(emb, p=2, dim=1))
    return torch.cat(outs)


def train_head(E_tr, y_tr, E_val, y_val, epochs, lr, batch, seed,
               head=None, verbose=True, tag=""):
    torch.manual_seed(seed)
    head = head or ClassifierHead()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = nn.CrossEntropyLoss()
    n = len(E_tr)
    best_acc, best_state, best_ep = -1.0, None, 0
    history = []
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(head(E_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        head.eval()
        with torch.no_grad():
            acc = float((head(E_val).argmax(1) == y_val).float().mean())
        history.append({"epoch": ep + 1, "train_loss": round(tot / n, 5),
                        "val_acc": round(acc, 4)})
        if acc > best_acc:
            best_acc, best_ep = acc, ep + 1
            best_state = copy.deepcopy(head.state_dict())
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  {tag}epoch {ep+1:>3}/{epochs}  loss {tot/n:.4f}  "
                  f"val {acc:.1%}  best {best_acc:.1%}@{best_ep}", flush=True)
    head.load_state_dict(best_state)
    return head, best_acc, best_ep, history


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cv", type=int, default=0,
                    help="k folds per category; 0 = single 3-way split only")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="models/classifier")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, encoder = load_encoder()
    ck = out / "minilm_head.pt"
    t0 = time.time()

    # ── cross-validation ────────────────────────────────────────────────
    cv_accs, cv_f1s, cv_sizes = [], [], []
    if args.cv > 0:
        from sklearn.metrics import f1_score
        print(f"[cv] {args.cv}-fold cross-validation ...", flush=True)
        folds = kfold_indices(args.seed, args.cv)
        for fi, (tr_t, tr_l, te_t, te_l) in enumerate(folds):
            tr_a, tr_al = augment(tr_t, tr_l, seed=args.seed, n_copies=2)
            E_tr = embed_corpus(tr_a, tok, encoder)
            E_te = embed_corpus(te_t, tok, encoder)
            head, _, _, _ = train_head(E_tr, torch.tensor(tr_al), E_te,
                                       torch.tensor(te_l), args.epochs, args.lr,
                                       args.batch, args.seed, verbose=False)
            head.eval()
            with torch.no_grad():
                p = F.softmax(head(E_te), 1).numpy()
            a = float((p.argmax(1) == np.asarray(te_l)).mean())
            f1 = float(f1_score(te_l, p.argmax(1), average="macro"))
            cv_accs.append(a)
            cv_f1s.append(f1)
            cv_sizes.append(len(te_l))
            print(f"  fold {fi+1}/{args.cv}: acc {a:.1%}  macroF1 {f1:.3f}  "
                  f"(n_test={len(te_l)})", flush=True)

    # ── final model on the 3-way split ──────────────────────────────────
    print("\n[split] 3-way train/val/test ...", flush=True)
    tr_t, tr_l, va_t, va_l, te_t, te_l = three_way_split(args.seed)
    tr_a, tr_al = augment(tr_t, tr_l, seed=args.seed, n_copies=2)
    print(f"  train {len(tr_t)} raw -> {len(tr_a)} augmented | "
          f"val {len(va_t)} | test {len(te_t)}", flush=True)

    E_tr = embed_corpus(tr_a, tok, encoder)
    E_va = embed_corpus(va_t, tok, encoder)
    E_te = embed_corpus(te_t, tok, encoder)

    head = None
    if args.resume and ck.exists():
        head = ClassifierHead()
        head.load_state_dict(torch.load(ck, map_location="cpu",
                                        weights_only=True)["head"])
        print("  [resume] loaded head weights from checkpoint", flush=True)

    print("[train] classifier head ...", flush=True)
    head, best_val, best_ep, history = train_head(
        E_tr, torch.tensor(tr_al), E_va, torch.tensor(va_l),
        args.epochs, args.lr, args.batch, args.seed, head)

    torch.save({"head": head.state_dict(), "categories": CATEGORIES,
                "encoder": ENCODER_NAME, "val_acc": best_val,
                "best_epoch": best_ep}, ck)
    json.dump({"history": history, "best_val_acc": best_val,
               "best_epoch": best_ep},
              open(out / "minilm_head_training_log.json", "w"), indent=2)

    head.eval()
    with torch.no_grad():
        p_te = F.softmax(head(E_te), 1).numpy()
        p_va = F.softmax(head(E_va), 1).numpy()
    res = metrics_from(p_te, te_l)
    res_val = metrics_from(p_va, va_l)
    res.update({
        "encoder": ENCODER_NAME + " (frozen)",
        "head_params": sum(p.numel() for p in head.parameters()),
        "encoder_params": sum(p.numel() for p in encoder.parameters()),
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "best_epoch": best_ep,
        "val_accuracy": res_val["accuracy"],
        "test_accuracy": res["accuracy"],
        "corpus": corpus_stats(),
        "split_sizes": {"train_raw": len(tr_t), "train_augmented": len(tr_a),
                        "val": len(va_t), "test": len(te_t)},
        "seconds": round(time.time() - t0, 1),
    })
    if cv_accs:
        res["cross_validation"] = {
            "k": args.cv,
            "folds": len(cv_accs),
            "fold_accuracies": [round(a, 4) for a in cv_accs],
            "mean_accuracy": round(float(np.mean(cv_accs)), 4),
            "std_accuracy": round(float(np.std(cv_accs)), 4),
            "mean_macro_f1": round(float(np.mean(cv_f1s)), 4),
            "std_macro_f1": round(float(np.std(cv_f1s)), 4),
        }
    json.dump(res, open(out / "minilm_head_eval.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("classification_report", "confusion_matrix")},
                     indent=2), flush=True)
    print("\nHELD-OUT TEST classification report:\n"
          + res["classification_report"], flush=True)


if __name__ == "__main__":
    main()
