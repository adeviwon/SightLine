"""
Error analysis for the SightLine document classifier.

Prints every validation sample the model gets wrong, with the true label, the
predicted label, and the confidence margin. This is the file to read when
accuracy is below target: it tells you whether the model is wrong because
the CORPUS is ambiguous (fix the data) or because the HEAD is underfit
(fix the training), instead of guessing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import CATEGORIES, make_corpus  # noqa: E402
from train_classifier import ENCODER_NAME, ClassifierHead  # noqa: E402


def main(ckpt="models/classifier/minilm_head.pt", seed=123):
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    saved = torch.load(ckpt, map_location="cpu", weights_only=True)
    head = ClassifierHead()
    head.load_state_dict(saved["head"])
    head.eval()

    _, _, va_t, va_l = make_corpus(seed=seed)
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    st = SentenceTransformer(ENCODER_NAME)
    st.eval()
    enc = tok(va_t, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        h = st[0].auto_model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    emb = F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), p=2, dim=1)

    with torch.no_grad():
        probs = F.softmax(head(emb), dim=1)
    pred = probs.argmax(1).numpy()
    yt = np.asarray(va_l)
    correct = (pred == yt)

    print(f"accuracy {correct.mean():.1%}  (n={len(yt)})\n")
    print("=" * 78)
    print("MISCLASSIFIED")
    print("=" * 78)
    for i in np.where(~correct)[0]:
        p = probs[i].numpy()
        order = np.argsort(-p)
        margin = float(p[order[0]] - p[order[1]])
        print(f"\n  true={CATEGORIES[yt[i]]:<8} pred={CATEGORIES[pred[i]]:<8} "
              f"margin={margin:.3f} conf={p[pred[i]]:.2f}")
        print(f"  top3: " + ", ".join(
            f"{CATEGORIES[j]}={p[j]:.2f}" for j in order[:3]))
        print(f"  text: {va_t[i][:150]}")

    print("\n" + "=" * 78)
    print("LOW-CONFIDENCE CORRECT (margin < 0.25 — fragile, likely to flip)")
    print("=" * 78)
    for i in np.where(correct)[0]:
        p = probs[i].numpy()
        order = np.argsort(-p)
        margin = float(p[order[0]] - p[order[1]])
        if margin < 0.25:
            print(f"  margin={margin:.3f} true={CATEGORIES[yt[i]]:<8} "
                  f"conf={p[pred[i]]:.2f}")
            print(f"  text: {va_t[i][:130]}")

    frac = float((correct & (probs.max(1).values.numpy() < 0.6)).sum())
    print(f"\ncorrect-but-low-confidence(<0.6): {frac}/{len(yt)} = {frac/len(yt):.1%}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["models/classifier/minilm_head.pt"]))
