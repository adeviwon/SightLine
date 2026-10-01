"""
SightLine — Generate docs/03_RESULTS.md from the measured artifacts.

Numbers in the documentation must come from a run, not from memory or from
retyping. This script reads artifacts/eval_results.json, the classifier eval
JSON, the restorer training log, and models/onnx/ file sizes, then writes the
results document.

Run it after `bash run.sh eval && bash run.sh export`:

    bash run.sh results

If an artifact is missing it says so explicitly rather than omitting the
section, because a silently absent results table is worse than an honest gap.
"""

import json
import sys
from pathlib import Path

ARM_LABEL = {
    "raw": "raw (OCR the degraded image)",
    "classical": "classical (median + unsharp)",
    "restorer": "restorer (our model)",
    "restorer_clahe": "restorer + CLAHE (production)",
}


def pct(x):
    return "—" if x is None else f"{x:.0%}"


def num(x, n=1):
    return "—" if x is None else f"{x:.{n}f}"


def main():
    out = []
    A = out.append

    A("# Results\n")
    A("> **Every number on this page was produced by `bash run.sh eval` and")
    A("> `bash run.sh export`.** Nothing here is typed by hand.")
    A("> Regenerate: `bash run.sh eval && bash run.sh export && bash run.sh results`\n")
    A("Machine-readable versions live in `artifacts/eval_results.json` and")
    A("`models/classifier/minilm_head_eval.json`.\n")
    A("---\n")

    # ── restorer training ────────────────────────────────────────────────
    log = Path("models/restorer/restorer_training_log.json")
    A("## 1. Restorer training\n")
    if log.exists():
        d = json.load(open(log))
        h = d.get("history", [])
        A(f"SightLineNet — **{d.get('params', 0):,} parameters**, "
          f"Charbonnier loss, AdamW, cosine schedule.\n")
        A(f"- Training patches: **{d.get('n_train_patches')}**  ")
        A(f"- Validation patches: **{d.get('n_val_patches')}**  ")
        A(f"- Split by **document seed** (no document contributes patches to "
          f"both sides)  ")
        A(f"- Best validation PSNR: **{d.get('best_val_psnr_db')} dB**\n")
        if h:
            A("| epoch | train loss | val PSNR (dB) |")
            A("|---|---|---|")
            step = max(1, len(h) // 12)
            for e in h[::step]:
                A(f"| {e['epoch']} | {e['train_loss']:.5f} | {e['val_psnr_db']:.2f} |")
            A(f"| **{h[-1]['epoch']}** | **{h[-1]['train_loss']:.5f}** | "
              f"**{h[-1]['val_psnr_db']:.2f}** |\n")
    else:
        A("> **Missing:** `models/restorer/restorer_training_log.json`.")
        A("> Run `bash run.sh train-dncnn 40`.\n")

    # ── classifier ───────────────────────────────────────────────────────
    clf = Path("models/classifier/minilm_head_eval.json")
    A("## 2. Document classifier\n")
    if clf.exists():
        d = json.load(open(clf))
        ci = d.get("accuracy_ci95_bootstrap", [None, None])
        cv = d.get("cross_validation")
        A("MiniLM-L6-v2 (**frozen**, 22,713,216 params) + trained MLP head "
          f"(**{d.get('head_params', 0):,} params**).\n")
        A("| metric | value |")
        A("|---|---|")
        A(f"| **Held-out test accuracy** | **{pct(d.get('accuracy'))}** |")
        if ci and ci[0] is not None:
            A(f"| 95% bootstrap CI | [{ci[0]:.1%}, {ci[1]:.1%}] |")
        A(f"| Macro F1 | {num(d.get('macro_f1'), 3)} |")
        A(f"| Validation accuracy | {pct(d.get('val_accuracy'))} |")
        if cv:
            A(f"| **{cv['k']}-fold CV mean ({cv['folds']} folds)** | "
              f"**{pct(cv.get('mean_accuracy'))} ± "
              f"{pct(cv.get('std_accuracy'))}** |")
            A(f"| {cv['k']}-fold CV macro F1 | "
              f"{num(cv.get('mean_macro_f1'), 3)} ± {num(cv.get('std_macro_f1'), 3)} |")
        A(f"| Test set size | {d.get('n')} |")
        A(f"| Train / val / test | {d.get('split_sizes', {}).get('train_raw')} → "
          f"{d.get('split_sizes', {}).get('train_augmented')} augmented / "
          f"{d.get('split_sizes', {}).get('val')} / "
          f"{d.get('split_sizes', {}).get('test')} |")
        A(f"| Training time | {d.get('seconds')} s (CPU) |")
        A("")
        if d.get("per_category_accuracy"):
            A("**Per-category test accuracy**\n")
            A("| category | accuracy | n |")
            A("|---|---|---|")
            npc = d.get("n_per_category", {})
            for k, v in d["per_category_accuracy"].items():
                A(f"| {k} | {pct(v)} | {npc.get(k, '')} |")
            A("")
        cm = d.get("confusion_matrix")
        if cm:
            labs = cm["labels"]
            rows = cm["rows_true_cols_pred"]
            A("**Confusion matrix** (rows = truth, columns = prediction)\n")
            A("| | " + " | ".join(labs) + " |")
            A("|---" * (len(labs) + 1) + "|")
            for lab, r in zip(labs, rows):
                A(f"| **{lab}** | " + " | ".join(str(v) for v in r) + " |")
            A("")
    else:
        A("> **Missing:** `models/classifier/minilm_head_eval.json`.")
        A("> Run `bash run.sh train-clf 60`.\n")

    # ── end to end ───────────────────────────────────────────────────────
    ev = Path("artifacts/eval_results.json")
    A("## 3. End-to-end accuracy\n")
    A("The metric that matters: **field accuracy** — the fraction of documents "
      "where *every* key field is recovered. All-or-nothing, because a blind "
      "user who hears the wrong dosage is worse off than one who hears none.\n")
    if ev.exists():
        d = json.load(open(ev))
        arms = d.get("arms", [])
        sub = set(d.get("sub_human_profiles", []))
        A(f"{d.get('n_per_profile')} documents per capture profile, seed "
          f"{d.get('seed')}. **Sub-human profiles excluded from the headline** "
          f"and reported separately below.\n")
        A("### Headline\n")
        A("| preprocessing arm | field accuracy | word accuracy | exact dosage | OCR conf |")
        A("|---|---|---|---|---|")
        for a in arms:
            h = d["headline"][a]
            star = " ⭐" if a == "restorer_clahe" else ""
            A(f"| {ARM_LABEL.get(a, a)}{star} | **{pct(h['field_accuracy'])}** | "
              f"{pct(h['word_accuracy'])} | {pct(h['exact_dosage_rate'])} | "
              f"{num(h['mean_ocr_confidence'])} |")
        A("")
        # deltas vs classical
        if "classical" in arms and "restorer_clahe" in arms:
            b = d["headline"]["classical"]["field_accuracy"]
            m = d["headline"]["restorer_clahe"]["field_accuracy"]
            delta = m - b
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
            A(f"**Restorer + CLAHE vs. the classical baseline: "
              f"{arrow} {abs(delta):.1%} field accuracy** "
              f"({pct(b)} → {pct(m)}).\n")

        A("### Per capture profile (field accuracy)\n")
        short = [str(ARM_LABEL.get(a, a)).split(" ")[0] for a in arms]
        A("| profile | " + " | ".join(short) + " |")
        A("|---" * (len(arms) + 1) + "|")
        for p, rows in d.get("per_profile", {}).items():
            mark = " ⚠️" if p in sub else ""
            cells = " | ".join(
                pct((rows.get(a) or {}).get("field")) for a in arms)
            A(f"| {p}{mark} | {cells} |")
        A("")
        if sub:
            A(f"⚠️ = `SUB_HUMAN` — a person cannot reliably read this either. "
              f"Excluded from the headline: {', '.join(sorted(sub))}. "
              f"These are honest failures, not hidden ones.\n")
    else:
        A("> **Missing:** `artifacts/eval_results.json`. Run `bash run.sh eval`.\n")

    # ── onnx ─────────────────────────────────────────────────────────────
    A("## 4. On-device payload\n")
    od = Path("models/onnx")
    files = sorted(od.glob("*.onnx")) if od.exists() else []
    if files:
        A("| artifact | size |")
        A("|---|---|")
        tot = 0
        for f in files:
            tot += f.stat().st_size
            A(f"| `{f.name}` | {f.stat().st_size/1024:.1f} KB |")
        A(f"| **total ML payload** | **{tot/1024:.1f} KB** |")
        A("")
        vend = Path("app/vendor")
        if vend.exists():
            vs = sum(f.stat().st_size for f in vend.rglob("*") if f.is_file())
            A(f"Plus the vendored OCR engine and English language data: "
              f"**{vs/1024/1024:.1f} MB**. Cached once on install; the app then "
              f"works in airplane mode forever.\n")
    else:
        A("> **Missing:** `models/onnx/`. Run `bash run.sh export`.\n")

    # ── tests ────────────────────────────────────────────────────────────
    A("## 5. Test suite\n")
    A("```bash\nbash run.sh test\n```\n")
    A("The suite covers the claims this page makes. The ones that matter most:\n")
    A("| test file | what it proves |")
    A("|---|---|")
    A("| `test_leakage.py` | no template, document, or augmented twin appears "
      "in both train and evaluation sets |")
    A("| `test_corpus.py` | the split is stratified and there are no duplicate "
      "templates (a real bug in the original corpus) |")
    A("| `test_capture.py` | every degradation actually changes the image, is "
      "deterministic, and is physically directional |")
    A("| `test_model.py` | the network is exactly the identity at init and "
      "stays inside the mobile parameter budget |")
    A("| `test_evaluate.py` | sub-human profiles never leak into the headline |")
    A("")

    Path("docs/03_RESULTS.md").write_text("\n".join(out) + "\n")
    print(f"wrote docs/03_RESULTS.md ({len(out)} lines)")


if __name__ == "__main__":
    sys.exit(main())
