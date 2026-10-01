#!/usr/bin/env bash
# SightLine — seed-robustness check for the classifier.
#
# A single seed's 100% test score proves very little: the split changes with
# the seed, so a different seed tests different documents. If accuracy holds
# across seeds, the model is not memorising one particular split.
#
# Usage:  bash seed_check.sh [n_seeds]
cd "$(dirname "$0")" || exit 1
N="${1:-5}"
for s in $(seq 1 "$N"); do
  seed=$((120 + s))
  echo "──────── seed $seed ────────"
  env -u PYTHONPATH -u VIRTUAL_ENV -u PYTHONHOME \
    ./.venv/bin/python ml/src/train_classifier.py \
      --epochs 60 --seed "$seed" --cv 0 \
      --out "artifacts/seedcheck_$seed" 2>&1 \
    | grep -E '"accuracy"|"macro_f1"|accuracy_ci95' | head -4
done
echo
echo "──────── summary ────────"
env -u PYTHONPATH -u VIRTUAL_ENV -u PYTHONHOME \
  ./.venv/bin/python - <<'PYEOF'
import json, glob, statistics as st
accs, f1s, cis = [], [], []
for f in sorted(glob.glob("artifacts/seedcheck_*/minilm_head_eval.json")):
    d = json.load(open(f))
    accs.append(d["accuracy"]); f1s.append(d["macro_f1"])
    ci = d.get("accuracy_ci95_bootstrap")
    if ci: cis.append(ci)

summary = {
    "n_seeds": len(accs),
    "mean_accuracy": round(st.mean(accs), 4) if accs else None,
    "min_accuracy": round(min(accs), 4) if accs else None,
    "max_accuracy": round(max(accs), 4) if accs else None,
    "std_accuracy": round(st.pstdev(accs), 4) if accs else None,
    "mean_macro_f1": round(st.mean(f1s), 4) if f1s else None,
    "per_seed_accuracy": accs,
    "note": ("Each seed resplits train/val/test, so this measures split "
             "sensitivity, not statistical independence over the same docs."),
}
json.dump(summary, open("artifacts/seed_summary.json", "w"), indent=2)

print(f"seeds evaluated : {len(accs)}")
print(f"test accuracy   : mean {st.mean(accs):.1%}  min {min(accs):.1%}  max {max(accs):.1%}")
print(f"                  stdev {st.pstdev(accs):.1%}")
print(f"macro F1        : mean {st.mean(f1s):.3f}  min {min(f1s):.3f}")
if cis:
    print(f"mean 95% CI     : [{st.mean(c[0] for c in cis):.1%}, {st.mean(c[1] for c in cis):.1%}]")
print("wrote artifacts/seed_summary.json")
PYEOF
