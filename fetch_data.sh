#!/usr/bin/env bash
# Fetch the real-document corpora for the text recogniser.
#
# Two REAL datasets, because training an OCR model on synthetic renders trains
# it on an imaging process that does not exist. See ml/src/realdata.py for why.
#
#   1. ICDAR2019-SROIE (Huang et al., ICDAR 2019 RRC)
#      626 train + 347 test real scanned receipts with per-WORD boxes AND
#      transcriptions. ~485 MB via the jsdnrs/ICDAR2019-SROIE HF mirror.
#
#   2. Express Expense (Auad et al., SIBGRAPI 2024), Zenodo 13688441
#      198 HAND-PHOTOGRAPHED restaurant receipts + 4-corner homographies.
#      CC-BY-4.0, 18 MB. This is the real-life capture case: shot by hand, at
#      night, dark background, plastic sleeve, rotated, unevenly lit, glare.
#
# Neither corpus is committed: they are large and carry their own licences.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p data/sroie data/handheld

# Strip leaked interpreter env, same as run.sh does.
unset PYTHONPATH VIRTUAL_ENV PYTHONHOME 2>/dev/null || true
PY=./.venv/bin/python

echo "== 1/3  ICDAR2019-SROIE (real scanned receipts) =="
if compgen -G "data/sroie/data/train-*.parquet" > /dev/null; then
  echo "   already present, skipping"
else
  "$PY" - <<'PYEOF'
from huggingface_hub import hf_hub_download
for f in ("data/train-00000-of-00001.parquet",
          "data/test-00000-of-00001.parquet"):
    p = hf_hub_download("jsdnrs/ICDAR2019-SROIE", f,
                        repo_type="dataset", local_dir="data/sroie")
    print("   got", f)
PYEOF
fi

echo "== 2/3  Express Expense (hand-photographed receipts, CC-BY-4.0) =="
if compgen -G "data/handheld/Labels/*.txt" > /dev/null; then
  echo "   already present, skipping"
else
  curl -fsSL -o data/handheld/Dataset.zip \
    "https://zenodo.org/api/records/13688441/files/Dataset.zip/content"
  ( cd data/handheld && unzip -q -o Dataset.zip )
  echo "   extracted"
fi

echo "== 3/3  verify =="
"$PY" - <<'PYEOF'
import glob, sys
tr = glob.glob("data/sroie/data/train-*.parquet")
te = glob.glob("data/sroie/data/test-*.parquet")
jpg = glob.glob("data/handheld/*-receipt.jpg")
lbl = glob.glob("data/handheld/Labels/*.txt")
print(f"   SROIE train parquets : {len(tr)}")
print(f"   SROIE test  parquets : {len(te)}")
print(f"   handheld photos      : {len(jpg)}")
print(f"   handheld labels      : {len(lbl)}")
ok = bool(tr) and bool(te) and bool(jpg)
if not ok:
    print("ERROR: a corpus is missing; see messages above", file=sys.stderr)
    sys.exit(1)
print("   all corpora present")
PYEOF

echo
echo "Next:  bash run.sh build-data --receipts 260"