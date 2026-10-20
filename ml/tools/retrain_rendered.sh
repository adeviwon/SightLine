#!/usr/bin/env bash
# Retrain the recogniser on the DOCUMENTS WE ACTUALLY SERVE.
#
# WHY. The shipped CRNN was trained on ICDAR SROIE receipt word crops and reads
# 4.4% of held-out crops on our own rendered documents exactly (CER 0.53), because
# it learned receipt glyph shapes and digit-ises our letters: PRESCRIPTION ->
# "PRE5CR1P710N", DATE01102026 -> "0A7E01102026". Training on our own fonts,
# layouts and capture profiles is the fix; `ml/src/render_corpus.py` builds that
# corpus from the same renderer the product's documents come from.
#
# WHAT IT DOES, in order:
#   1. backs up the current checkpoint (training overwrites models/ocr/crnn.pt
#      whenever validation improves)
#   2. waits for any in-flight corpus build to finish, then packs the corpus into
#      the trainer's cache schema
#   3. trains on BOTH sources — real receipts (SROIE) and our rendered documents —
#      so the model keeps its robustness on photographs and gains our domain
#
# RUN IT IN TMUX. This takes hours on 4 CPU cores, and a foreground/background
# Hermes process is killed well before that:
#
#   tmux new-session -d -s retrain 'bash ml/tools/retrain_rendered.sh 30'
#   tmux attach -t retrain            # watch it
#   tail -f /tmp/retrain_render.log   # or just the log
#
# AFTERWARDS, in this order (each number must be compared to its `before`):
#   bash run.sh eval-render --split test    # crop-level on our own documents
#   bash run.sh eval-ocr                    # held-out SROIE + real photos
#   bash run.sh bench-pipeline && bash run.sh bench-hackathon   # end to end
# and only then `bash run.sh export` + deploy, so the app ships a measured model.
set -euo pipefail
cd "$(dirname "$0")/../.."

EPOCHS="${1:-30}"
RENDER_CORPUS="data/render_corpus"
CACHE="$RENDER_CORPUS/crops_train.npz"
CKPT="models/ocr/crnn.pt"
BACKUP="models/ocr/crnn_before_render.pt"
LOG="/tmp/retrain_render.log"

if [ ! -f "$CKPT" ]; then
  echo "[retrain] no checkpoint at $CKPT — nothing to back up, nothing to improve on" >&2
  exit 1
fi
if [ ! -f "$BACKUP" ]; then
  cp "$CKPT" "$BACKUP"
  echo "[retrain] backed up the current checkpoint -> $BACKUP"
else
  echo "[retrain] backup already exists ($BACKUP) — leaving it alone"
fi

echo "[retrain] waiting for any in-flight corpus build"
while pgrep -f "render_corpus.py --pages" >/dev/null 2>&1; do sleep 30; done
if [ ! -f "$RENDER_CORPUS/crops.npz" ]; then
  echo "[retrain] no corpus yet; building a small one first"
  bash run.sh render-corpus --pages 500
fi
bash run.sh render-corpus --cache-only

if [ ! -f "artifacts/ocr_train.npz" ]; then
  echo "[retrain] missing artifacts/ocr_train.npz — run: bash run.sh build-data" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1   # without this `| tee` held every epoch line for 70 min
echo "[retrain] training $EPOCHS epochs on SROIE + our rendered documents"
echo "[retrain] log: $LOG"
bash run.sh py ml/src/train_ocr.py \
  --data "artifacts/ocr_train.npz,$CACHE" \
  --epochs "$EPOCHS" --threads 4 2>&1 | tee "$LOG"

echo
echo "[retrain] done. Score it against the 'before' numbers:"
echo "  before: crop-level on our documents 4.4% exact / CER 0.53 (eval-render)"
echo "  bash run.sh eval-render --split test && bash run.sh bench-pipeline && bash run.sh bench-hackathon"
echo "  the backup checkpoint is $BACKUP if this model is worse"
