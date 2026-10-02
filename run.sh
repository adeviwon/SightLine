#!/usr/bin/env bash
# SightLine — task runner. Executes the project's python entrypoints under the
# project venv with Hermes' PYTHONPATH stripped.
#
#   bash run.sh capture          -> ml/src/capture.py
#   bash run.sh train-clf 40     -> train the MiniLM classifier (40 epochs)
#   bash run.sh train-denoise 20 -> train the denoiser (20 epochs)
#   bash run.sh eval             -> full evaluation gate
#   bash run.sh test             -> pytest suite
#   bash run.sh export           -> ONNX export + parity check
#   bash run.sh all              -> train-clf + train-denoise + eval + export
cd "$(dirname "$0")" || exit 1
chmod +x py 2>/dev/null
PYRUN="env -u PYTHONPATH -u VIRTUAL_ENV -u PYTHONHOME ./.venv/bin/python"
mkdir -p models ml/models docs artifacts

cmd="${1:-help}"; shift 2>/dev/null || true

case "$cmd" in
  capture)     exec $PYRUN ml/src/capture.py "$@" ;;
  corpus)      exec $PYRUN ml/src/corpus.py "$@" ;;
  bench)       exec $PYRUN ml/src/bench.py "$@" ;;
  bundle-check) exec $PYRUN ml/src/bundle_check.py "$@" ;;
  repro-check) exec $PYRUN ml/src/repro_check.py "$@" ;;
  diagnose)    exec $PYRUN ml/src/diagnose.py "$@" ;;
  gating)      exec $PYRUN ml/src/gating.py "$@" ;;
  privacy)     exec $PYRUN ml/src/privacy_audit.py "$@" ;;
  results)     exec $PYRUN ml/src/make_results_doc.py "$@" ;;
  errors)      exec $PYRUN ml/src/error_analysis.py "$@" ;;
  train-clf)   ep="${1:-60}"; shift 2>/dev/null || true
               exec $PYRUN ml/src/train_classifier.py --epochs "$ep" --cv 5 "$@" ;;
  train-dncnn) ep="${1:-30}"; shift 2>/dev/null || true
               exec $PYRUN ml/src/train_restorer.py --epochs "$ep" "$@" ;;
  eval)        exec $PYRUN ml/src/evaluate.py "$@" ;;
  export)      exec $PYRUN ml/src/export_onnx.py "$@" ;;
  test)        exec $PYRUN -m pytest ml/tests -q "$@" ;;
  # ── real-data text recogniser (replaces Tesseract) ──────────────────────
  fetch-data) exec bash fetch_data.sh "$@" ;;
  build-data)  exec $PYRUN ml/src/build_dataset.py "$@" ;;
  train-ocr)   ep="${1:-40}"; shift 2>/dev/null || true
               exec $PYRUN ml/src/train_ocr.py --epochs "$ep" "$@" ;;
  eval-ocr)    exec $PYRUN ml/src/eval_ocr.py "$@" ;;
  # Generic passthrough so any module can be run through the venv with
  # PYTHONPATH stripped: bash run.sh py ml/src/realdata.py --selftest
  py)          exec $PYRUN "$@" ;;
  serve)       port="${1:-8080}"
               echo "Serving ./app on http://0.0.0.0:${port}"
               ip=$(hostname -I 2>/dev/null | awk '{print $1}')
               echo "  phone on the SAME wifi: http://${ip:-<your-ip>}:${port}"
               echo "  iOS:    Safari -> Share -> Add to Home Screen"
               echo "  Android: Chrome -> menu -> Install app"
               exec $PYRUN -m http.server "$port" --directory app --bind 0.0.0.0 ;;
  shell)       exec env -u PYTHONPATH -u VIRTUAL_ENV ./.venv/bin/python -i "$@" ;;
  all)
    set -e
    bash run.sh corpus
    bash run.sh train-clf 40
    bash run.sh train-dncnn 20
    bash run.sh eval
    bash run.sh export
    bash run.sh test
    ;;
  *) echo "usage: bash run.sh {capture|corpus|train-clf|train-dncnn|eval|export|test|all}"; exit 1 ;;
esac
