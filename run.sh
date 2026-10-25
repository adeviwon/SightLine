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
  test-ui)     exec node app/tools/ui_redesign_check.cjs "$@" ;;
  test-bilingual) exec node --test app/tools/bilingual_contract.test.cjs ;;
  test-field-localization) exec node --test app/tools/field_localization.test.cjs ;;
  test-medicine-sections) exec node --test app/tools/medicine_sections.test.cjs ;;
  test-medicine-ui) exec node app/tools/medicine_sections_visual.cjs ;;
  test-bilingual-ui) exec node app/tools/bilingual_history_check.cjs ;;
  test-history) exec node app/tools/history_persistence_check.cjs ;;
  test-native-history) exec bash app/tools/native_history_check.sh ;;
  test-speech) exec node --test app/tools/speech_contract.test.cjs ;;
  test-speech-ui) SPEECH_PHASE=after exec node app/tools/speech_ui_check.cjs ;;
  test-receipts) exec node --test app/tools/receipt_regression.test.mjs ;;
  test-offline-receipt) exec node app/tools/receipt_offline_check.cjs ;;
  test-native-offline)
    build_dir=$(mktemp -d /tmp/sightline-native-audit.XXXXXX)
    swiftc ios/Sources/OfflineAssets.swift ios/Sources/WebAssetServer.swift app/tools/native_offline/main.swift -o "$build_dir/audit" || exit 1
    exec "$build_dir/audit" "$PWD/app" "$PWD/artifacts/aria-merge/sroie-X00016469612.jpg" "$PWD/artifacts/aria-merge/native-webkit-offline.json" ;;
  test-native-resume)
    build_dir=$(mktemp -d /tmp/sightline-native-resume.XXXXXX)
    swiftc ios/Sources/OfflineAssets.swift ios/Sources/WebAssetServer.swift app/tools/native_offline/main.swift -o "$build_dir/audit" || exit 1
    exec "$build_dir/audit" "$PWD/app" "$PWD/artifacts/aria-merge/sroie-X00016469612.jpg" "$PWD/artifacts/bilingual-history/native-resume.json" --restart-server ;;
  test-reader) exec node --test app/tools/ppocr_contract.test.mjs "$@" ;;
  test-reader-loading) exec node app/tools/reader_loading_check.cjs "$@" ;;
  bench-reader) exec node app/tools/ppocr_native.mjs "$@" ;;
  # ── real-data text recogniser (replaces Tesseract) ──────────────────────
  fetch-data) exec bash fetch_data.sh "$@";;
  # iOS gate: parse-time Safari compatibility scan of the shipped JS, then run
  # the whole app in WebKit (the engine iOS Safari uses), including real
  # crnn.onnx inference. Needs the Playwright WebKit browser on disk.
  ios-check)   node app/tools/ios_compat_scan.js || exit 1
               exec node app/tools/ios_webkit_check.js "$@";;
  # Full battery: the real controller (file input -> runScan) plus the failure
  # paths, in WebKit and Chromium. ~12 min for both engines.
  ios-full)    exec node app/tools/ios_full_check.js "$@";;
  build-data)  exec $PYRUN ml/src/build_dataset.py "$@" ;;
  train-ocr)   ep="${1:-40}"; shift 2>/dev/null || true
               exec $PYRUN ml/src/train_ocr.py --epochs "$ep" "$@" ;;
  eval-ocr)    exec $PYRUN ml/src/eval_ocr.py "$@" ;;
  # The Imperial brief's step 3 ("test and evaluate"), scored the way it asks:
  # bench-pipeline drives the shipped app in WebKit and writes RAW records;
  # bench-hackathon reads them plus the training artifacts and reports every
  # required metric with its source. See docs/14_HACKATHON_BENCHMARK.md.
  bench-pipeline) exec node app/tools/ios_full_check.js --engine=webkit --suites=bench --json=artifacts/pipeline_raw.json ;;
  bench-hackathon) exec $PYRUN ml/src/bench_hackathon.py "$@" ;;
  # The recogniser's accuracy on OUR OWN documents (crop-level, detector excluded),
  # and the corpus it is measured on. See ml/src/render_corpus.py.
  render-corpus) exec $PYRUN ml/src/render_corpus.py "$@" ;;
  eval-render) exec $PYRUN ml/src/eval_render_corpus.py "$@" ;;
  # Lane A/E tooling: the char-LM + CTC decoders, and the run-vs-run judge that
  # every accuracy experiment is scored by (IMPROVED / REGRESSED / INCONCLUSIVE).
  build-lm)    exec $PYRUN ml/src/build_lm.py "$@" ;;
  decode-ab)   exec $PYRUN ml/src/eval_decode_ab.py "$@" ;;
  # The word-level LM: build/export the shipped trie, sweep the decoder's
  # hyper-parameters on held-out crops, and prove the BROWSER decoder returns the
  # same string as the Python one on the recogniser's own logits.
  word-lm)     exec $PYRUN ml/src/word_lm.py "$@" ;;
  decode-sweep) exec $PYRUN ml/src/eval_decode_sweep.py "$@" ;;
  word-lm-vectors) exec $PYRUN ml/src/gen_wordlm_vectors.py "$@" ;;
  word-lm-parity) exec node app/tools/word_lm_parity.js "$@" ;;
  compare)     exec $PYRUN ml/src/compare_runs.py "$@" ;;
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
  *) echo "usage: bash run.sh <target> [args]"; echo "  train/test: train-clf train-dncnn train-ocr eval eval-ocr test test-reader test-ui"; echo "  measure:    bench bench-reader bench-pipeline bench-hackathon gating results bundle-check repro-check ios-check ios-full"; echo "  data/other: capture corpus build-data export diagnose privacy errors py serve shell"; exit 1 ;;
esac
