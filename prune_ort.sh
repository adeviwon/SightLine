#!/usr/bin/env bash
# Trim the vendored onnxruntime-web to the minimum a browser actually loads.
#
# The full dist/ is 89 MB: webgpu, webgl, node, jsep, and bundle variants.
# A PWA service worker must precache every byte, so shipping all of it makes
# first load absurd and bloat the repo for no benefit.
#
# What SightLine needs:
#   ort.min.js          the loader (UMD, sets window.ort)
#   ort-wasm-simd-threaded.wasm   the CPU backend
#   ort-wasm-simd-threaded.mjs    its loader shim
#
# Everything else (webgpu/webgl/jsep/node/bundle) is dead weight: the app runs
# CPUExecutionProvider on WASM threads only, which works on every iOS 16+ and
# Android 8+ device we target.
set -u
cd "$(dirname "$0")" || exit 1
D="app/vendor/ort"

echo "=== before ==="
du -sh "$D" 2>/dev/null

KEEP="ort.min.js ort-wasm-simd-threaded.wasm ort-wasm-simd-threaded.mjs"
for f in $KEEP; do
  if [ ! -f "$D/$f" ]; then
    echo "  ERROR: required file missing: $f"; exit 1
  fi
done

# Remove everything not in KEEP, reporting what goes.
removed=0
for f in "$D"/*; do
  b=$(basename "$f")
  case " $KEEP " in
    *" $b "*) continue ;;
  esac
  if [ -f "$f" ]; then
    sz=$(du -h "$f" | cut -f1)
    rm -f "$f" && removed=$((removed+1))
  fi
done

echo "  removed $removed unused files"
echo
echo "=== after ==="
du -sh "$D"
ls -la "$D" | tail -n +4
echo
echo "kept:"
for f in $KEEP; do
  printf "  %-36s %8.1f KB\n" "$f" "$(echo "scale=1; $(stat -c%s "$D/$f")/1024" | bc)"
done
echo
echo "First-load payload now: $(du -sh "$D" | cut -f1) for ONNX runtime"
echo "(plus app/models/*.onnx for the model weights)"
echo
echo "Next: add <script src=\"vendor/ort/ort.min.js\"></script> to app/index.html"
echo "and list these three files in the app/sw.js precache array."
