#!/usr/bin/env bash
# Vendor onnxruntime-web into app/vendor/ort/ and the trained models into
# app/models/.
#
# WHY: without onnxruntime-web the app silently falls back to the classical
# median+unsharp stack and the trained restorer contributes nothing. Measured
# cost of this being absent: 42% end-to-end field accuracy instead of whatever
# the trained model actually achieves. See docs/10_ACCURACY_GAP.md § 5.
#
# This script is idempotent. It prefers a local npm cache, falls back to the
# unpkg CDN, and verifies every file landed before writing the manifest.
set -u
cd "$(dirname "$0")" || exit 1

ORT_VERSION="${ORT_VERSION:-1.20.1}"
ORT_DEST="app/vendor/ort"
mkdir -p "$ORT_DEST" app/models

echo "=== [1/4] fetching onnxruntime-web@$ORT_VERSION ==="
if command -v npm >/dev/null 2>&1; then
  tmp=$(mktemp -d)
  ( cd "$tmp" && npm pack "onnxruntime-web@$ORT_VERSION" >/dev/null 2>&1 ) \
    && tar -xzf "$tmp"/onnxruntime-web-*.tgz -C "$tmp" 2>/dev/null \
    && cp -r "$tmp"/package/dist/* "$ORT_DEST"/ 2>/dev/null \
    && echo "  from npm" || echo "  npm failed, trying CDN"
  rm -rf "$tmp"
fi

if [ ! -f "$ORT_DEST/ort.min.js" ] && [ ! -f "$ORT_DEST/ort.all.min.js" ]; then
  echo "  downloading from unpkg ..."
  BASE="https://unpkg.com/onnxruntime-web@$ORT_VERSION/dist"
  for f in ort.min.js ort.wasm.min.js ort-wasm-simd-threaded.jsep.mjs \
           ort-wasm-simd-threaded.jsep.wasm ort-wasm-simd-threaded.mjs \
           ort-wasm-simd-threaded.wasm; do
    curl -fsSL "$BASE/$f" -o "$ORT_DEST/$f" 2>/dev/null \
      && echo "    got $f" || echo "    MISS $f"
  done
fi

echo
echo "=== [2/4] copying trained models ==="
if [ -f models/onnx/restorer.onnx ]; then
  cp models/onnx/restorer.onnx app/models/ && echo "  restorer.onnx"
else
  echo "  models/onnx/restorer.onnx missing - run 'bash run.sh export' first"
fi
for f in minilm_encoder.onnx minilm_head.onnx vocab.txt tokenizer.json; do
  [ -f "models/onnx/$f" ] && cp "models/onnx/$f" app/models/ && echo "  $f"
done

echo
echo "=== [3/4] writing availability manifest ==="
if [ -f "$ORT_DEST/ort.min.js" ] || [ -f "$ORT_DEST/ort.all.min.js" ]; then
  ORT=true
else
  ORT=false
fi
cat > app/models/ort.json <<JSON
{
  "ort": $ORT,
  "ort_version": "$ORT_VERSION",
  "restorer": $([ -f app/models/restorer.onnx ] && echo true || echo false),
  "classifier": $([ -f app/models/minilm_encoder.onnx ] && echo true || echo false),
  "generated_by": "vendor_ort.sh"
}
JSON
cat app/models/ort.json

echo
echo "=== [4/4] verifying ==="
missing=0
for f in "$ORT_DEST"/*.js "$ORT_DEST"/*.wasm "$ORT_DEST"/*.mjs; do
  [ -e "$f" ] || continue
  printf "  %-46s %8.1f KB\n" "$(basename "$f")" "$(echo "scale=1; $(stat -c%s "$f")/1024" | bc)"
done
echo
echo "total ORT payload: $(du -sh "$ORT_DEST" 2>/dev/null | cut -f1)"
echo "app/models:       $(du -sh app/models 2>/dev/null | cut -f1)"
echo
if [ "$ORT" = "true" ]; then
  echo "ORT vendored. Add the ORT script tag to app/index.html and list the"
  echo "new files in the app/sw.js precache array, then re-run:"
  echo "  bash run.sh privacy && node app/tools/browser_smoke.js"
else
  echo "ORT NOT vendored - the app will use the classical fallback."
fi
exit $missing
