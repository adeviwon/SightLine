#!/usr/bin/env bash
# SightLine — one-shot environment setup (paste-and-go)
set -u
cd "$(dirname "$0")"
VENV=".venv"
PY="$VENV/bin/python"

echo "=== [1/3] creating venv (py3.12) ==="
if [ ! -x "$PY" ]; then
  uv venv --python 3.12 "$VENV" || python3 -m venv "$VENV"
fi

echo "=== [2/3] installing deps ==="
if [ -x "$(command -v uv)" ]; then
  uv pip install --python "$PY" \
    "torch>=2.2" --torch-backend=cpu
  uv pip install --python "$PY" \
    numpy Pillow opencv-python-headless onnx onnxruntime onnxscript \
    scikit-learn pytest pytesseract
else
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  "$PY" -m pip install numpy Pillow opencv-python-headless onnx \
    onnxruntime onnxscript scikit-learn pytest pytesseract
fi

echo "=== [3/3] verifying ==="
"$PY" - <<'PYEOF'
mods = ["torch", "numpy", "cv2", "onnxruntime", "sklearn", "PIL", "pytest"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"  OK  {m:14s} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  FAIL {m:14s} {e}")
PYEOF
echo "=== tesseract ==="
tesseract --version 2>&1 | head -2 || echo "tesseract missing (apt install tesseract-ocr)"
echo "=== setup done ==="
