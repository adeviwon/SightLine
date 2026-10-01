#!/usr/bin/env bash
# Install sentence-transformers + transformers into the project venv.
set -u
cd "$(dirname "$0")"
PY=".venv/bin/python"
uv pip install --python "$PY" sentence-transformers transformers 2>&1 | tail -6
echo "=== verify ==="
"$PY" -c "
import sentence_transformers, transformers, torch
print('sentence_transformers', sentence_transformers.__version__)
print('transformers', transformers.__version__)
print('torch', torch.__version__)
"
