#!/usr/bin/env bash
# SightLine — run any project python entrypoint in the correct interpreter.
#
# WHY THIS EXISTS: the Hermes agent process exports PYTHONPATH pointing at its
# own site-packages, which shadows the project venv and breaks binary wheels
# (PIL fails with "cannot import name '_imaging'"). Every script here MUST be
# launched via ./py so it is reproducible for a human reader and for CI.
#
# USAGE:  ./py ml/src/capture.py
#         ./py -m pytest ml/tests -q
cd "$(dirname "$0")" || exit 1
exec env -u PYTHONPATH -u VIRTUAL_ENV -u PYTHONHOME \
  ./.venv/bin/python "$@"
