# Troubleshooting

## A. Environment

### `ImportError: cannot import name '_imaging' from 'PIL'`

The project venv is being shadowed by another Python environment's
`PYTHONPATH`.

```bash
bash run.sh test          # instead of .venv/bin/python -m pytest
```

`run.sh` runs the interpreter with `PYTHONPATH`, `VIRTUAL_ENV` and
`PYTHONHOME` stripped. Every documented command goes through it.

If you must call Python directly:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV -u PYTHONHOME ./.venv/bin/python your_script.py
```

### `No module named 'torch'` after setup

`setup_env.sh` installs the CPU wheel. If a system Python is being picked up,
your shell has an active virtualenv that overrides `.venv`. Deactivate it, or
use `run.sh`.

### `uv: command not found`

`setup_env.sh` falls back to `python3 -m venv` and `pip`. It will be slower
(the PyTorch CPU wheel is large) but works.

### Torch install is slow or stalls

The CPU wheel is ~200 MB. On a slow connection, install it separately:

```bash
curl -O https://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp312-cp312-linux_x86_64.whl
./.venv/bin/python -m pip install ./torch-2.2.0+cpu-cp312-cp312-linux_x86_64.whl
```

---

## B. Training

### Validation PSNR is flat around 11–13 dB

Check whether a checkpoint exists at all:

```bash
ls -la models/restorer/restorer.pt
```

`eval` prints `WARNING no checkpoint; restorer arms run at identity` when it is
missing — an untrained (identity) network produces exactly this flat curve.

If the checkpoint exists and training is genuinely flat, it is underfit:

```bash
bash run.sh train-dncnn 60 --docs-per-profile 25
```

### Training is very slow

```bash
bash run.sh bench     # measures your actual s/epoch
```

On 4 CPU threads expect ~52 s/epoch at the default network size. 40 epochs is
~35 minutes. To go faster, cut data before cutting epochs:

```bash
bash run.sh train-dncnn 25 --docs-per-profile 8
```

### Classifier accuracy below the previous run

Read the failures before changing anything:

```bash
bash run.sh errors
```

The output distinguishes systematic category errors from low-confidence
corrects — those need different fixes (data vs. calibration).

---

## C. Evaluation

### `tesseract: command not found`

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng    # Debian/Ubuntu
brew install tesseract                               # macOS
```

Or `scoop install tesseract` on Windows.

Verify: `tesseract --version`.

### Evaluation is extremely slow

OCR shells out to Tesseract once per (profile × document × arm) — 10 × 6 × 4 =
240 invocations. Each takes 0.3–2 s.

```bash
bash run.sh eval --n 3                    # fewer documents
bash run.sh eval --arms raw,restorer      # fewer arms
```

### `field_accuracy` is 0 everywhere

Tesseract is not running, or the fixture paths are wrong. Check:

```bash
tesseract --version
bash run.sh capture      # writes sample images to /tmp
```

If the arms all report 0, the problem is upstream of the model — OCR is
returning nothing. Open one degraded sample and run OCR on it manually.

---

## D. ONNX export

### Parity check fails

**A release blocker.** An ONNX graph that loads but computes the wrong thing
will confidently misread dosages.

```bash
bash run.sh export --verify-only     # reports which artifact and by how much
```

| Cause | Fix |
|---|---|
| opset too new for `onnxruntime-web` | `bash run.sh export --opset 14` |
| stale artifacts | `rm -rf models/onnx && bash run.sh export` |
| unsupported op | `python -c "import onnx; print(onnx.load('models/onnx/restorer.onnx'))"` |

### `torch.onnx.export` deprecation warning

Expected on torch 2.x. The export still works. To silence it, pass
`dynamo=False` explicitly on newer torch versions.

---

## E. The app

See [`04_RUNNING_ON_IOS_AND_ANDROID.md`](04_RUNNING_ON_IOS_AND_ANDROID.md#7-troubleshooting)
for device-side issues. The two most common:

| Symptom | Fix |
|---|---|
| Camera will not open | Needs HTTPS or `localhost`/LAN IP; grant permission in Settings |
| Works online, fails offline | Service worker did not precache — check the console for `cache.addAll` failures |

---

## F. Still stuck

1. `bash run.sh test` — the suite often names the broken invariant directly.
2. `bash run.sh errors` — prints the actual misclassified samples.
3. `bash run.sh bench` — confirms you are not CPU-starved.
4. Read `docs/07_EVALUATION_METHODOLOGY.md` — several "bugs" are actually
   intentional honesty guards.
