# Retraining the models

Everything in this repository is reproducible from source. This page is the
full procedure, including the parts that go wrong.

---

## 0. Environment

```bash
bash setup_env.sh
```

Creates `.venv` (Python 3.12) and installs PyTorch CPU, OpenCV, ONNX Runtime,
scikit-learn, Pillow, pytesseract, pytest, and sentence-transformers.

**Every command must go through `bash run.sh <cmd>` or `./py <script>`.**

> **Why?** Some agent/CI environments export a `PYTHONPATH` pointing at a
> different site-packages tree. That shadows the project venv and binary
> wheels fail with confusing errors like
> `ImportError: cannot import name '_imaging' from 'PIL'`.
> `run.sh` strips `PYTHONPATH`, `VIRTUAL_ENV` and `PYTHONHOME`, so the
> interpreter always sees its own packages. If you hit that error, you
> bypassed `run.sh`.

Verify:

```bash
bash run.sh test        # should collect tests without ImportError
```

---

## 1. Full rebuild (everything from scratch)

```bash
bash run.sh all
```

Runs, in order: corpus stats → classifier training + 5-fold CV → restorer
training → end-to-end evaluation → ONNX export + parity → test suite.

Wall clock on 4 CPU threads: **~40 minutes**, dominated by restorer training.

---

## 2. Individual steps

### The classifier

```bash
bash run.sh train-clf 60                  # 60 epochs, 5-fold CV
bash run.sh train-clf 60 --resume         # continue from checkpoint
bash run.sh train-clf 60 --epochs-free    # see below
```

| Flag | Default | Notes |
|---|---|---|
| `--epochs` | 60 | total, not incremental |
| `--batch` | 32 | |
| `--lr` | 2e-3 | head-only; the encoder is frozen |
| `--seed` | 123 | |
| `--cv` | 5 | folds per category (20 total) |
| `--resume` | off | load `models/classifier/minilm_head.pt` |

Takes ~40 seconds including cross-validation. Writes
`models/classifier/minilm_head.pt` and `minilm_head_eval.json`.

### The restorer

```bash
bash run.sh train-dncnn 40
bash run.sh train-dncnn 40 --docs-per-profile 20    # bigger corpus, slower
bash run.sh train-dncnn 40 --resume                 # continue
```

| Flag | Default | Notes |
|---|---|---|
| `--epochs` | 30 | total, not incremental |
| `--docs-per-profile` | 14 | 14 docs × 10 profiles × 3 bands = 420 patches |
| `--batch` | 16 | |
| `--lr` | 1e-3 | |
| `--resume` | off | |

Training data is cached at `artifacts/restorer_data.npz`. Delete it to
regenerate. Takes ~35 minutes for 40 epochs.

### Evaluation

```bash
bash run.sh eval
bash run.sh eval --n 12              # more documents per profile
bash run.sh eval --arms restorer,restorer_clahe   # subset of arms
```

Writes `artifacts/eval_results.json`. The slowest stage is OCR — it shells out
to Tesseract once per (profile × document × arm).

### ONNX export

```bash
bash run.sh export
bash run.sh export --verify-only     # re-check existing artifacts
```

Exits non-zero if any parity check fails. **Do not ship artifacts from a run
that exited non-zero.**

### Tests

```bash
bash run.sh test
bash run.sh test -k leakage          # just the anti-leakage suite
bash run.sh test -v                  # verbose
```

---

## 3. The network size trade-off

`ml/src/bench.py` measures this on 4 CPU threads with 336 patches:

```bash
bash run.sh bench
```

| Config | Params | s/epoch | 40 epochs |
|---|---|---|---|
| `ch=48 depth=8` | 126,097 | 124 | 83 min |
| `ch=32 depth=6` **(default)** | 37,793 | 52 | 35 min |
| `ch=24 depth=6` | 22,753 | 29 | 19 min |

The default is the largest that trains inside a hackathon night while staying
small enough for a phone. If you have more time, pass a bigger config:

```python
# ml/src/model.py — SightLineNet.__init__
def __init__(self, channels=48, depth=8):
```

Then retrain and re-export. Check `models/restorer/restorer_training_log.json`
for the resulting val PSNR before deciding whether the extra time was worth it.

---

## 4. Diagnosing a bad result

### "Validation PSNR is flat"

The restorer is underfit or the task is too hard.

```bash
bash run.sh bench                       # confirm you are not CPU-starved
bash run.sh train-dncnn 40 --docs-per-profile 25   # more data
```

Also confirm the checkpoint is actually loading: `eval` warns
`no checkpoint; restorer arms run at identity` when `models/restorer/restorer.pt`
is missing, and an identity network produces exactly the flat curve you see.

### "Classifier accuracy dropped"

Almost always a corpus problem. Read the failures:

```bash
bash run.sh errors
```

It prints each misclassified validation sample with true label, predicted
label, and confidence margin. Then:

- **Systematically wrong on one category** → that category's templates are too
  few or too similar. Add templates to `ml/src/corpus.py`.
- **Confused between two related categories** → the categories genuinely
  overlap. Make the templates more distinctive, or accept the confusion and
  note it honestly in the docs.
- **Correct but low-confidence** → the decision boundary is fine; the
  calibration is not. Lower the retake threshold, or add training data.

### "ONNX parity failed"

```bash
bash run.sh export --verify-only
```

It reports which artifact diverged and by how much. Usual causes:

- **opset too new for `onnxruntime-web`.** Lower `--opset` (try 13 or 14).
- **A non-tensor op** snuck in. Check with `python -c "import onnx; onnx.load(...)"`.
- **Stale artifact.** Delete `models/onnx/` and re-export.

A parity failure is a **release blocker**, not a warning: an ONNX graph that
loads but computes the wrong thing will confidently misread a dosage.

### "Everything is slow"

```bash
nproc                                   # how many cores
python -c "import torch; print(torch.get_num_threads())"
```

On a 2-core box, halve `--docs-per-profile` rather than raising epochs. Cut
epochs before data — an under-trained model on a big corpus is useless, but a
well-trained model on a small corpus is merely weak.

---

## 5. Making the training faster

In rough order of impact:

1. **Use `--docs-per-profile` 8 instead of 14.** Halves data generation and
   per-epoch time. Slightly weaker model.
2. **Fewer epochs.** The cosine schedule means 25 epochs reaches most of the
   benefit of 40.
3. **Shrink the network** to `ch=24 depth=6` in `model.py`.
4. **Skip cross-validation** with `bash run.sh train-clf 60 --cv 0`.
5. **A machine with more cores.** `torch.set_num_threads` scales nearly linearly
   to 4–8 threads.

---

## 6. Adding your own document types

To add a fifth category (say, `utility` bills):

1. Add ~45 templates to `ml/src/corpus.py` under a new `UTILITY` list.
2. Add `"utility"` to `CATEGORIES`.
3. Add utility ground-truth fields to `FIELDS` in `ml/src/evaluate.py` if you
   want field-level scoring on it.
4. Add keyword-density weights in `classify_doc()` in `evaluate.py`, and mirror
   them in `app/js/pipeline.js`.
5. `bash run.sh train-clf 60 && bash run.sh eval && bash run.sh export`
6. The head's output layer grows automatically — `ClassifierHead` derives
   `n_classes` from the data path.

Re-run the test suite: several tests assert that `FIELDS` keys match the
document types in `capture.py`, and that no category is empty.
