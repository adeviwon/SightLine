# SightLine documentation

Read these in order if you are new. Jump straight to what you need if you are
not.

---

## The two entry points

**If you want to win a hackathon with this:**
→ [`00_HACKATHON_PLAYBOOK.md`](00_HACKATHON_PLAYBOOK.md)

Includes the 60-second pitch, the build schedule, the slide deck, the
questions judges ask, and what to cut if you fall behind.

**If you want to understand or extend the code:**
→ [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md) → [`02_ML_PIPELINE.md`](02_ML_PIPELINE.md)

---

## Full index

| # | Document | What it answers |
|---|---|---|
| 00 | [Hackathon playbook](00_HACKATHON_PLAYBOOK.md) | How do I win with this? Pitch, schedule, slides, Q&A prep |
| 01 | [Architecture](01_ARCHITECTURE.md) | How does a photo become speech? Why is offline architectural? |
| 02 | [ML pipeline](02_ML_PIPELINE.md) | What are the two models, why those architectures, how were they trained? |
| 03 | [Results](03_RESULTS.md) | What accuracy do we actually get? *(auto-generated from artifacts)* |
| 04 | [Running on iOS and Android](04_RUNNING_ON_IOS_AND_ANDROID.md) | How do I install it on a phone? Why a PWA and not native? |
| 05 | [Demo runbook](05_DEMO_RUNBOOK.md) | How do I demo this in 4 minutes without it falling over? |
| 06 | [Retraining](06_RETRAINING.md) | How do I retrain, tune, debug, or extend the models? |
| 07 | [Evaluation methodology](07_EVALUATION_METHODOLOGY.md) | How do I verify your accuracy claims? Why should I believe them? |
| 08 | [Troubleshooting](08_TROUBLESHOOTING.md) | Something is broken. What? |
| 09 | [Interpreting the numbers](09_INTERPRETING_THE_NUMBERS.md) | Why is the accuracy 92–100% and not just "100%"? |
| 10 | [The accuracy gap](10_ACCURACY_GAP.md) | **What we did not achieve, measured honestly, and how to close it** |
| — | [Privacy](PRIVACY.md) | How is the offline guarantee enforced, and how do I verify it? |

---

## Command reference

Every command goes through `run.sh` (see [`06_RETRAINING.md`](06_RETRAINING.md#0-environment)
for why this matters):

```bash
bash setup_env.sh                       # one-time: venv + dependencies

bash run.sh all                         # train everything, evaluate, export, test
bash run.sh corpus                      # corpus statistics
bash run.sh capture                     # render sample capture-profile images
bash run.sh train-clf 60                # train the MiniLM classifier (+5-fold CV)
bash run.sh train-dncnn 40              # train the restorer
bash run.sh eval                        # end-to-end accuracy harness
bash run.sh export                      # ONNX export + numerical parity check
bash run.sh results                     # regenerate docs/03_RESULTS.md
bash run.sh test                        # pytest suite
bash run.sh errors                      # per-sample classifier failures
bash run.sh diagnose                    # restoration headroom analysis
bash run.sh bench                       # training-speed benchmark
bash run.sh serve [port]                # serve the app for a phone
```

---

## Where the numbers come from

| Claim | Source of truth | Regenerate with |
|---|---|---|
| Classifier accuracy / F1 / confusion matrix | `models/classifier/minilm_head_eval.json` | `bash run.sh train-clf 60` |
| End-to-end field accuracy | `artifacts/eval_results.json` | `bash run.sh eval` |
| Restorer training curve | `models/restorer/restorer_training_log.json` | `bash run.sh train-dncnn 40` |
| ONNX payload sizes | `models/onnx/` | `bash run.sh export` |
| Results document | `docs/03_RESULTS.md` | `bash run.sh results` |

No number in this documentation is typed by hand. The results page is
generated from the JSON artifacts by `ml/src/make_results_doc.py`.

---

## Three principles this project holds to

1. **The offline guarantee is verifiable.** Not a claim — turn on airplane
   mode, or read `app/sw.js`. See [`01_ARCHITECTURE.md § 2`](01_ARCHITECTURE.md#2-why-offline-is-an-architectural-requirement-not-a-feature).

2. **Accuracy claims carry their uncertainty.** Held-out test sets, bootstrap
   confidence intervals, cross-validation. See
   [`07_EVALUATION_METHODOLOGY.md`](07_EVALUATION_METHODOLOGY.md).

3. **Failing safely beats succeeding wrongly.** When the model cannot read a
   document, SightLine says so and asks for a retake. It never guesses a
   dosage. This is a UX decision and a safety property, and it is tested.
