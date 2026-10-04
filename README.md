# SightLine

> **Read documents aloud. Nothing leaves the phone.**

An offline-first document reader for blind and visually impaired people.
Point the camera at a bank statement, a prescription, or a letter — SightLine
reads it back to you in plain speech. No account, no internet, no upload.

Built for the **Imperial College London Hackathon (Hong Kong & Macau)**.

---

## Start here

| I want to… | Read |
|---|---|
| **Hackathon submission** (the 5 required steps, start here) | [`docs/12_HACKATHON_SUBMISSION.md`](docs/12_HACKATHON_SUBMISSION.md) |
| **Show a judge** (start here — one document, the whole project) | [`docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md`](docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md) |
| **Win the hackathon** (strategy and pitch) | [`docs/00_HACKATHON_PLAYBOOK.md`](docs/00_HACKATHON_PLAYBOOK.md) |
| Understand how it works | [`docs/01_ARCHITECTURE.md`](docs/01_ARCHITECTURE.md) |
| Understand the ML | [`docs/02_ML_PIPELINE.md`](docs/02_ML_PIPELINE.md) |
| See measured results | [`docs/03_RESULTS.md`](docs/03_RESULTS.md) |
| Run it on a phone | [`docs/04_RUNNING_ON_IOS_AND_ANDROID.md`](docs/04_RUNNING_ON_IOS_AND_ANDROID.md) |
| Run the demo for judges | [`docs/05_DEMO_RUNBOOK.md`](docs/05_DEMO_RUNBOOK.md) |
| Retrain the models | [`docs/06_RETRAINING.md`](docs/06_RETRAINING.md) |
| Check our claims | [`docs/07_EVALUATION_METHODOLOGY.md`](docs/07_EVALUATION_METHODOLOGY.md) |
| Fix something | [`docs/08_TROUBLESHOOTING.md`](docs/08_TROUBLESHOOTING.md) |

---

## What it does

```
   📷 camera photo  /  📄 PDF  /  🖼 gallery image
        │
        ▼
   ┌─────────────────────────────────────────────────────────┐
   │ 1. RESTORE   PyTorch SightLineNet (ONNX, on-device)     │
   │              undoes motion blur, sensor noise, shadow   │
   │              + CLAHE contrast                            │
   ├─────────────────────────────────────────────────────────┤
   │ 2. READ      Tesseract OCR (WASM, offline)              │
   ├─────────────────────────────────────────────────────────┤
   │ 3. UNDERSTAND  MiniLM classifier → document type        │
   │                + field extraction (dosages, amounts,    │
   │                  account numbers, dates, clauses)       │
   ├─────────────────────────────────────────────────────────┤
   │ 4. SPEAK     offline speech synthesis                   │
   └─────────────────────────────────────────────────────────┘
        │
        ▼
   🔊 "This is a prescription. Amoxicillin, 500 milligrams,
       one capsule three times daily for seven days."
```

**Zero network calls after install.** Not "we promise" — verifiable by
reading [`app/sw.js`](app/sw.js) and by pulling the phone's network log into
airplane mode mid-demo.

---

## The trained models

| Model | Framework | Params | What it does | Where it runs |
|---|---|---|---|---|
| **SightLineNet** | PyTorch → ONNX | 37,793 | Restores blurry/noisy/shadowed photos before OCR | Phone CPU (WASM) |
| **MiniLM-L6-v2 + head** | sentence-transformers → ONNX | 22.7M frozen + 49,796 trained | Classifies the document (banking / medical / legal / general) | Phone CPU (WASM) |

Both are trained from real code in this repository, exported to ONNX, and
**numerically verified against their PyTorch originals** before shipping.
See [`docs/03_RESULTS.md`](docs/03_RESULTS.md) for measured numbers.

---

## Accuracy: what we measured, honestly

**Read [`docs/10_ACCURACY_GAP.md`](docs/10_ACCURACY_GAP.md) before quoting any
number.**

| Requirement | Status |
|---|---|
| Trained models, ONNX-exported, int8, parity + inference verified | ✅ done — `bash run.sh bundle-check` |
| Runs offline on iOS and Android | ✅ verified in headless Chromium ⚠️ *not on real iOS/Android hardware* |
| Zero network calls after install | ✅ verified by static audit + source |
| Reproducible evaluation | ✅ fixed — `bash run.sh repro-check` |
| Document classification ≥ 95% | ✅ **95.0% mean** across 5 seeds (min 91.7%, CI 86.7–100%), macro F1 0.947 |
| **End-to-end text accuracy on blurry images ≥ 95%** | ❌ **62.2% best arm** — reproducible, and short of target |

The end-to-end number is the real one, and it is not close to 95%. On a clean
scan the app recovers every field. On a realistic handheld capture it usually
does not: `handheld_light`, `handheld_heavy` and `low_light` score **0% on
every preprocessing arm**. A typical failure is one wrong digit —
`2024-CV-00456` read as `2024-CV-00458` — which is why the app **omits a field
it cannot read** rather than speaking a wrong one.

Where the restorer genuinely helps, measured per profile:

| Profile | raw | restorer | |
|---|---|---|---|
| hand_shadow | 20% | **80%** | the model rescues shadowed captures (this IS the +4.4) |
| jpeg_social | 100% | 80% | restoration *hurts* compressed-but-legible text |
| everything else | — | — | tie |

That is a +4.4 point headline built from one profile. Three things are known
and fixable:

1. **Restore on optical blur only, not high-frequency loss generally.** The
   current gate treats JPEG compression like blur (both collapse
   variance-of-Laplacian) and over-restores text that OCR reads fine
   un-restored. This is why `jpeg_social` regresses.
2. The restorer is selected on **PSNR**, when it should be selected on **field
   accuracy** — it scores −55.76 dB on undamaged input and +5.81 dB on badly
   damaged input, and PSNR averages those into a number describing neither.
3. `off_axis` and `jpeg_social` failures are partly geometric (skew, layout),
   and no amount of denoising fixes those.

---

## Repository layout

```
sightline/
├── app/                    # the phone app (PWA — runs on iOS + Android)
│   ├── index.html          #   UI
│   ├── js/                 #   pipeline, ONNX inference, fallbacks
│   ├── sw.js               #   service worker — the offline guarantee
│   └── vendor/             #   Tesseract WASM + language data (vendored)
├── ml/
│   ├── src/
│   │   ├── corpus.py       # document corpus + OCR-noise augmentation
│   │   ├── capture.py      # simulated phone-capture degradations
│   │   ├── model.py        # SightLineNet architecture
│   │   ├── train_classifier.py   # MiniLM fine-tune (3-way split + k-fold CV)
│   │   ├── train_restorer.py     # restorer training
│   │   ├── evaluate.py     # end-to-end accuracy harness
│   │   └── export_onnx.py  # ONNX export + parity verification
│   ├── models/             # trained weights (committed)
│   └── tests/              # pytest suite
├── docs/                   # everything above
└── run.sh                  # every command you need
```

---

## Quick start

```bash
# 1. Install (creates a venv, installs PyTorch CPU + deps)
bash setup_env.sh

# 2. Train both models from scratch (~2 minutes on a laptop CPU)
bash run.sh all

# 3. Run the app
bash run.sh serve          # then open the printed URL on your phone
```

Full instructions: [`docs/04_RUNNING_ON_IOS_AND_ANDROID.md`](docs/04_RUNNING_ON_IOS_AND_ANDROID.md).

---

## The honesty policy

SightLine reports what it can actually do, including where it fails.

- Every accuracy number in this repo comes from a **seeded, reproducible run**
  of `ml/src/evaluate.py`. Re-run it yourself.
- A **held-out test set** is used for the headline number, never the set used
  to pick the model.
- Capture conditions a **human cannot read** are labelled `SUB-HUMAN` and
  excluded from the headline — reported separately, not quietly dropped.
- When confidence is low, SightLine **says so and asks for a retake** instead
  of inventing a dosage. A wrong number spoken aloud is worse than no number.

If a number in this repository cannot be reproduced with the documented
command, that is a bug — please open an issue.
