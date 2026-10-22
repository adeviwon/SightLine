# SightLine

> **A medicine label, spoken in Cantonese. Nothing leaves the phone.**

An offline-first **medicine-label** reader for elderly people who cannot see the
label clearly **and** cannot read the English printed on it. Two barriers, one
gesture: photograph the box, and SightLine reads the drug name, strength and dose
and says it back in Cantonese. No account, no internet, no upload.

> **Honest state of the build.** Cantonese output is the *next* build — the speech
> layer and the translation lexicon are specified and costed in
> [`docs/26_IDEA_REFINED.md`](docs/26_IDEA_REFINED.md). What ships today reads
> printed English text and speaks English, and **every measurement in this repo was
> taken on that build.** We are not claiming a Cantonese result we have not measured.

Built for the **Imperial College London Hackathon (Hong Kong & Macau)**.

**Try it now — no install, no account:** open
<https://adeviwon.github.io/SightLine/> on a phone, add to Home Screen. It
installs, then works in airplane mode. That URL is the demo; the mic drop is
turning the network off and scanning anyway.

> **AI agent taking over?** Read [`AGENTS.md`](AGENTS.md) first — verified
> state, standing rules, known traps, and ranked open work with done-criteria.

---

## The result, first

Measured by driving the **shipped app** in **WebKit** (the engine iOS Safari
uses) over **12 synthetic rendered documents** — 3 types × 4 capture profiles,
**one record per document** — on the **current** capture-profile fixtures.
Reproduce it: `bash run.sh bench-pipeline && bash run.sh bench-hackathon`.
The **previous** fixture generation — rendered before the capture profiles were
re-tuned — produced the **45.9%** this table used to headline; those rows are
**not directly comparable** to the current ones and every one is labelled below.
A controlled A/B separates the two effects: **+1.7 points is the benchmark
getting more realistic, +9.5 is the retrain** ([`docs/14`](docs/14_HACKATHON_BENCHMARK.md)).

| metric | shipped now (current fixtures) | previous fixture generation | source |
|---|---|---|---|
| **word accuracy** (normalised token recall — see below) | **57.2%** | 45.9% (3.6% at the first bench) | `artifacts/pipeline_raw_shipped.json`; previous gen. `pipeline_raw_deskew.json` / `pipeline_raw.json` |
| **CER** | **0.217** | 0.303 (0.708 at the first bench) | same |
| **ground-truth fields present in the read text** | **14 of 24** | 9 of 24 (0 of 24 at the first bench) | same (stricter extracted-field metric: **8 of 24**, was 5 of 24) |
| **document classification (app, end to end)** | **91.7%** | 83.3% (33.3% at the first bench) | same |
| cleanest profile (`studio_clean`) | **60.1%**, 100% classified correctly | 52.3% (10.3% at the first bench) | same |
| the recogniser alone, 300 paired held-out crops | **57.0% exact line match, CER 0.156** | 3.3% (pre-retrain, same 300 crops) | `app/tools/arm_ab.js` for the 57.0%; the before-value is the project's paired measurement, not a tracked file |
| the recogniser alone, shipped **retrain3**, 1,500 paired held-out crops (the render corpus) | **68.1% exact, CER 0.150** | 27.7% / 0.201 (pre-retrain, same crops) | `bash run.sh py ml/src/score_checkpoints.py`, commit `348cb6d` — real rectified photos 25.2% / 0.474 (was 14.8% / 0.523), SROIE scans 74.1% / 0.098 (was 73.4% / 0.101) |
| document classifier, held-out text, 5 seeds (n=24) | **95.8% accuracy, 95.7% macro-F1** (±2.6) | a stored 100% from one lucky seed | `artifacts/step3_classifier_metrics.json` |

**Read the metric, not just the number.** `word_accuracy` above is a
**normalised token recall** — the fraction of ground-truth words present in the
read text after punctuation and case are stripped — **not** exact-word match.
Both metric definitions live in `ml/src/bench_hackathon.py`; the browser side
computes no metrics on purpose, so the two cannot drift apart.

**The trajectory is the result — and it is labelled by fixture generation.**
`3.6% → 22.4% → 17.1% → 35.0% → 45.9%` on the same 12 documents, **all measured
on the previous fixture generation**: four train/serve contract defects we found
in our own code and fixed (the served width, inert-looking right-padding, an
image transform applied at serve but never at train, and a deskew correction
applied backwards), plus one retrain step that **regressed** and is kept in the
record. On the **current** fixtures the same build reads 47.7%, the shipped
retrain reads **57.2%**, and the A/B attributes **+1.7 points to the benchmark
becoming realistic and +9.5 to the retrain** — the two must never be quoted as one
model gain. Raw records for every step are in `artifacts/`; the full table, the
A/B and the fixture-freshness gate are in
[`docs/14_HACKATHON_BENCHMARK.md`](docs/14_HACKATHON_BENCHMARK.md).

**And what it still fails at**, because a judge should hear it from us first:

- **Real hand-photographed pages.** The line detector is the ceiling: **67.9%
  detection recall, 1.8% line-exact, 56.5% CER** on 275 lines from 20
  hand-photographed receipts ([`docs/03_RESULTS.md`](docs/03_RESULTS.md) §3).
  Three of those 20 pages are resolution-bound — 12–17 px per line against a
  32 px recogniser input — and no crop rule or model can repair them.
- **Night and heavy-handheld captures.** In our own harness
  (`bash run.sh eval --engine=crnn`, `artifacts/eval_results.json`) `low_light`
  and `handheld_heavy` are the two weakest profiles: **2.2%** and **2.0%** word
  accuracy on the raw arm. They are reported, not excluded.
- **Chinese.** The recogniser's alphabet is `0-9 A-Z` and space — **37 symbols,
  zero CJK characters**. A 繁體中文 medicine label cannot be read at all today.
  This is the most serious gap in the project and it is a data problem, not an
  architecture one.
- **The target.** The brief asks for ≥95% end-to-end text accuracy. We are at
  **57.2%** on **12 synthetic rendered documents** (current fixtures) and 1.8%
  line-exact on real photographs. That gap is the honest headline, and it is
  quantified rather than hidden.

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
   │ 2. READ      our own CRNN+CTC (ONNX, WASM, offline)     │
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
| **CRNN + CTC** (the recogniser) | PyTorch → ONNX | **942,166** | Reads text lines; 37-symbol alphabet (`0-9 A-Z` + space); dynamic width, `[1,1,32,width]` | Phone CPU (WASM) |
| **SightLineNet** | PyTorch → ONNX | 37,793 | Restores blurry/noisy/shadowed photos before OCR | Phone CPU (WASM) |
| **MiniLM-L6-v2 + head** | sentence-transformers → ONNX | 22.7M frozen + 49,796 trained | Classifies the document (banking / medical / legal / general) | Phone CPU (WASM) |

All three are trained from real code in this repository, exported to ONNX, and
**numerically verified against their PyTorch originals** before shipping.

**Install size, as a fact rather than a boast:** models + tokenizer =
**26.4 MB** (`bash run.sh bundle-check` prints 27,735,952 B), and the ONNX
runtime's single non-JSEP wasm adds **10.73 MiB**
(`ort-wasm-simd-threaded.wasm`, 11,246,032 B) — **≈37.2 MiB total** on the
phone, paid once. Nothing else is fetched at any point.

See [`docs/03_RESULTS.md`](docs/03_RESULTS.md) for every measured number and
[`docs/14_HACKATHON_BENCHMARK.md`](docs/14_HACKATHON_BENCHMARK.md) for the
end-to-end benchmark and its caveats.

---

## The honesty policy

SightLine reports what it can actually do, including where it fails.

- Every accuracy number in this repo comes from a **seeded, reproducible run**.
  Re-run it yourself.
- A **held-out test set** is used for the headline number, never the set used
  to pick the model.
- Capture conditions a **human cannot read** are labelled `SUB-HUMAN` and
  excluded from the headline — reported separately, not quietly dropped.
- When confidence is low, SightLine **says so and asks for a retake** instead
  of inventing a dosage. A wrong number spoken aloud is worse than no number.
- Numbers we got wrong are kept visible: the stored 100% classifier score (one
  lucky seed) and every failed fix are recorded in
  [`docs/12_HACKATHON_SUBMISSION.md`](docs/12_HACKATHON_SUBMISSION.md) §6 and
  [`docs/18_CRNN_ACCURACY_STRATEGIES.md`](docs/18_CRNN_ACCURACY_STRATEGIES.md).

If a number in this repository cannot be reproduced with the documented
command, that is a bug — please open an issue.

---

## Start here

| I want to… | Read |
|---|---|
| **Hackathon submission** (the 5 required steps, start here) | [`docs/12_HACKATHON_SUBMISSION.md`](docs/12_HACKATHON_SUBMISSION.md) |
| **Show a judge** (start here — one document, the whole project) | [`docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md`](docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md) |
| **Round 1 deliverables** (150-word proposal, video shot list, countdown) | [`docs/13_ROUND1_SUBMISSION_KIT.md`](docs/13_ROUND1_SUBMISSION_KIT.md) |
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

## Repository layout

```
sightline/
├── app/                    # the phone app (PWA — runs on iOS + Android)
│   ├── index.html          #   UI
│   ├── js/                 #   pipeline, ONNX inference, fallbacks
│   ├── sw.js               #   service worker — the offline guarantee
│   ├── models/             #   exported ONNX graphs (committed)
│   └── vendor/             #   ORT wasm + pdf.js (vendored, no OCR library)
├── ml/
│   ├── src/                # corpus.py, capture.py, ocr_model.py,
│   │                       # train_ocr.py, bench_hackathon.py, export_onnx.py
│   └── tests/              # pytest suite
├── models/ocr/crnn_epoch29_final.pt  # THE SHIPPED recogniser's source (942,166 params, epoch 29)
├── models/ocr/crnn.pt      # live training target — overwritten by whichever run is active; NOT the shipped source
├── artifacts/              # raw benchmark records + training metrics
├── docs/                   # everything above
└── run.sh                  # every command you need
```

---

## Quick start

```bash
# 1. Install (creates a venv, installs PyTorch CPU + deps)
bash setup_env.sh

# 2. Train the classifier + restorer and run every gate
#    (the OCR recogniser is a separate, long CPU run: `bash run.sh train-ocr 40`,
#     ~2.8 h on 4 cores, no GPU)
bash run.sh all

# 3. Run the app
bash run.sh serve          # then open the printed URL on your phone
```

Full instructions: [`docs/04_RUNNING_ON_IOS_AND_ANDROID.md`](docs/04_RUNNING_ON_IOS_AND_ANDROID.md).

---

## Platform note

The neural path needs **iOS 16.4+**: `ort-wasm-simd-threaded.wasm` requires
WASM SIMD, which Safari shipped in 16.4 (March 2023). Older iOS is detected at
startup and told why, instead of failing mid-scan. Android Chrome is not
affected. Verified in **WebKit** (the engine iOS Safari uses) *and* headless
Chromium — but **not yet on physical iPhone/Android hardware**, which is a
known gap, not a claim.
