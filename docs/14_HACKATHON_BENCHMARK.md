# 14 — The Imperial College London Hackathon benchmark

The brief's step 3 ("Test and evaluate") asks for named metrics on held-out data.
This document is the record of those measurements, the exact command behind each
one, and what the numbers do **not** prove. Every figure here is produced by a
command in this repo, not typed in by hand — a judge can re-run the three lines at
the bottom and get the same output.

## The required metrics, and where each comes from

| brief asks for | metric | value | command / artifact |
|---|---|---|---|
| classifier, held-out | accuracy | **95.8%** (±2.6) | `bash run.sh train-clf` → `artifacts/step3_classifier_metrics.json` |
| classifier, held-out | macro-F1 | **95.7%** (±2.7) | same run — 5 seeds, n=24 test documents |
| OCR, held-out crops | word accuracy | **28.0%** | `bash run.sh train-ocr` → `artifacts/ocr_training.json` (7,875 held-out crops) |
| OCR, held-out real scans | word accuracy **29.70%**, CER **11.75%** | 361 SROIE receipts, 17,945 words | `bash run.sh eval-ocr` section A |
| OCR, 198 hand-photographed receipts | line accuracy **1.96%**, CER **63.46%** | 60 pages, 868 detected lines | `bash run.sh eval-ocr` section B — the honest hard case |
| the shipped app, end to end | field recovery, classification, word accuracy, CER | **table below** | `bash run.sh bench-pipeline` → `artifacts/pipeline_raw.json` |
| one documented failure | a real misread | `ACCOUNT NUMBER 40218877` → `ICOUNT NUMBER 4021887` | `docs/03_RESULTS.md` §3 |

## The shipped app, end to end (the number that decides whether it works)

`bash run.sh bench-pipeline` drives the **real app** in WebKit — iOS Safari's
engine — over 12 rendered documents (3 types × 4 capture profiles) and records
what came out. `bash run.sh bench-hackathon` scores those records:

| profile | n | field recovery | classification | word accuracy | CER |
|---|---|---|---|---|---|
| studio_clean | 3 | 33.3% | 100.0% | 61.1% | 16.8% |
| handheld_light | 3 | 0.0% | 0.0% | 3.5% | 73.9% |
| off_axis | 3 | 0.0% | 33.3% | 8.7% | 85.4% |
| jpeg_social | 3 | 16.7% | 66.7% | 16.1% | 54.0% |
| **all samples** | **12** | **12.5%** | **50.0%** | **22.4%** | **57.5%** |

Raw records: `artifacts/pipeline_raw_final.json`. Row-level evidence in
`artifacts/hackathon_metrics.json`.

### These numbers used to read 0.0% / 33.3% / 3.6% / 70.8%, and the cause was one constant

The app's ONNX graph was exported at a FIXED 256px while `build_dataset.crop_to_array`
builds every training crop at its NATURAL width clamped to `MAX_W = 320` — and the
corpus median IS that clamp (only 40 of 3,270 held-out crops are <= 256px). So the
app fed the model the first 256px of nearly every line. On 200 held-out crops,
crop-level exact match was **44.7% at natural width against 0.0% cut to 256**, and
the read changed on **200/200**: `PATIENT JANE DOE DOB 04021979` came back as
`PERIENT JANE DOE 2O08010`, `DATE 01102026` as `DATE 01020`, `PRESCRIBER DR A AHMED
MBBS` as `PRESCRIBER DR A AHMNT`.

Every symptom previously attributed to the model — `STATEMENI OT ACCOUN`, "read
tokens 19-54 against 73-90 ground-truth words", "the CRNN digit-ises our letters" —
was the tail never reaching it. The end-to-end effect of the fix, same 12
documents, same metric, is the table above: word accuracy **3.6% -> 22.4%**, mean
CER **70.8% -> 57.5%**, read tokens **468 -> 680 (+45%)**, and on the clean profile
**10.3% -> 61.1%** word accuracy with classification **66.7% -> 100%** (which also
confirms the document classifier was being fed the truncation, not failing on its
own — MiniLM measures 95.8% macro-F1 on real text).

The graph is now exported at the training clamp, and `export_crnn`/`verify_crnn`
default their width to `realdata.MAX_W` so the number cannot drift from the data
again. `ml/tests/test_graph_width_contract.py` asserts EQUALITY with the clamp; the
old assertion was `width <= MAX_W`, which any narrower graph satisfies — it passed
hardest on the worst value, which is how 256 shipped.

Words are scored with `evaluate.word_accuracy` (fraction of ground-truth words
present in the read text, after `norm()`), and CER as (substitutions + insertions
+ deletions) / ground-truth characters on normalised text. Both definitions live in
`ml/src/bench_hackathon.py` and are unit-tested in `ml/tests/test_bench_hackathon.py`
— the browser side deliberately computes no metrics, so the two cannot drift apart.

## What these numbers do NOT prove

- **The recogniser never trained on these renders**; the classifier's templates
  are in its training corpus, so its end-to-end agreement is *not* a generalisation
  claim. The classifier's generalisation claim is the held-out n=24, 5-seed figure
  above, and that figure carries its own caveat (`saturated`: a near-perfect score
  on 24 documents means the task is easy, not that the model is strong).
- **The classifier figure is not the app's classifier in isolation.** There are two
  classifiers in this repo: the ONNX MiniLM head (an introduction-time experiment)
  and the keyword classifier the app actually ships. `classify_doc` in
  `ml/src/evaluate.py` is the shipped one's twin.
- The profile split above is a *reporting* split, not an exclusion. Only
  `evaluate.SUB_HUMAN` profiles (`worst_case`, unreadable to a human) leave the
  headline, and none appear in this sample set.

## The honest comparison a judge should see

The stage-by-stage numbers in `docs/03_RESULTS.md` were measured with a Tesseract
arm. The app no longer contains Tesseract: it ships a 942k-parameter CRNN trained
on SROIE receipt crops whose held-out **crop** word accuracy is 28.0%. The same
harness now measures the shipped engine by default (`--engine=tesseract` is the
opt-in baseline), so the comparison is direct:

| engine | field accuracy | word accuracy | exact dosage | OCR conf |
|---|---|---|---|---|
| tesseract (the removed engine) | 62.2% | 65.4% | 84.4% | 69.7 |
| **crnn (what we ship)** | **7.4%** | **26.4%** | **68.5%** | 82.7 |
| crnn, no restoration (raw) | 5.6% | 26.9% | 68.5% | 85.2 |

10 capture profiles × 6 documents, seed 123, sub-human profiles excluded;
`bash run.sh eval` (default engine), artifact `artifacts/eval_results.json` with
`"engine": "crnn"`.

Page-level recognition is harder than crop-level — it adds the detector's mistakes
and the squeeze to the graph's fixed 256px width — and the end-to-end table above is
what that costs: **the swap to our own recogniser measurably reduced end-to-end
quality**, from 62.2% field accuracy to 0% end to end and 7.4% at the best stage.

A second finding from the re-run: the preprocessing-arm ordering in `docs/03`
(restorer beats classical) does **not** survive the engine change. On the shipped
recogniser `raw` ties or beats the restorer arms, which agrees with
`app/tools/restore_ab.js`'s independent result that restoration *costs* OCR yield.
The restorer's justification was a Tesseract-era measurement.

Two consequences, both already written into `HANDOFF.md` §4:

1. `ml/src/evaluate.py` still measures Tesseract, so `artifacts/eval_results.json`
   describes an engine this application does not contain. Its numbers must not be
   quoted as the product's.
2. The recognition gap is the project's main open problem. The measured levers, in
   order of expected effect: train on page-level text rather than receipt crops;
   re-export the graph at width 320 (the training clamp) instead of 256; and only
   then revisit the preprocessing arms.

## Re-running this

```bash
bash run.sh bench-pipeline     # drive the shipped app (WebKit), write raw records  (~4 min)
bash run.sh bench-hackathon    # score them + the training artifacts, one report     (~10 s)
bash run.sh train-clf          # the 5-seed classifier metrics                      (~2 min)
bash run.sh train-ocr          # the recogniser's held-out crop accuracy            (~40 min, GPU)
bash run.sh eval-ocr           # whole-page word accuracy + CER, detector included  (~10 min)
```

`bash run.sh bench-hackathon` names anything it could not measure instead of
defaulting it to zero, and exits non-zero when a required artifact is absent — a
missing measurement must look missing, not bad.
