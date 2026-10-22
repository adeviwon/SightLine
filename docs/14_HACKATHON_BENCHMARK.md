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
| OCR, held-out crops | word accuracy | **28.0%** | `bash run.sh train-ocr` → `artifacts/ocr_training.json` (7,875 held-out crops) — the **pre-render-retrain** ledger row |
| OCR, held-out crops, the **SHIPPED** retrain3 model | exact crop match / CER, paired on identical crops | render corpus (n=1500) **68.1% / 0.150**; real rectified photos (n=115) **25.2% / 0.474**; SROIE scans (n=1500) **74.1% / 0.098** | `bash run.sh py ml/src/score_checkpoints.py` — commit `348cb6d`. The same crops scored on the pre-retrain graph read 27.7% / 0.201, 14.8% / 0.523, 73.4% / 0.101 |
| OCR, held-out real scans (SROIE) | word accuracy **29.30%**, CER **11.95%** | 361 SROIE receipts, 17,945 words | `bash run.sh eval-ocr` section A; `docs/03_RESULTS.md` §3 |
| OCR, 20 real hand-photographed pages | line-exact **1.8%**, CER **56.5%**, detector recall **67.9%** | 275 lines from 20 pages | `bash run.sh eval-ocr` section B — the honest hard case (`docs/03_RESULTS.md` §3) |
| OCR, the 193-page hand-photographed set, arm B | line-exact **2.4%**, CER **58.4%**, detector recall **71.6%** | harness reproduction of the shipped checkpoint | `HANDOFF.md` §3 (`ml/experiments/eval_official_ab.py`) |
| the shipped app, end to end | field recovery, classification, word accuracy, CER | **57.2% word acc · 0.217 CER · 33.3% fields · 91.7% classification** | `bash run.sh bench-pipeline` → `artifacts/pipeline_raw_shipped.json` — the shipped build on the **current** fixtures. (`pipeline_raw_deskew.json` is the previous fixture generation; see the A/B below.) |
| one documented failure | a real misread | `ACCOUNT NUMBER 40218877` → `ICOUNT NOUMBER 4021887` | `.hermes/plans/2026-10-20_164918-sightline-ocr-accuracy.md` |

## The shipped app, end to end (the number that decides whether it works)

`bash run.sh bench-pipeline` drives the **real app** in WebKit — iOS Safari's
engine — over 12 rendered documents (3 types × 4 capture profiles) and records
what came out, one record per document. `bash run.sh bench-hackathon` scores
those records with the definitions in `ml/src/bench_hackathon.py`.

**The shipped number, scope stated: 57.2% word accuracy, CER 0.217, 33.3%
ground-truth field recovery (8 of 24), 91.7% document classification** — on
**12 synthetic rendered documents** (3 document types × 4 capture profiles),
**WebKit**, **one record per document**, measured on the **current**
capture-profile fixtures with the shipped `retrain3` model
(`artifacts/pipeline_raw_shipped.json`). Real hand-photographed pages remain the
honest hard case, and are stated in full under "What these numbers do NOT prove"
below.

### Two fixture generations, and the controlled A/B that separated them

The bench fixtures were rendered **2026-10-20 07:37**. `CAPTURE_PROFILES` was then
re-tuned **2026-10-20 23:54**, sixteen hours later, to match real photographs — the
old profiles measured **17–67× harsher than the photographs** they were standing in
for. From the moment of that re-tune, the bench scored every model against images
made with the old profiles while the trainer learned the new ones, so every
end-to-end row measured before the re-render describes a distribution the model
never trained on. A controlled A/B — the **old** graph extracted from git, re-benched
on the **same regenerated fixtures** — separates the two effects:

| model / fixtures | word | CER | fields | classification | artifact |
|---|---|---|---|---|---|
| OLD model / **stale** fixtures — what these docs used to report | 45.9% | 0.303 | 20.8% | 83.3% | `pipeline_raw_deskew.json` |
| OLD model / **current** fixtures — the A/B control | 47.7% | 0.278 | 20.8% | 91.7% | `pipeline_raw_oldmodel_currentfix.json` |
| NEW model / **stale** fixtures — the apparent regression | 43.2% | 0.316 | 16.7% | 75.0% | `pipeline_raw_retrain3_oldfixtures.json` |
| **NEW model / current fixtures — SHIPPED** | **57.2%** | **0.217** | **33.3%** | **91.7%** | `pipeline_raw_shipped.json` |

- **fixture change alone: +1.7 points** (45.9% → 47.7%, the same old weights) — the
  stale benchmark *understated* the app;
- **model change alone: +9.5 points** (47.7% → 57.2%, same current fixtures);
- the **−2.7-point "regression"** (45.9% → 43.2%) was an artefact of comparing
  across two fixture generations — the retrain was being scored on images the
  trainer no longer produced.

**Never read 45.9% → 57.2% as +11.3 points of model improvement.** +1.7 of it is the
benchmark becoming more realistic, and only the +9.5 is attributable to the retrain.
Rows 1–5 below were measured on the **previous fixture generation**: they are records
of that generation, not rows in the same series as 57.2%.

### How we found this — and the gate that stops it recurring

Nothing here is caught by reading the code: the profile parameters were right when
the fixtures were made and right when the model was trained — they were just right
at **different times**. The bench fixtures predated the profile re-tune by sixteen
hours, so the benchmark measured a distribution the model never trained on, and the
result looked exactly like a fact (a 2.7-point regression from a retrain that was,
on held-out crops, +40.4 points better).

The fix is not "regenerate the fixtures" — that is a repeat appointment. It is a
**profile fingerprint** — a deterministic `sha256` over the profile *parameters*
(canonical JSON, sorted keys) stamped into every rendered artifact at render time —
plus **`ml/tests/test_fixture_freshness.py`**, which compares each stamp against the
live parameters and **fails naming the re-render command**. Seen failing first (two
failures: *"carries NO profiles_fingerprint … Render it again: bash run.sh py
app/tools/render_samples.py"*), then passing. A gate never seen failing is not known
to work.

This is a **strength, not an embarrassment**, and it is the same class of defect as
the four train/serve contract defects below: a reference that quietly stopped
describing the thing it was measuring, found by measurement rather than by a reader.
It is the third stale-reference trap of this project — after the browser width
fixture and the legacy ONNX graph — and it is now the only one of the three with a
stamp that survives a file being touched.

### The trajectory (measured on the PREVIOUS fixture generation — kept as the record it is)

Same 12 documents, same metric, each step a defect found in our own train/serve
contract, with the step that regressed kept in. **Every row below was measured on the
stale fixture generation** and is **not directly comparable** to the 57.2% above:

| # | word accuracy | CER | field recovery | classification | what changed / raw records |
|---|---|---|---|---|---|
| 1 | 3.6% | 0.708 | 0.0% | 33.3% | as shipped on 10-20 — graph baked at a FIXED 256 px against 320 px training crops, `preprocess()` could only rank among degraded arms, and deskew was applied in the wrong sense · `artifacts/pipeline_raw.json` |
| 2 | 22.4% | 0.575 | 4.2% | 50.0% | export width brought to the training clamp (320), same checkpoint — the app had been feeding the model the first 256 px of **98.8%** of lines · `artifacts/pipeline_raw_final.json` |
| 3 | 17.1% | 0.600 | 4.2% | 58.3% | **a step back on purpose:** the dynamic-width graph *and* retrained weights went in together and measured worse on these 12 documents, even though at crop level the same retrain moved exact line match **3.3% → 57.0%** · `artifacts/pipeline_raw_dynamic.json` |
| 4 | 35.0% | 0.465 | 16.7% | 66.7% | the untransformed image made reachable — `preprocess()` already offered the raw deskewed grayscale, but a later branch had replaced the candidate list with the restorer's three degraded arms, so the arm that measures best could never win · `artifacts/pipeline_raw_rawarm.json` |
| 5 | **45.9%** | **0.303** | **20.8%** | **83.3%** | **shipped at the time, on the previous fixture generation:** deskew applied in the correct sense — the estimator measured the right angle and the correction **doubled** the tilt instead of removing it (`off_axis` 1.1% → 45.1%, CER 0.882 → 0.234) · `artifacts/pipeline_raw_deskew.json` |

Raw records for every row, in that order: `artifacts/pipeline_raw.json`,
`pipeline_raw_final.json`, `pipeline_raw_dynamic.json`,
`pipeline_raw_rawarm.json`, `pipeline_raw_deskew.json`.

Re-benched on the **current** fixtures, the same two model generations read
**47.7% / 0.278 / 20.8% / 91.7%** (old weights,
`artifacts/pipeline_raw_oldmodel_currentfix.json`) and **57.2% / 0.217 / 33.3% /
91.7%** (the shipped retrain3 weights, `artifacts/pipeline_raw_shipped.json`).

### Per capture profile

On the **previous fixture generation**, shipped-at-the-time (deskew) build:

| profile | n | field recovery | classification | word accuracy | CER |
|---|---|---|---|---|---|
| studio_clean | 3 | 33.3% | 100.0% | 52.3% | 22.5% |
| handheld_light | 3 | 16.7% | 66.7% | 46.5% | 37.6% |
| off_axis | 3 | 16.7% | 100.0% | 45.1% | 23.4% |
| jpeg_social | 3 | 16.7% | 66.7% | 39.8% | 37.8% |
| **all samples** | **12** | **20.8%** | **83.3%** | **45.9%** | **30.3%** |

On the **current** fixtures, shipped (retrain3) build —
`artifacts/pipeline_raw_shipped.json`:

| profile | n | field recovery | classification | word accuracy | CER |
|---|---|---|---|---|---|
| studio_clean | 3 | 33.3% | 100.0% | 60.1% | 16.9% |
| handheld_light | 3 | 66.7% | 100.0% | 75.2% | 7.5% |
| off_axis | 3 | 0.0% | 100.0% | 49.6% | 25.5% |
| jpeg_social | 3 | 33.3% | 66.7% | 43.7% | 36.9% |
| **all samples** | **12** | **33.3%** | **91.7%** | **57.2%** | **21.7%** |

All 12 scans completed with no crash and no failed scan (`"error": null` on every
record). `field_recovery` is the app's own extracted-field metric — 33.3% = **8 of
24** ground-truth fields on the shipped build (**20.8% = 5 of 24** on the previous
fixture generation); the weaker "present anywhere in the read text" count is **14 of
24** (was 9 of 24), and both come from the same 12 records, so quote which one you
mean. Row-level evidence in `artifacts/hackathon_metrics.json`.

### Each remaining step was a defect in our own contract, not a model change

**The 256 px graph.** The app's ONNX graph was exported at a FIXED 256 px while
`build_dataset.crop_to_array` builds every training crop at its NATURAL width
clamped to `MAX_W = 320` — and the corpus median IS that clamp (only 40 of 3,270
held-out crops are <= 256 px). So the app fed the model the first 256 px of nearly
every line. On 200 held-out crops, crop-level exact match was **44.7% at natural
width against 0.0% cut to 256**, and the read changed on **200/200**: `PATIENT JANE
DOE DOB 04021979` came back as `PERIENT JANE DOE 2O08010`, `DATE 01102026` as `DATE
01020`, `PRESCRIBER DR A AHMED MBBS` as `PRESCRIBER DR A AHMNT`. Every symptom
previously attributed to the model — `STATEMENI OT ACCOUN`, "read tokens 19-54
against 73-90 ground-truth words", "the CRNN digit-ises our letters" — was the tail
never reaching it. The end-to-end effect on the same 12 documents: word accuracy
**3.6% → 22.4%**, mean CER **0.708 → 0.575**, read tokens **468 → 680 (+45%)**, and
on the clean profile **10.3% → 61.1%** word accuracy with classification **66.7% →
100%** (which also confirms the classifier was being fed the truncation, not
failing on its own — MiniLM measures 95.8% macro-F1 on real text).

**Dynamic width + retrain.** The next change swapped in a dynamic-width graph and
retrained weights together; on these 12 documents the pair measured *worse*
(22.4% → 17.1%), even though at crop level the same retrain moved exact line match
**3.3% → 57.0%** (CER 0.156). The disagreement is kept in the record on purpose: it
is why the next fix was measured on the preprocessing arms rather than on the model.

**The elected arm.** `preprocess()` already offered the raw deskewed grayscale as
one candidate, but a later branch had replaced the candidate list with the
restorer's three degraded arms, so the arm the repo's own measurements rank best
could never be chosen. Restoring it (and its `structure_score` election) took the
total from 17.1% to 35.0%.

**The deskew sign.** `pipeline.estimateSkew` measured the right angle, and
`preprocess()` then **doubled** the tilt instead of removing it. Correcting the sign
took `off_axis` from **1.1%** to **45.1%** word accuracy (CER 0.882 → 0.234) and the
end-to-end total from 35.0% to **45.9%** on the **previous fixture generation** — the
same fix, re-benched on the current fixtures, is the A/B control row at 47.7%. This
was one sign, not a model change.

The shipped graph now carries a DYNAMIC width axis (`app/models/crnn.onnx` =
`artifacts/crnn_dynamic.onnx`, 3,785,916 B), so line width is no longer a constant
that can drift from the training crops the way 256 did. The contract is pinned by
`ml/tests/test_graph_width_contract.py` (a fixed axis must EQUAL the clamp `MAX_W`,
or the axis must be symbolic — the old `width <= MAX_W` assertion is what let 256
ship), `app/tools/crop_contract.js` and `app/tools/dynamic_width_check.js`.

Words are scored with `evaluate.word_accuracy` (fraction of ground-truth words
present in the read text, after `norm()`), and CER as (substitutions + insertions
+ deletions) / ground-truth characters on normalised text. Both definitions live in
`ml/src/bench_hackathon.py` and are unit-tested in `ml/tests/test_bench_hackathon.py`
— the browser side deliberately computes no metrics, so the two cannot drift apart.

## What these numbers do NOT prove

- **This is not a model improvement.** The 35.0% → 45.9% step — the one that
  produced the number these docs used to ship — fixed a defect in our own serving
  path: the deskew estimator measured the right angle and `preprocess()` applied the
  correction in the **wrong sense**, doubling the tilt instead of removing it. The
  recogniser's weights did not change. The earlier 3.6% → 22.4% step is the same
  story (a graph baked at 256 px fed the model the first 256 px of 98.8% of lines).
  Read the whole trajectory as train/serve-contract repair, not as a claim about how
  strong the recogniser is.
- **And the fixture correction is not a model improvement either.** Of the 11.3
  points between the old 45.9% headline and the current 57.2%, **+1.7 is the
  benchmark getting more realistic** (stale → current fixtures, same old weights) and
  **+9.5 is the retrain** (current fixtures, old → shipped weights). The two effects
  are separated by the controlled A/B above; quoting 45.9% → 57.2% as a single model
  gain would be the overclaim this section exists to prevent.
- **There is a 37% capture-profile coverage gap.** `ml/src/profile_coverage.py`
  measures the app's own crop rule over the bench documents: 155 crops, of which
  **58 (37%)** come from `off_axis` and `jpeg_social`. The corpus the shipped
  checkpoint trained on was built from `render_corpus.py`'s default `PROFILES[:4]`
  — studio_clean, handheld_light, handheld_heavy, low_light. So on 37% of what the
  app is served, the model is asked about a photographic world it has never seen.
  That is a corpus-coverage result, not a property of the recogniser.
- **Real hand-held photographs still read at 1.8% line-exact.** The 57.2% above is
  on **12 synthetic rendered documents**, in WebKit, one record per document; on 20
  real hand-photographed pages the line detector recovers 67.9% of lines and
  line-exact accuracy is **1.8%** (CER 56.5%, `docs/03_RESULTS.md` §3). The two are
  not the same measurement on the same data, and 57.2% must never be quoted as if
  the recogniser had closed that gap.
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

Page-level recognition is harder than crop-level — it adds the detector's mistakes —
and the end-to-end table above is what that costs: **the swap to our own recogniser
measurably reduced end-to-end quality**, from the Tesseract arm's 62.2% field
accuracy to **33.3% field recovery (8 of 24 fields)** on the current fixtures — 20.8%
(5 of 24) on the previous fixture generation, which is what this paragraph used to
quote — and 7.4% at the best *stage* of the `evaluate.py` harness.

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
   close the 37% capture-profile coverage gap
   (`ml/src/profile_coverage.py`); and only then revisit the preprocessing arms.
   The graph-width lever that used to head this list is **closed** — the shipped
   graph is dynamic-width now, not baked at 256 px.

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
