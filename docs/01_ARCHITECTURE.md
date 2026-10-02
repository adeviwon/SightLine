# Architecture — how SightLine actually works

> **Status:** two neural models, both trained by us, both running on-device.
> Tesseract has been removed. Nothing is sent to a server.

---

## 1. The one-paragraph version

You point a phone at a document. A small **restoration CNN** decides whether the
image is damaged and, if so, undoes motion blur, defocus and shadow. A
**CRNN+CTC recogniser** we trained on real receipts reads the text. A **fine-tuned
MiniLM** classifies what kind of document it is. Field extraction pulls out the
safety-critical numbers. The phone speaks them aloud. Every model runs locally in
the browser; no network request is made after installation.

---

## 2. Why there are two vision models and not one

A restored image is easier to *read*; it is not automatically easier to
*classify*. The two tasks want different things from the pixels, so they get
separate models:

| Stage | Model | Trained on | Job |
|---|---|---|---|
| 1. Restore | `SightLineNet` — 37,793 params CNN | synthetic degraded patches | make damaged text legible |
| 2. Recognise | `SightLineCRNN` — 942,166 params CRNN+CTC | **30,021 real word crops** | read the text |
| 3. Classify | MiniLM L6 V2 + trained head | 174 document templates | decide banking/medical/legal/general |
| 4. Extract | regex + structure scoring | — | pull out dosages, account numbers, dates |
| 5. Speak | Web Speech API | — | read the confirmed fields |

Stage 2 is new and is the subject of most of this document, because it replaces
Tesseract.

---

## 3. The recogniser (stage 2) — our own OCR

### 3.1 Why we stopped using Tesseract

Two reasons, in order of weight.

**Tesseract's ceiling was real and we proved it.** Before removing it we ran
`app/tools/psm_sweep.js`: 336 configurations (12 samples × 7 page-segmentation
modes × 4 preprocessing candidates). On the realistic handheld captures it
recovered **no safety-critical field at all**, at any setting. That is a
statement about Tesseract, not about our preprocessing — which is exactly the
kind of claim that deserves a measurement instead of an assertion.

**It is not our model.** The brief asks for models we train. Shipping a
third-party OCR engine as the centrepiece would mean the most important claim
in the demo — "we trained this" — was false for the stage that does the actual
reading.

### 3.2 Why the recogniser is CRNN and not a transformer

It has to run inside a phone browser, in WASM, on no GPU. That budget decides
the architecture:

- **CTC greedy decoding** is a softmax over 38 classes per timestep. No beam
  search, no lexicon, no autoregressive loop. That is what makes one
  self-contained ONNX file viable in a browser.
- **CTC needs no alignment labels.** SROIE gives word boxes, so we could have
  supervised character positions directly; we deliberately did not. CTC
  marginalises over every valid alignment, which is why the synthetic corpus can
  be added as extra training signal later without any label work.
- **A BiLSTM** reads the line sequentially in one recurrent pass. Self-attention
  over W/8 timesteps would be a bigger graph and more memory for a task that is
  fundamentally sequential alignment.

### 3.3 The shape of the network

```
input   [B, 1, 32, W]        grayscale word crop, width varies 16..320
   |
CNN      7 blocks, stride on width only     ->  [B, 128, 1, W/8]
   |
BiLSTM   2 layers x 128 hidden, bidirectional ->  [B, W/8, 256]
   |
Linear   128*2 -> 38                        ->  [B, W/8, 38]  logits
   |
CTC greedy decode                          ->  text
```

**Height must land on exactly 1 row.** The LSTM reads one row; that row is the
time axis. Getting this wrong is silent — a model with height 2 trains on two
interleaved sequences and never converges. `forward()` asserts it.

**Width reduces to W/8, i.e. ~8px per timestep.** Measured over the 30,021
crops that is a median **2.16 characters per timestep** (p5 = 1.48), which is the
healthy range for CTC: it needs ≥1 timestep per character and benefits from 2–3
so it has room to place the blank between repeats.

**GroupNorm, not BatchNorm.** Batch statistics would make the exported graph
behave differently at batch size 1, which is exactly the case the phone hits.

### 3.4 The charset is deliberately small

37 symbols: 10 digits + 26 letters + space. **No punctuation.**

Punctuation is the highest-error class in practice and contributes nothing to
any field this app extracts. Spending model capacity on it would cost accuracy
where it matters. Every field regex tolerates a missing decimal point.

This is a real trade and it has a visible consequence: `9.00` is read as `900`.
For the fields we extract that is safe, because `TOTAL 900` and `TOTAL 9.00`
both match the same pattern. It would not be safe for a product that had to
report currency to the cent, and this is written down so nobody discovers it by
accident.

### 3.5 The training data is real photographs

This is the part that differs most from the rest of the project, and it is the
part that matters.

| Corpus | What it is | License |
|---|---|---|
| **ICDAR2019-SROIE** (Huang et al. 2019) | 626 train + 347 test **real scanned receipts**, per-word boxes *and* transcriptions | research use |
| **Express Expense** (Auad et al., SIBGRAPI 2024) | 198 **hand-photographed** restaurant receipts + 4-corner homographies | CC-BY-4.0 |

`capture.py` still generates synthetic documents, and it still has a job: when
you know exactly what damage was applied you can test the degradation **gate**
against a known answer. But it is the wrong distribution for training a
recogniser, and the difference is not subtle:

| | synthetic render | real receipt |
|---|---|---|
| ink | uniform black | thermal, fades across the page |
| paper | flat `#f8f8f8` | pink cast, uneven |
| extras | none | barcode, handwriting, ghost print from reverse |
| strokes | clean, one font | thin, small, JPEG-ringing |

No amount of augmentation bridges a gap in the **imaging process**.

**Splits are by document, never by crop.** A receipt contributes ~40 word crops.
Splitting crops would put the same receipt in train and test and report
memorisation as generalisation. We split on receipt ID: **182 train / 39 val /
39 test receipts.**

**Augmentation is applied at build time, once, deterministically.** Nine
variants, each mapped to a failure we actually observed: `clean`, `jpeg`,
`motion`, `defocus`, `dark`, `bright`, `lowres`, `shadow`, and `composite`
(2–3 stacked — real captures are never singly degraded, and a model trained only
on single degradations fails on the combination, which is the only case that
occurs in the field).

---

## 4. Stage 1 — the restorer, and the gate that decides when to use it

The restorer is a 37,793-parameter residual CNN on 64×256 patches with identity
initialisation. Best validation PSNR **14.13 dB**.

Unconditionally applying it is actively harmful: on undamaged patches it costs
**−55.76 dB**. It is a net −0.10 dB on the mean.

So a **degradation gate** decides per image whether to restore. It measures
sharpness (variance of a Laplacian) over `64×256` **tiles** and takes the
**median** tile weight, because blank margins must not dilute a blurred page and
one noisy tile must not force restoration of a clean one.

```
SHARP_CLEAN = 0.080     SHARP_BAD = 0.008     CONTENT_MIN = 0.12
```

Measured effect:

| Patches | identity | gated | Δ |
|---|---|---|---|
| undamaged (n=71) | −55.76 dB | **+0.00 dB** | clean scans untouched |
| mildly damaged (n=184) | +1.25 dB | ~0.00 dB | neutral |
| badly damaged (n=165) | +5.81 dB | **+4.69 dB** | where the work is |

The Python and JavaScript gates are **bit-for-bit equivalent across 30 test
tiles**, checked by `app/tools/gating_parity.js`. Two different gates exist and
they are not interchangeable:

- `pipeline.js assessQuality` — full-image, luminance 0–255, **warns the user**
- `gating.js degradation_weight` — tiled, luminance 0–1, **controls restoration**

---

## 5. Stage 3 — MiniLM, fine-tuned

`sentence-transformers/all-MiniLM-L6-v2`, frozen encoder with a trained
49,796-parameter classification head over 4 document classes.

**MiniLM is a text encoder. It cannot read pixels.** That is not a workaround —
it is why stages 1 and 2 exist. MiniLM consumes the *output* of recognition and
answers "what kind of document is this?" The honest framing for a judge:

> The recogniser turns pixels into text. MiniLM turns that text into a document
> type. Both are our models, and both run on your phone.

Five-seed result: **mean 95.0%**, range 91.7–100%, σ 3.1%, bootstrap CI
[86.7–100%]. We report the mean and the spread, never the 100% seed.

Quantised to **INT8** for the browser: 86.8 MB → **22.16 MB**, with 26/26
held-out class agreement against FP32.

---

## 6. Stage 4 — extraction, and why it is all-or-nothing

Field extraction is regex-based plus a **structure score**. Neither Tesseract nor
our recogniser is trusted with a number:

- A field is emitted **only if it appears complete and well-formed**.
- `2024-CV-OO456` (letter O's) is **rejected**, not corrected. A near-match that
  looks like a case number is more dangerous than a missing one.
- Uncertain fields are **omitted, not guessed**. The app says less rather than
  lying. This asymmetry is deliberate: a false negative is recoverable, a false
  positive announces a wrong dosage.

The structure score counts recognisable case references, account/sort codes,
dosage patterns, dates and currency amounts. It exists because OCR **candidate
selection** was losing fields even when they were present: a result at 81%
confidence recovering *both* fields used to lose to a result at 69% recovering
*neither*, purely because confidence was the only ranking criterion.

---

## 7. Deployment

Single static PWA. One codebase for iOS Safari and Android Chrome.

| Model | Format | Size | Runs in |
|---|---|---|---|
| Restorer | ONNX FP32 | ~0.2 MB | onnxruntime-web WASM |
| CRNN recogniser | ONNX FP32 | ~3.8 MB | onnxruntime-web WASM |
| MiniLM encoder | ONNX **INT8** | 22.16 MB | onnxruntime-web WASM |
| Classification head | ONNX FP32 | ~0.2 MB | onnxruntime-web WASM |
| Tokenizer | `tokenizer.json` | 0.68 MB | vendored |
| **Total payload** | | **~22.8 MB** | |

All ONNX files are **self-contained**. ONNX Runtime Web cannot mount Python-style
external `.onnx.data` sidecars — the failure is
`Module.MountedFiles is not available`, and `bundle-check` now *rejects* any
browser model containing external tensor references.

Ship gates, all mutation-tested (the historical bug is reintroduced and the gate
is confirmed to fail):

```
repro-check   pixel-identical corpus across separate PYTHONHASHSEED processes
bundle-check  real ONNX inference, no external data, tokenizer present, payload size
privacy       no network/analytics/SDK in shipped JS; data files URL-scanned
```

---

## 8. Why offline is an architectural requirement, not a feature

Every existing alternative uploads the document:

| App | Where the OCR runs | Privacy consequence |
|---|---|---|
| Google Lens | Google servers | Document image leaves the device |
| Microsoft Seeing AI | Azure servers | Same |
| Be My Eyes | Volunteer's phone | A stranger sees your medical document |
| **SightLine** | **The user's own phone** | **Nothing is transmitted. Ever.** |

This matters most for exactly the documents SightLine targets — prescriptions
(reveals medical conditions), bank statements (reveals financial distress),
legal correspondence (reveals disputes).

The guarantee is verifiable three ways:

1. **Read the source.** `app/sw.js` precaches every asset and never falls back
   to the network after install.
2. **Break the network.** Turn on airplane mode. The app keeps working.
3. **Capture the proof.** `sudo tcpdump -i en0 | grep -i sightline` during a
   scan produces no packets. Show this on a slide.

No service-worker update path, no analytics, no telemetry, no crash reporting.
That is a feature.

---

## 9. Module map

| Component | Python (training/eval) | JavaScript (device) | Model artifact |
|---|---|---|---|
| Capture simulation | `ml/src/capture.py` | — | — |
| Corpus + augmentation | `ml/src/corpus.py` | — | — |
| Restorer | `ml/src/model.py` | `app/js/restorer.js` | `restorer.onnx` |
| Degradation gate | `ml/src/gating.py` | `app/js/gating.js` | — |
| **Recogniser** | `ml/src/ocr_model.py`, `train_ocr.py` | `app/js/recognizer.js` | `crnn.onnx` |
| Real-data prep | `ml/src/realdata.py`, `build_dataset.py` | — | `ocr_train.npz` |
| Classifier | `ml/src/train_classifier.py` | `app/js/classifier.js` | `minilm_encoder.onnx`, `minilm_head.onnx` |
| Field extraction | `ml/src/evaluate.py` | `app/js/pipeline.js` | — |
| Evaluation | `ml/src/eval_ocr.py` | `app/tools/*` | — |

Each JavaScript module mirrors a Python file. The mapping is documented in
`app/js/ORCHESTRATION.md`.

---

## 10. Threading and memory

```
camera frame (48 MB for 12 MP)
  └─► downscale to 1600px long edge      (~3 MB)
        └─► grayscale                     (~1.5 MB)
              └─► 64×256 tiled inference  (restorer, ~4 KB/tile)
                    └─► line detection + word crops  (~32 KB per crop)
                          └─► CRNN inference per word
                                └─► text string (a few KB)
```

| Work | Thread | Why |
|---|---|---|
| Camera capture | main | browser requirement |
| Restoration | Web Worker | keeps the UI at 60fps during a scan |
| ONNX inference | Web Worker (WASM) | `ort.env.wasm.numThreads = min(4, cores-1)` |
| Recognition | Web Worker | per-word inference is many small calls |
| Speech | main | browser requirement |

---

## 11. What happens when it fails

| Failure | Detection | Response |
|---|---|---|
| Image too blurry | Stage-1 gate weight or PSNR gain | "Hold the phone steadier and retake" |
| Too dark | Mean luminance below threshold | "Move somewhere with more light" |
| Low recognition confidence | Mean CTC softmax over emitted characters | "I could not read this clearly — retake" |
| Unknown document type | Softmax max below threshold | Reads it anyway, but says "I am not sure what kind of document this is" |
| Wrong field | Field regex fails to parse | That field is **omitted from the spoken output**, never guessed |
| Model file corrupt | ONNX load throws | Silent fallback to classical path |

That last column is the contract: **the app degrades, it never guesses.**

---

## 12. What is honest about the numbers

| Claim | Number | Note |
|---|---|---|
| MiniLM classification | **95.0%** mean over 5 seeds | the metric the brief's 95–100% bar is met by |
| End-to-end field accuracy | **62.2%** | *synthetic* corpus, 180 samples |
| Real held-out receipts | see `docs/03_RESULTS.md` | real photographs |
| Handheld blur, Tesseract | **0%** | proven over 336 configurations |

The 95–100% target is met **for the classifier**. It is **not** met for
end-to-end field extraction, and we do not conflate the two. Field extraction
is all-or-nothing by design, which is a much harder bar than per-character
accuracy.

See `docs/10_ACCURACY_GAP.md` for the full accounting.

---

## 13. Reading order for a judge

1. This file — the shape of the system.
2. `docs/02_ML_PIPELINE.md` — how each model is trained, with commands.
3. `docs/09_INTERPRETING_THE_NUMBERS.md` — which number means what.
4. `docs/10_ACCURACY_GAP.md` — what does not work, and the evidence.
5. `docs/00_HACKATHON_PLAYBOOK.md` — the demo, the pitch, the Q&A.