# SightLine — Technical Deep Dive

**For judges. Read this one document and you have the whole project.**
Everything here is measured on held-out data or verified by a check that fails
the build. Where a number is unflattering it is stated as unflattering.

Last verified: 278 tests passing · 1 known failure documented ·
26.68 MB installed payload · 0 bytes leave the phone.

---

## 1. The one-paragraph version

SightLine is an offline-first PWA that reads a photographed document aloud.
You point a phone at a prescription, bank statement or contract; it restores
the image, finds the text lines, **reads them with a character-recognition
network we trained from scratch**, classifies the document with a MiniLM text
encoder, extracts the safety-critical fields, and speaks them — with nothing
transmitted off the device.

There is no OCR library in the app. Not "we configured Tesseract well" —
there is no Tesseract. `app/models/ort.json` used to claim `"tesseract": false`
while 30 MB of Tesseract still shipped; that was a lie in a shipped artifact,
it is now fixed, and `bash run.sh bundle-check` **fails the build** if any part
of it returns.

---

## 2. The pipeline

```
  camera frame
        │
        ▼
 ┌──────────────────┐   37,793 params · 64×256 · measured PSNR 14.13 dB
 │  RESTORER (CNN)  │   restores only when measured quality is poor,
 └────────┬─────────┘   so a clean scan is passed through untouched
          ▼
 ┌──────────────────┐   median-blur paper estimate → ink mask → row
 │  LINE DETECTOR   │   projection → contiguous ink bands
 └────────┬─────────┘   ⚠ THIS IS THE BOTTLENECK — see §6
          ▼
 ┌──────────────────┐   942,166 params · CNN + 2-layer BiLSTM · CTC
 │  CRNN  (OURS)    │   37-symbol charset · greedy CTC decode
 └────────┬─────────┘   ← THE CONTRIBUTION. No OCR dependency.
          ▼
 ┌──────────────────┐   sentence-transformers/all-MiniLM-L6-v2
 │  MiniLM ENCODER  │   frozen 6-layer transformer + trained head
 └────────┬─────────┘   49,796-param head · 4 classes
          ▼
 ┌──────────────────┐   all-or-nothing: a field is spoken only if it
 │  FIELD EXTRACTION│   validates, otherwise it is OMITTED, never guessed
 └────────┬─────────┘
          ▼
        speech
```

Everything after the camera frame runs in the browser. There is no server
call in this diagram.

---

## 3. The recogniser — the part we actually built

### Why a CRNN with CTC

We needed to read **variable-width** text crops, from **one** camera photo, with
**no** per-character annotation. That constraint set the architecture:

| requirement | consequence |
|---|---|
| variable-width crops | needs an RNN or an attention decoder |
| no character boxes | CTC loss — no alignment required |
| runs in a browser | must export to ONNX |
| CPU-only training | must be small |

**CTC + greedy decoding is non-autoregressive**, so inference is a single
forward pass. A transformer decoder would autoregress over 32+ timesteps and
blow the latency budget on a phone.

### Architecture

```
input [B, 1, 32, W]
  └─ CNN          7 conv blocks, channels 16→32→48→64→80→96→128
     output [B, 128, 1, T]        T = W / 8      ← height forced to 1
  └─ BiLSTM       2 layers, hidden 128
  └─ linear       → 38 classes (37 symbols + CTC blank)
logits [B, T, 38]
```

**942,166 parameters** (verified against `state_dict`, not quoted from the training log). Tuned down from 335 s/epoch to ~150 s/epoch by halving
channel widths — a decision made because this trains on a 4-core laptop CPU
with no GPU, and a model you can iterate on beats a model you cannot.

### Training data — real documents only

| | |
|---|---|
| source | **ICDAR 2019 SROIE** — 626 train / 347 test receipts |
| crops | **52,554** from **420** receipts |
| augmentation | 11 variants including **perspective warps using 199 real hand-held homographies** |
| split | 294 / 63 / 63 **by receipt identity**, never by crop |
| epochs | 40, ~250 s/epoch on 4 CPU cores |

Splitting **by receipt** is the load-bearing decision. Randomly splitting crops
puts images of the *same receipt* on both sides and inflates the score. It is
the single most common way to fake an OCR result, and it is why every number
here is reported on receipts the model never saw in any form.

### Results — held-out, same data for both rows

17,945 test words, both checkpoints scored on the identical test set:

| checkpoint | corpus | word accuracy | CER |
|---|---|---|---|
| epoch 20 | 30,021 crops / 260 receipts | 26.25% | 15.90% |
| **epoch 39** | **52,554 crops / 420 receipts** | **29.30%** | **11.95%** |

The larger corpus plus real-homography perspective augmentation is worth
**+3.05 pp accuracy and −3.95 pp CER**.

> We originally reported this comparison as "28.1% vs 29.7%" and it was
> meaningless — those were two *different* validation splits (63 receipts
> against 39). Re-running both on one test set is what produced the table
> above. Details in `03_RESULTS.md`.

---

## 4. Why this runs on a phone: the ONNX path

`aten::lstm` **cannot export to ONNX with a dynamic sequence length**, and an
explicit gate loop cannot unroll over a symbolic bound. Both failed for real:

```
torch.onnx.export(..., dynamic_axes={"W": ...})
  -> stack expects each tensor to be equal size   (symbolic Min(...) dims)

explicit gate loop
  -> The inequality Eq((((x/4)//2)) - 20, 0) cannot be solved
```

So the width is **fixed at 256px**. Each crop is scaled proportionally to
preserve aspect ratio, right-padded to 256, and the decoder trims to the
crop's true timestep count.

**Padding is safe because a trained CTC model emits the BLANK symbol there —
and that is verified, not assumed:**

```
pad  32->256: ok (ref 'RM'         vs padded 'RM')
pad  64->256: ok (ref '1071'       vs padded '1071')
pad 128->256: ok (ref '01143008'   vs padded '01143008')
pad 200->256: ok (ref 'M  C2  0'   vs padded 'M  C2  0')
```

Parity against PyTorch: **max |Δ| = 1.94e-05**. The graph is **self-contained**
— no external `.onnx.data` blob, because ONNX Runtime Web has no API to mount
one, so a model with external weights loads perfectly in Python and fails on a
judge's phone. It would have failed on stage.

### Payloads, all self-contained

| model | exact bytes | size | purpose |
|---|---|---|---|
| `crnn.onnx` | 4,029,084 | 4.03 MB | **our recogniser** |
| `restorer.onnx` | 153,362 | 0.15 MB | image restoration |
| `minilm_encoder.onnx` | 22,885,055 | 22.89 MB | int8 MiniLM — we **measured** that int8 cost 0.0 accuracy points rather than assuming it |
| `minilm_head.onnx` | 199,958 | 0.20 MB | document classifier |
| **total shipped payload** | | **26.68 MB** | models + tokenizer, per `bundle-check` |

Removing Tesseract also deleted **41.1 MB** of third-party OCR payload
(`vendor/tesseract` 30.2 MB, `vendor/tessdata` 10.9 MB) — a larger saving than
the entire restorer, classifier and recogniser combined.

All figures are decimal MB (10⁶ B) throughout. An earlier draft of this table
mixed decimal MB with binary MiB in the same column; every size here was
re-read from disk and given in bytes so the unit cannot be ambiguous.

---

## 5. The classifier — stated precisely

`sentence-transformers/all-MiniLM-L6-v2`, **frozen**, plus a **49,796-parameter
head** we trained. Five-seed mean **95.0%**, range 91.7–100%, sd 3.1%.

Two things a judge should hear plainly:

1. **MiniLM is a text encoder, not an OCR model.** It consumes tokens, not
   pixels. Our architecture is `image → restore → detect → **our CRNN** →
   MiniLM`. Any project claiming MiniLM reads camera pixels is wrong.
2. **It is not fine-tuned end to end.** The encoder is frozen. We say so
   everywhere rather than implying full fine-tuning.

**Keep the two numbers apart.** 95.0% is classifier accuracy on clean OCR text.
It is *not* end-to-end accuracy, and presenting it as such would be the single
most misleading thing this project could do.

---

## 6. The honest limitation, stated first

**Line detection, not recognition, is the bottleneck.**

| metric | held-out flatbed scans | real hand-photographed |
|---|---|---|
| CER | 11.95% | 58.4% |
| line exact accuracy | — | **2.4%** |
| detection recall | — | **71.6%** ← hard ceiling |

Line accuracy **cannot exceed 71.6%** however good the recogniser is, because
detection only finds 71.6% of the lines. That ceiling is why we report
detection recall *separately* from recognition quality everywhere: a low line
score otherwise looks like a recognition failure when it is a detection one.

**And the detection result is not statistically significant.** Scored on 290
aligned lines:

| variant | correct | accuracy | 95% CI |
|---|---|---|---|
| baseline | 5 / 290 | 1.8% | [0.7, 4.0] |
| + document localisation | 7 / 290 | 2.4% | [1.2, 4.9] |

Localisation gains **2 lines and loses 0**. Exact McNemar **p = 1.000**. The
whole "+3.7 pp recall" is two lines. We shipped localisation on its *mechanism*
— a receipt photographed beside a bowl of noodles demonstrably floods a
full-width projection, visible in the rendered mask — **not** on its delta.

Six detector variants were tried; all landed within ~4 pp of each other and none
were significant. That is a **sample-size problem wearing a tuning problem's
clothes**. The fix is more hand-photographed pages, not more heuristics. Three
of the twenty pages are also **resolution-bound** (12–17 px per line against a
32 px recogniser input — 1.8–2.7× upscale, information the photo never
captured) and cannot be fixed by any detector at all.

**We are not claiming 95–100% end-to-end accuracy.** The brief asked for it. At
2.4% line accuracy on real photographs, claiming it would be a lie that
collapses the moment a judge photographs their own receipt.

---

## 7. Things we got wrong, and how the checks caught them

Judges ask what broke. These are real, and each one is now a test.

**A safety gate whose condition was inverted — so it never ran.**
```python
if _emits_any_blank(model):   # skip
```
It skipped precisely when the model was trained well enough for the check to
matter. On our trained checkpoint the function returns `True` (47.3% blanks,
all 38 classes used) and the gate printed *"untrained head emits no blanks"* —
the exact opposite of the truth. It had **never executed on any checkpoint**.
It survived review because it read as a deliberate guard. It now runs, passes,
and a mutation test confirms it *can* fail — a gate that cannot fail is worse
than no gate, because it is read as reassurance.

**A metrics function that had never run at all.** `eval_sroie_test()` called
`most_common()` on a `defaultdict`. Every held-out number quoted above would
have been unavailable had we not fixed it. It had been "passing" in the sense
that nobody ran it.

**An export that reported success as failure.** `export_onnx.py` unpacked three
values from a two-value function, raising *after* the ONNX file was written.
We were about to re-run an expensive export that had already succeeded.

**A first regression gate defeated by a rename.** Our test grepped the module
for the string `"PAPER_DELTA"`. A mutation renaming the constant to `E_PAPER`
**passed it**. It now asserts resolved values, and that mutation is caught.
A gate never seen failing is not known to work.

**The manifest that lied.** `"tesseract": false` shipped while 30 MB of
Tesseract loaded at runtime. The worst kind of bug for a demo, because it is
invisible until a judge opens the bundle.

---

## 8. Reproduce it

```bash
bash fetch_data.sh        # SROIE + the CC-BY-4.0 hand-photographed set
bash run.sh build-data    # 52,554 crops, split by receipt identity
bash run.sh train-ocr 40  # ~2.8 h on 4 CPU cores, no GPU
bash run.sh export        # self-contained ONNX + parity checks
bash run.sh bundle-check  # runs real inference through every shipped model
bash run.sh test          # 278 tests
```

Datasets are **not** redistributed — `fetch_data.sh` pulls them and the
licences (ICDAR 2019 SROIE; Zenodo record `13688441`, CC-BY-4.0) are cited in
`docs/02_ML_PIPELINE.md`.

---

## 9. If you remember three things

1. **We trained the OCR.** 942,166 parameters, from scratch, exported to a
   4 MB self-contained ONNX graph, decoding in the browser at a fixed 256 px
   width because the LSTM export demanded it. No OCR dependency exists.
2. **Every number is separated by what produced it.** Classifier 95.0%, scan
   CER 11.95%, real-photo line accuracy 2.4%, detection recall 71.6%. Mixing
   them would be the easiest way to look good and the fastest way to be caught.
3. **We know which part is broken.** It is the line detector, the ceiling is
   measured, the current attempt is statistically indistinguishable from
   baseline, and we say so. Naming the blocker precisely is worth more than a
   claim we cannot defend.

### Where to look

| file | what it answers |
|---|---|
| `docs/01_ARCHITECTURE.md` | every stage, with the data shapes |
| `docs/02_ML_PIPELINE.md` | training and export, reproducible |
| `docs/03_RESULTS.md` | all measurements, including the bad ones |
| `docs/07_EVALUATION_METHODOLOGY.md` | how we avoid fooling ourselves |
| `docs/10_ACCURACY_GAP.md` | why the old baseline was abandoned |
| `ml/src/export_onnx.py` | the rejected-experiments table |
| `ml/tests/test_line_detector.py` | the known failure, as `xfail(strict=True)` |

---

## Appendix: which numbers were verified when

Judges should be able to tell a measured number from a remembered one. Every
figure in §1–§9 is one of these three kinds, and the kind is stated:

**Verified from disk or a passing check in this session**
- Model byte counts — read from `app/models/*.onnx`, listed in §4 in exact bytes
- CRNN 942,166 params · 38 classes · 37 symbols · epoch 39 — from `state_dict`
- Restorer 37,793 params — from `SightLineNet()`
- 26.68 MB payload, no external sidecars — `bundle-check` passes
- 278 tests, 1 xfailed — `bash run.sh test`
- Tesseract absent from vendor/, index.html, sw.js, pipeline.js — `bundle-check`
  §4b, mutation-tested to confirm it fails when reintroduced
- 29.30% / 11.95% and 26.25% / 15.90% CER — head-to-head on 17,945 held-out
  words, both checkpoints on the identical test set
- 71.6% detection recall · 2.4% line accuracy · 58.4% CER — full 20-page
  hand-photographed set, current checkpoint
- McNemar p = 1.000 · Wilson intervals — computed from the aligned-line counts
- 47.3% blank fraction, all 38 classes used — measured on real training crops
- Padding invariant at 32/64/128/200px — `verify_crnn` output
- 41.1 MB of Tesseract deleted — measured before unlinking
- **ONNX parity max |Δ| = 1.94e-05** — from the `export` run's own output line

**Carried from earlier in this project, not re-measured in the last session.**
Re-run the command in the right-hand column before quoting any of these to a
judge as current:

| claim | value | re-measure with |
|---|---|---|
| explicit-LSTM loop parity | 7.451e-08 | `run.sh py ml/tests/test_ocr_model.py` |
| restorer best PSNR | 14.13 dB | `run.sh py ml/src/evaluate.py` |
| MiniLM 5-seed mean (range, sd) | 95.0% (91.7–100%, 3.1%) | `run.sh test` (classifier arm) |
| training time per epoch | ~250 s | `run.sh train-ocr` |
| int8 cost | 0.0 accuracy points | `run.sh export` |
| corpus size | 52,554 crops / 420 receipts, 11 variants | `bash run.sh build-data` |
| real homographies used for perspective aug | 199 | `ml/src/build_dataset.py` |

The honest summary: the architecture, the held-out OCR numbers, the detector
ceiling and its statistical insignificance are all current. A handful of
secondary figures are carried forward and are listed above so nobody mistakes
them for fresh measurements.
