# Results

> **Every number on this page was produced by `bash run.sh eval` and
> `bash run.sh export`.** Nothing here is typed by hand.
> Regenerate: `bash run.sh eval && bash run.sh export && bash run.sh results`

Machine-readable versions live in `artifacts/eval_results.json` and
`models/classifier/minilm_head_eval.json`.

> **Provenance: current.** The corpus generator uses `seedutil.name_hash` (zlib.crc32), `bash run.sh repro-check` passes, and these artifacts carry the matching seeder stamp. These numbers can be re-derived exactly:

> ```bash
> bash run.sh eval && bash run.sh export && bash run.sh results
> ```

**Section 2 reports the default split, which scored 100%.** Across 5 seeds the honest figure is **95.0% mean, macro F1 0.947, min 91.7%, CI [86.7%, 100%]** on a 24-document test set where one document is worth 4.17 points. Quote the 5-seed number.

---

## 1. Restorer training

SightLineNet — **37,793 parameters**, Charbonnier loss, AdamW, cosine schedule.

- Training patches: **336**  
- Validation patches: **84**  
- Split by **document seed** (no document contributes patches to both sides)  
- Best validation PSNR: **14.131 dB**

| epoch | train loss | val PSNR (dB) |
|---|---|---|
| 1 | 0.15615 | 12.48 |
| 4 | 0.12755 | 13.57 |
| 7 | 0.12826 | 13.28 |
| 10 | 0.12338 | 12.85 |
| 13 | 0.10850 | 12.99 |
| 16 | 0.10489 | 13.56 |
| 19 | 0.10220 | 13.58 |
| 22 | 0.09449 | 13.50 |
| 25 | 0.08992 | 14.08 |
| 28 | 0.09167 | 13.87 |
| 31 | 0.08606 | 14.13 |
| 34 | 0.08516 | 14.07 |
| 37 | 0.08430 | 13.98 |
| 40 | 0.08376 | 13.98 |
| **40** | **0.08376** | **13.98** |

## 2. Document classifier

MiniLM-L6-v2 (**frozen**, 22,713,216 params) + trained MLP head (**49,796 params**).

| metric | value |
|---|---|
| **Held-out test accuracy** | **100%** |
| 95% bootstrap CI | [100.0%, 100.0%] |
| Macro F1 | 1.000 |
| Validation accuracy | 96% |
| **5-fold CV mean (20 folds)** | **100% ± 0%** |
| 5-fold CV macro F1 | 1.000 ± 0.000 |
| Test set size | 24 |
| Train / val / test | 126 → 378 augmented / 24 / 24 |
| Training time | 36.6 s (CPU) |

**Per-category test accuracy**

| category | accuracy | n |
|---|---|---|
| banking | 100% | 6 |
| medical | 100% | 7 |
| legal | 100% | 5 |
| general | 100% | 6 |

**Confusion matrix** (rows = truth, columns = prediction)

| | banking | medical | legal | general |
|---|---|---|---|---|
| **banking** | 6 | 0 | 0 | 0 |
| **medical** | 0 | 7 | 0 | 0 |
| **legal** | 0 | 0 | 5 | 0 |
| **general** | 0 | 0 | 0 | 6 |

## 3. OCR recogniser (our CRNN)

**This is the number that answers "did you actually train something that
reads?", and it is the weakest result in the project. It is reported here
before the classifier because the classifier is easy and the recogniser is
hard, and putting the easy one first is how a demo ends up overclaiming.**

Model: **942,166 parameters**, 37-symbol charset, CTC loss, 2-layer
bidirectional LSTM, 3.84 MB as one self-contained ONNX file.

Trained on real word crops from **260 SROIE receipts** (flatbed scans), split
by receipt so no document appears on both sides.

| metric | value |
|---|---|
| **Held-out word accuracy** (17,945 SROIE test words) | **29.30%** |
| Held-out CER | 11.95% |
| Final epoch | 39 of 40 |
| Training time | ~250 s/epoch, CPU, 4 cores |
| Corpus | 52,554 crops from 420 receipts |
| Split (by receipt) | 294 train / 63 val / 63 test |

### Head-to-head: did the bigger corpus help?

Two runs were compared on the **same 17,945 held-out test words**, because
their validation splits differed (39 receipts against 63) and comparing across
those would have meant nothing:

| checkpoint | corpus | word acc | CER |
|---|---|---|---|
| epoch 20 | 30,021 crops / 260 receipts | 26.25% | 15.90% |
| **epoch 39** | **52,554 crops / 420 receipts** | **29.30%** | **11.95%** |

The 75% larger corpus plus real-homography perspective augmentation is worth
**+3.05pp accuracy and −3.95pp CER**. Reproduce with
`bash run.sh py ml/src/compare_checkpoints.py`.

### The fixed-width export is verified, not assumed

`crnn.onnx` takes a fixed 256px input, so crops are right-padded. That is only
safe because a trained CTC model emits the blank symbol in padded regions.
This was checked and it passes:

```
pad  32->256: ok (ref 'RM'         vs padded 'RM')
pad  64->256: ok (ref '1071'       vs padded '1071')
pad 128->256: ok (ref '01143008'   vs padded '01143008')
pad 200->256: ok (ref 'M  C2  0'   vs padded 'M  C2  0')
```

Worth knowing: this gate's condition was **inverted** and it had never run on
any checkpoint — it skipped precisely when the model was good enough for the
check to matter. A mutation test now confirms the gate can fail.

### The domain gap, measured

The same checkpoint (epoch 39), two held-out sets:

| held-out set | what it is | CER |
|---|---|---|
| SROIE test | flatbed scan, even light, square to lens | **11.95%** |
| Zenodo handheld | night photo, plastic sleeve, tilted, glare | **56.5%** |

Both numbers are real and measured on the shipped model. **A 4.7× gap**, and it
is not a scoring artefact — it is the difference between a document scanned
flat and a document photographed by a person. This is the single largest
weakness in the project and the thing to fix first.

### Real hand-photographed receipts — measured, not projected

The full 20-page handheld set, projection-profile line detection, whole-page
transcripts as ground truth:

| metric | value |
|---|---|
| **Detection recall** | **67.9%** ← hard ceiling on line accuracy |
| **Line exact-match accuracy** | **1.8%** |
| **CER** | **56.5%** |
| Lines scored | 275 |
| Lines missed by the detector | 130 |
| Bands invented by the detector | 12 |

**Read the detection recall first.** The line detector finds 67.9% of the text
lines, so line accuracy cannot exceed 67.9% however good the recogniser is. The
bottleneck is **detection, not recognition**.

Slice-based detection was tried to improve this and made it measurably worse
(0.39–0.66 bands per transcript line against 0.79 for plain global projection
— see `app/js/layout.js`). One page, `1002-receipt`, still defeats any single
global threshold: its row profile never falls below 6% ink, so all 26 of its
lines collapse into 2 bands. Fixing this properly needs an adaptive projection
peak-finder or a learned detector, not another constant.

A representative failure, from the error breakdown — the recogniser is reading
real text, just the wrong band:

```
'GREEN FIELD'                      -> 'NVIH'
'Order #: 69923 dine In'           -> '15305 E PACIFIC COAS'
'SUB TOTAL: 5.00'                  -> '99031 MILER LIE'
"Friendly Red's"                   -> 'FRIENDLY RED S'   (correct!)
'He Pho Ga'                        -> 'NO PIO 66'
```

`Friendly Red's` is read essentially correctly, which is why the headline
number is not zero. The band misalignment in line 2 is the detector pairing
the wrong crop with the right label.

### Why we are not quoting a field-accuracy number yet

End-to-end field accuracy cannot be honestly computed for the custom OCR path
yet, because the recogniser is not accurate enough for the number to mean
anything. Quoting it would require a detector well above 85% recall first. The
historical end-to-end figures in section 4 are **Tesseract** results and must
not be presented as ours.

## 4. End-to-end accuracy (LEGACY — TESSERACT PATH, NOT OURS)

The metric that matters: **field accuracy** — the fraction of documents where *every* key field is recovered. All-or-nothing, because a blind user who hears the wrong dosage is worse off than one who hears none.

5 documents per capture profile, seed 123. **Sub-human profiles excluded from the headline** and reported separately below.

Corpus seeder: `zlib.crc32/seedutil-v1`

### Headline

| preprocessing arm | field accuracy | word accuracy | exact dosage | OCR conf |
|---|---|---|---|---|
| raw (OCR the degraded image) | **58%** | 60% | 82% | 68.0 |
| classical (median + unsharp) | **56%** | 59% | 80% | 66.8 |
| restorer (our model) | **62%** | 65% | 84% | 69.7 |
| restorer + CLAHE (production) ⭐ | **62%** | 64% | 84% | 66.0 |

**Restorer + CLAHE vs. the classical baseline: ▲ 6.7% field accuracy** (56% → 62%).

### Per capture profile (field accuracy)

| profile | raw | classical | restorer | restorer |
|---|---|---|---|---|
| studio_clean | 100% | 100% | 100% | 100% |
| handheld_light | 0% | 0% | 0% | 0% |
| handheld_heavy | 0% | 0% | 0% | 0% |
| low_light | 0% | 0% | 0% | 0% |
| hand_shadow | 20% | 20% | 80% | 60% |
| off_axis | 100% | 80% | 100% | 100% |
| glossy_glare | 100% | 100% | 100% | 100% |
| jpeg_social | 100% | 100% | 80% | 100% |
| paper_texture | 100% | 100% | 100% | 100% |
| worst_case ⚠️ | 0% | 0% | 0% | 0% |

⚠️ = `SUB_HUMAN` — a person cannot reliably read this either. Excluded from the headline: worst_case. These are honest failures, not hidden ones.

## 5. On-device payload

| artifact | size |
|---|---|
| `minilm_encoder.onnx` (INT8) | 22348.7 KB |
| `minilm_head.onnx` | 1.0 KB |
| `restorer.onnx` | 3.9 KB |
| **`crnn.onnx` (our recogniser)** | **3840 KB** |
| `tokenizer.json` | 695 KB |
| **total ML payload** | **26880 KB (26.3 MB)** |

**Tesseract and its 52.2 MB of language data are still shipped and still
called by `app/js/pipeline.js`.** Removing them is the single largest payload
win available (~52 MB, a 66% reduction) and is not yet done. The custom
recogniser replaces it in principle and the decoder is verified against the
Python reference, but the browser pipeline has not been migrated, so the
shipped app still depends on Tesseract working.

## 6. Test suite

```bash
bash run.sh test
```

**258 tests pass.** The ones that matter most:

| test file | what it proves |
|---|---|
| `test_ocr_model.py` | CRNN geometry (height exactly 1), timestep count, and that the ONNX export loop equals the fused `nn.LSTM` to 7.45e-08 |
| `test_line_align.py` | the handheld line aligner survives inserted and missing bands, duplicate lines, and total mismatch without silently reporting 0% |
| `test_leakage.py` | no template, document, or augmented twin appears in both train and evaluation sets |
| `test_corpus.py` | the split is stratified and there are no duplicate templates (a real bug in the original corpus) |
| `test_capture.py` | every degradation actually changes the image, is deterministic, and is physically directional |
| `test_model.py` | the network is exactly the identity at init and stays inside the mobile parameter budget |
| `test_evaluate.py` | sub-human profiles never leak into the headline |

Plus a browser-side gate: `app/tools/recognizer_parity.js` (24/24) compares
the JavaScript CTC decoder against the **actual Python** `decode_greedy`,
reading the reference out of the interpreter rather than hardcoding a second
copy of the expected answers, so the two cannot drift together.

