# 24 · Engine head-to-head: Tesseract vs the shipped CRNN

**Same documents, same crops, same ground truth. Only the engine varies.**

The judge question this answers: *"why not just use Tesseract, or Google Lens?"*
The repo's previous answer was measured **with the Tesseract-era recogniser** —
`docs/10_ACCURACY_GAP.md` §"WHICH ENGINE" and `docs/17_OCR_ACCURACY_RESEARCH.md`
§1 record *"field accuracy, shipped engine: **7.4%** (Tesseract arm on the same
generator: **62.2%**)"*. That comparison was recorded before the recogniser was
retrained and before four train/serve contract defects were fixed, so it is
misleading in **both** directions. This page replaces it, and the number it
replaces is the one to stop quoting.

Everything below is produced by one command and one artifact:

```
env -u PYTHONPATH -u VIRTUAL_ENV ./.venv/bin/python ml/src/tesseract_compare.py
env -u PYTHONPATH -u VIRTUAL_ENV ./.venv/bin/python ml/src/tesseract_compare.py --selfcheck   # 18 checks
```

→ `artifacts/engine_head_to_head.json` (every per-document row, both paired
comparisons, the error samples and the limits list).

## 1 · What was measured

| | |
|---|---|
| documents | the app's own 12-document bench: `prescription`, `banking`, `legal` × `studio_clean`, `handheld_light`, `off_axis`, `jpeg_social` (`/tmp/sightline_samples/`) |
| ground truth | the lines the renderer **drew** (`ml/src/capture.py` DOCS via `gt_lines.json`) — 47 distinct lines, 188 scored rows, 30 of the 47 carrying punctuation |
| crop set | lines detected **once** per document on the deskewed grayscale, with the app's own detector (`app/js/layout.js` `detectLines` + `cropForRecognizer`, ported to cv2/scipy). Every engine crops **the same boxes** |
| detection health | 193 bands for 188 GT lines (per-document counts in the artifact) |
| ours | `app/models/crnn.onnx`, 3,785,916 B, 942,166 params, dynamic-width graph, greedy CTC, 37-symbol alphabet, run on `onnxruntime` CPU — the same graph the browser runs in `onnxruntime-web` |
| Tesseract | **verbatim** `tesseract --version`: `tesseract 4.1.1` / `leptonica-1.82.0` (libgif 5.1.9, libjpeg 8d, libpng 1.6.37, libtiff 4.3.0, zlib 1.2.11, libwebp 1.2.2, libopenjp2 2.4.0, Found AVX2/AVX/FMA/SSE); available languages: **`eng`, `osd`** — no other language data is installed |
| Tesseract invocation | CLI, `-l eng`, default OEM (LSTM), `--psm 7` for a line crop and `--psm 3` for a page, `OMP_THREAD_LIMIT=1` |
| metrics | imported, not reimplemented: `ml/src/bench_hackathon.py` (CER on `norm()`ed text) and `ml/src/evaluate.py` (`word_accuracy`, `FIELDS`) — the definitions the app's own bench report uses |

Case is uppercased and whitespace collapsed on **both** sides before a line
match is claimed: our alphabet is 37 symbols with no lowercase and no
punctuation, so scoring raw sentence-case Tesseract output against us would
charge us 26 letters we are not allowed to emit. What that costs is measured
separately (§3, `lnExNrm`).

## 2 · The five arms, and why arm (a) is labelled unfair

| arm | input |
|---|---|
| `ours_c32_raw` | our CRNN on the app's **32 px line crops**, from the raw deskewed page |
| `ours_c32_clahe` | our CRNN on the same boxes from `CLAHE(deskewed)` — the app's own shipped candidate 1 (`app/js/pipeline.js`) |
| `tess_c32_raw` | **Tesseract arm (a)**: the *identical* 32 px pixels `ours_c32_raw` read |
| `tess_nat_raw` | **Tesseract arm (b)**: the same box at **native page resolution** with a small margin (its fair input) |
| `tess_page_raw` | **Tesseract arm (c)**: the whole page, `--psm 3`, its own layout analysis — the drop-in integration, and the row to compare with the app's end-to-end number |

Arm (a) is genuinely unfair to Tesseract and is reported as such everywhere. The
32 px crop is **our model's contract**, including the 320 px width clamp that
horizontally squeezes long lines; Tesseract's page segmentation was built for
page-like input, not for a 32-pixel-tall strip. It is in the table because it is
the *identical-pixels* control: whatever else differs between the engines, the
pixels do not. **Arm (b) and arm (c) are the rows to quote.**

Both engines also get the same deskew (`js/pipeline.js` `estimateSkew`, ±8°
sweep, applied before detection on both profiles' pages), so the `off_axis`
tilt is not being used as a strawman against either.

## 3 · Results — all 12 documents, all 188 GT lines

| arm | line exact | line exact (punct-blind) | word acc | CER | field recovery | punct in output |
|---|---|---|---|---|---|---|
| ours — 32 px crops, raw page | 11.7% (22/188) | 41.0% (77/188) | 61.2% | 0.197 | **83.3%** (20/24) | 0.0% |
| ours — 32 px crops, CLAHE page | 11.7% (22/188) | 41.0% (77/188) | 60.2% | 0.201 | 83.3% (20/24) | 0.0% |
| **tesseract (a)** — identical 32 px crops | 17.6% (33/188) | 23.9% (45/188) | 49.8% | 0.357 | 29.2% (7/24) | 5.2% |
| **tesseract (b)** — native-res line crops | 64.4% (121/188) | 68.6% (129/188) | 75.3% | 0.164 | 70.8% (17/24) | 5.2% |
| **tesseract (c)** — whole page, `--psm 3` | 72.9% (137/188) | 75.5% (142/188) | 80.1% | 0.134 | 79.2% (19/24) | 4.5% |

*`lnExNrm` ignores punctuation and case (both sides `norm()`ed) — the fair
line-level comparison for an engine whose alphabet has neither.*

**Read the aggregate honestly: on this bench Tesseract wins, and it wins on the
metric a judge will reach for first.** Arm (c) reads 72.9% of lines exactly
against our 11.7%, at a CER of 0.134 against 0.197, and it is better on word
accuracy (80.1% vs 61.2%) too. The only column we lead is field recovery, and
we lead arm (c) by **one field out of 24** — that is not a win, it is a tie
within the resolution of a 24-field sample.

The aggregate is also the least interesting view of it. Here is the same data
split by capture profile, which is where the product decision actually lives:

| arm | `studio_clean` | `off_axis` | `jpeg_social` | `handheld_light` |
|---|---|---|---|---|
| ours (raw page) line exact | 10.6% | 8.5% | 14.9% | **12.8%** |
| ours (raw page) word acc / CER | 59.9% / 0.191 | 43.0% / 0.370 | 73.4% / 0.080 | **68.4% / 0.148** |
| ours (raw page) field recovery | 83.3% | 66.7% | 100% | **83.3%** |
| tesseract (b) line exact | 74.5% | 93.6% | 87.2% | 2.1% |
| tesseract (b) word acc / CER | 94.3% / 0.011 | 98.2% / 0.015 | 96.5% / 0.003 | 12.4% / 0.627 |
| tesseract (b) field recovery | 83.3% | 100% | 100% | 0% |
| tesseract (c) line exact | 95.7% | 95.7% | 95.7% | 4.3% |
| tesseract (c) word acc / CER | 99.1% / 0.000 | 99.1% / 0.000 | 98.6% / 0.000 | 23.6% / 0.536 |
| tesseract (c) field recovery | 100% | 100% | 100% | 16.7% |

**The finding is a split, not a sweep.** Tesseract is essentially perfect on the
three compliant profiles (95.7% of lines exactly, CER 0.000–0.015) — a synthetic
clean render, a perspective/rotation render and a JPEG-recompressed render are
exactly what its LSTM plus leptonica binarisation are built for. On the one
profile that models a real hand-held capture (`handheld_light` =
`defocus=0.5, iso=400, light=0.08` in `ml/src/capture.py`) Tesseract **collapses
and we do not**: 12.4% word accuracy and 0% field recovery for arm (b) against
our 68.4% and 83.3%, and arm (c) manages 23.6% / 16.7%. That is a 4–6× gap in
the direction that matters for a phone app, on the profile that is closest to
how a user actually holds a camera.

## 4 · Error profile — what each engine actually gets wrong

### Ours: wins digits, loses punctuation and prose

Structural, by contract:

- **It cannot emit punctuation at all.** The colon in `Account Holder: John
  Smith`, the `-` in `40-11-04`, the `/` in `01/03`, the `.` in `1,204.33` — the
  37-symbol alphabet
  has none of them, so 120 of the 188 scored rows can never be a *string*-exact
  match however right the read is. That is the whole distance between 11.7% and
  41.0% line exact, and it is why `SORT CODE 401104` (digits perfect, field
  recovered) counts as a miss while `Sort Code: 40-11-04` counts as a hit.
- **Oversized prose lines degenerate.** On the clean studio page,
  `Take one capsule three times daily with food` came back as
  `BAK OECAOSDETE IESCSD IT T00`, and `Rx  Amoxicillin 500mg capsules` as
  `RX AMOXICHIN J0UHO CBOSUIES`. Lowercase sentence-case prose in a
  proportional font is out of distribution for a model trained on uppercase
  receipt crops — the recognised **digits inside the same line survive**
  (`PATIENT ANE DOE DOB 04021979`: `J`→`A`, but the date is intact), which is
  exactly why word accuracy (61.2%) and field recovery (83.3%) look far better
  than line exactness (11.7%).
- Alphabet-level confusions dominate: by GT character, the most-missed are
  `A` (150), `E` (144), `R` (136), `T` (128), `S` (123), `O` (122) — the classic
  receipt-font confusables (`J/A`, `F/T`, `S/5`, `O/0`), i.e. the failure mode
  is letter shapes, not layout.
- It never invents punctuation (0.0% of its output characters are outside the
  alphabet) and never emits an empty line: `dropped lines 0` for every profile.
- Line count is not the bottleneck here: 193 detected bands for 188 GT lines.

### Tesseract: near-perfect on compliant input, word salad when the binariser fails

On the clean page (arm b, the honest input) it produced the whole statement
essentially verbatim, with these characteristic errors:

- `Rx  Amoxicillin 500mg capsules` → `RX AMOXICILLIN 300MG CAPSULES`
  — **a dosage misread**: `500mg` → `300mg`. A plausible-word substitution on
  the metric this product is *for*, on a clean scan.
- `Take one capsule three times daily with food` → `... DALLY WITH FOODQ`,
  `Every 6 to 8 hours` → `EVERY 6 TO 8 NOURS`, `Complete the full 7 day course`
  → `COMPLETE THE FULL / DAY COURSE`, `01/03 RENT DD 1100` → `01/03 RENTDD1100_`.
  Its language model produces real-looking words (`DALLY`, `NOURS`), so a wrong
  read still looks like text — there is no signal in the output that says which
  of these is the invention.
- ~5% of its output characters are punctuation; that is by design and is not
  scored against it here.

On `handheld_light` arm (b)/(c) stop being recognisers:

```
GT   Account Holder: John Smith        tess (b)  a act wee et See
GT   Account Number: 40218877          tess (b)  &ccouet Num@er S02 1887?
GT   Sort Code: 40-11-04               tess (b)  Set Code 43-11-06
GT   Opening Balance        GBP 1,204.33   tess (b)  Qeesee Geerce GRP 1D RD
GT   01/03 SALARY CREDIT   GBP 2,900.00    tess (b)  OL) SALAeY CREDT Ger 2 SOO OO
```

The mechanism is the **binarisation stage between the pixels and the net**:
leptonica thresholds a mildly defocused, noisy, underexposed page, the strokes
break up, and the LSTM then decodes plausible English over the broken mask.
Our model is fed the grayscale crop directly, has no thresholding step, and was
trained on degraded real captures, so the same line degrades instead of
collapsing (`ACOUNT NUMBER 40218877`, `SORT CODE 401104`, `0103 SALARY CREDIT
GBP 290000` — typographic damage, digits intact). Arm (a) shows the same
pattern, so it is not an artefact of crop resolution.

### Paired, line by line, on the same GT lines

| comparison | both right | ours only | Tesseract only | both wrong |
|---|---|---|---|---|
| identical 32 px pixels, strict | 10 | 12 | 23 | 143 |
| identical 32 px pixels, punct-blind | 29 | **48** | 16 | 95 |
| ours (CLAHE) vs tesseract (b), strict | 17 | 5 | 104 | 62 |
| ours (CLAHE) vs tesseract (b), punct-blind | 57 | 20 | 72 | 39 |
| ours (CLAHE) vs tesseract (c) page, strict | 20 | 2 | 117 | 49 |
| ours (CLAHE) vs tesseract (c) page, punct-blind | 61 | 16 | 81 | 30 |

Given the *same* pixels and a punctuation-blind comparison, ours takes 48 lines
that Tesseract misses and loses 16 — the crop-set control behaves as expected,
which is the check that the harness is measuring the engine and not the crops.
Against Tesseract's fair inputs we lose that contest on this bench (20 vs 72
line crops, 16 vs 81 pages). Both-wrong is 30–39 lines even against the page
arm: on this bench the residual ceiling is real for everyone.

### The app's own end-to-end number, for context

`artifacts/pipeline_raw_deskew.json` (the shipped browser pipeline — restorer,
then the three-candidate election, then its own detection) reads **45.9% word
accuracy, CER 0.303, 20.8% field recovery** on these same 12 documents
(`ml/src/bench_hackathon.py` definitions). The harness here reads 61.2% /
0.197 / 83.3% with the same model on the raw deskewed candidate. The difference
is **preprocessing, not recognition** — the app ranks `raw`, `CLAHE` and
`CLAHE+Otsu` candidates by `structure_score` and ships the winner, and the repo's
own measurements already report `raw` beating the restorer arms on this bench
(`docs/10_ACCURACY_GAP.md`). That is an inference from two measurements, not
something this lane measured: what this lane proves is that our harness rows are
**not** the app's end-to-end rows, and that the shipped pipeline gives away more
than the engine comparison does.

## 5 · Decision table

| condition | ship | why |
|---|---|---|
| clean flat scan, one language, a vendored engine is acceptable | **Tesseract (b)/(c)** | On the studio/off-axis/JPEG renders it is 95.7% line-exact at CER ≈ 0.00. We do not beat it there and no amount of measurement will make us; quoting an aggregate that pretends otherwise is the cherry-pick this page exists to prevent. |
| offline web/iOS app, one bundle, no vendor engine | **ours** | 3.6 MB ONNX + onnxruntime-web, already vendored. Tesseract means a second WASM/JS runtime **plus its language files** — on the box this was measured on, `eng.traineddata` is 4,113,088 B and `osd.traineddata` another 10,562,727 B — and the build gate fails the build if Tesseract returns to the bundle. |
| one model across clean and degraded captures | **ours** | The `handheld_light` split above: 68.4% vs 12.4% word accuracy (5.5×) and 83.3% vs 0% field recovery, against arm (b). A second engine would buy the clean case and cost the degraded one. |
| structured fields only (dosage, account number, sort code) | **ours + the field policy** | Its alphabet cannot emit punctuation, so `500mg` cannot return as `500 mg.` and `40218877` cannot return as `S02 1887?`; the cost is paid in CER on prose, not on the fields the product extracts. |
| input too poor to read — tell the user | **ours** | Per-line CTC confidence feeds the app's speak/refuse policy (`app/js/field_guard.js`); the direct Tesseract integration measured here has no equivalent gate wired in. |
| arbitrary languages, handwriting, tables, 100+ docs/scan | **Tesseract (or cloud OCR)** | We ship **one** 37-symbol English charset trained from scratch, no punctuation, no lexicon. That is declared scope, not a capability claim. |

## 6 · Limits of this measurement

- **12 synthetic renders, one font, three templates.** Not photographs. A
  render bench is what the repo has for end-to-end documents; the recogniser
  never trained on these renders (its corpus is SROIE + hand-photographed
  receipts), so this is a domain-gap measurement in both directions, and the
  app's template text is also its ground truth.
- **Tesseract 4.1.1 with `eng` only**, default OEM (LSTM). A Tesseract 5, a
  different `traineddata`, `--oem 0`, a character whitelist, `--psm 6`, or a
  lexicon-constrained decode are all different engines and would need a re-run.
  Nothing here generalises across versions.
- **One detector for the line-level arms** (`ours`, `tess_c32`, `tess_nat`), so
  their ceilings are shared and capped by it; the page arm (c) uses Tesseract's
  own layout analysis and is not capped by ours. Detection was healthy on this
  bench (193 bands / 188 GT lines) — that is a property of clean renders and
  will not hold on real photos.
- **Arm (a) is unfair** and is labelled as such in every table it appears in.
- **Case and whitespace are normalised on both sides**, and punctuation is
  reported as its own column rather than scored away; the punctuation-blind line
  column is reported alongside the strict one.
- **The field metric has 24 values over 12 documents** (2 per document). A
  one-field difference is noise; per-profile rates are over 3 documents each and
  are a direction only. No confidence thresholding, lexicon or field-constrained
  decode is applied to either engine — this is raw recognition, not the field
  policy.
- **Our rows are not the app's end-to-end rows** (§4, last block): the app's
  restorer and candidate election are not reproduced here.
- The deskew is a cv2 re-expression of the app's `estimateSkew` (rotation sweep
  instead of the JS shear sweep), not the same code path; the detector is a
  faithful port of `js/layout.js`, not bit-identical to it.

## 7 · What this lane changed, and what it deliberately did not

Added: `ml/src/tesseract_compare.py`, this page, and
`artifacts/engine_head_to_head.json`. **Nothing in `app/` was touched and
Tesseract was not added to the bundle** — the build gate that forbids that is
deliberate, and this is a measurement of a baseline, not a product change.
