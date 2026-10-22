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

## 0 · Two fixture generations — read this before any table below

The bench's 12 documents are **renders**, produced from the capture profiles in
`ml/src/capture.py`. Those profiles were re-tuned on 2026-10-22, so the images
were regenerated. The first head-to-head was measured **2026-10-21 23:18** on the
*previous* generation and is kept, unedited, at
`artifacts/engine_head_to_head_oldfixtures.json`. Every row in **§3 onwards is a
re-measurement on the current generation**, confirmed by
`/tmp/sightline_samples/manifest.json` `profiles_fingerprint = 0174a59e2608`,
which equals `capture.profiles_fingerprint()`.

**The regeneration changed one profile.** Each Tesseract arm's text is
**byte-identical** across the two generations on `studio_clean`, `off_axis` and
`jpeg_social`, and differs on the three `handheld_light` documents **only**
(9 of 12 documents unchanged, all three arms). So those three profiles are the
same pixels, and only `handheld_light` was re-rendered — which is exactly the
profile the previous version of this page rested its conclusion on.

**This run also changed our engine.** `app/models/crnn.onnx` was retrained and
re-exported (mtime 2026-10-22 12:15, i.e. *after* the previous artifact), and our
arms' text changed on **12/12** documents — including the three profiles whose
pixels are provably unchanged. So:

| what moved | mechanism | separable? |
|---|---|---|
| every Tesseract row | the re-rendered `handheld_light` profile; nothing about Tesseract changed (same 4.1.1, same CLI, same flags) | yes — pinned by the three pixel-identical profiles |
| every "ours" row | the fixture correction **and** a retrained model | not from this harness alone — §3a prices the model on the three unchanged profiles |

Never read a row that moved here as an engine improvement. Two things changed
and only one of them is an engine.

## 1 · What was measured

| | |
|---|---|
| documents | the app's own 12-document bench: `prescription`, `banking`, `legal` × `studio_clean`, `handheld_light`, `off_axis`, `jpeg_social` (`/tmp/sightline_samples/`) |
| fixture generation | **current** — re-rendered 2026-10-22 12:09, `profiles_fingerprint 0174a59e2608` = `capture.profiles_fingerprint()`; the previous generation's results are kept at `artifacts/engine_head_to_head_oldfixtures.json` |
| ground truth | the lines the renderer **drew** (`ml/src/capture.py` DOCS via `gt_lines.json`) — 47 distinct lines, 188 scored rows, 30 of the 47 carrying punctuation |
| crop set | lines detected **once** per document on the deskewed grayscale, with the app's own detector (`app/js/layout.js` `detectLines` + `cropForRecognizer`, ported to cv2/scipy). Every engine crops **the same boxes** |
| detection health | 193 bands for 188 GT lines (per-document counts in the artifact) — identical count in both generations |
| ours | `app/models/crnn.onnx`, 3,785,916 B, 942,166 params, dynamic-width graph, greedy CTC, 37-symbol alphabet, run on `onnxruntime` CPU — the same graph the browser runs in `onnxruntime-web`; **the retrained export in this generation** |
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

## 3 · Results — current fixtures (generation of 2026-10-22), all 12 documents, all 188 GT lines

| arm | line exact | line exact (punct-blind) | word acc | CER | field recovery | punct in output |
|---|---|---|---|---|---|---|
| ours — 32 px crops, raw page | 10.6% (20/188) | 48.9% (92/188) | 67.2% | 0.152 | 75.0% (18/24) | 0.0% |
| ours — 32 px crops, CLAHE page | 11.2% (21/188) | 48.4% (91/188) | 67.1% | 0.154 | 83.3% (20/24) | 0.0% |
| **tesseract (a)** — identical 32 px crops | 22.9% (43/188) | 31.9% (60/188) | 68.0% | 0.183 | 45.8% (11/24) | 5.0% |
| **tesseract (b)** — native-res line crops | 86.7% (163/188) | 93.1% (175/188) | 96.8% | 0.007 | 95.8% (23/24) | 5.0% |
| **tesseract (c)** — whole page, `--psm 3` | 95.7% (180/188) | 99.5% (187/188) | 98.9% | 0.000 | 100.0% (24/24) | 4.9% |

*`lnExNrm` ignores punctuation and case (both sides `norm()`ed) — the fair
line-level comparison for an engine whose alphabet has neither.*

**Read the aggregate honestly: on this bench Tesseract wins, and it wins on every
column except our punctuation-blind line count.** Arm (c) reads 95.7% of lines
exactly against our 10.6%, at a CER of 0.000 against 0.152, and it is better on
word accuracy (98.9% vs 67.2%) and field recovery (100% vs 75.0%) too. The three
compliant profiles are now read essentially perfectly by both Tesseract arms. We
do not lead any column of this table; the one place we lead a *sub-comparison* is
the identical-pixels, punctuation-blind control (§4), which is a statement about
what our 37-symbol alphabet is allowed to emit, not about who reads better.

### 3a · What moved between the two generations, row by row

`old` = `artifacts/engine_head_to_head_oldfixtures.json` (2026-10-21 23:18,
previous fixtures). `new` = `artifacts/engine_head_to_head.json` (current
fixtures). Word acc / CER / fields:

| arm | word acc old → new | CER old → new | fields old → new | same pixels on 3 of 4 profiles? |
|---|---|---|---|---|
| ours — raw page | 61.2% → **67.2%** (+6.0) | 0.197 → **0.152** | 83.3% → **75.0%** | no — our reads changed on 12/12 docs, so the model changed |
| ours — CLAHE page | 60.2% → **67.1%** (+6.9) | 0.201 → **0.154** | 83.3% → 83.3% | no — same |
| tesseract (a) — 32 px crops | 49.8% → **68.0%** (+18.2) | 0.357 → **0.183** | 29.2% → **45.8%** | yes — identical texts on 9/12 docs |
| tesseract (b) — native-res crops | 75.3% → **96.8%** (+21.5) | 0.164 → **0.007** | 70.8% → **95.8%** | yes |
| tesseract (c) — whole page | 80.1% → **98.9%** (+18.8) | 0.134 → **0.000** | 79.2% → **100.0%** | yes |

| arm | line exact old → new | line exact punct-blind old → new |
|---|---|---|
| ours — raw page | 11.7% (22) → 10.6% (20) | 41.0% (77) → **48.9% (92)** |
| ours — CLAHE page | 11.7% (22) → 11.2% (21) | 41.0% (77) → **48.4% (91)** |
| tesseract (a) | 17.6% (33) → 22.9% (43) | 23.9% (45) → 31.9% (60) |
| tesseract (b) | 64.4% (121) → 86.7% (163) | 68.6% (129) → 93.1% (175) |
| tesseract (c) | 72.9% (137) → 95.7% (180) | 75.5% (142) → 99.5% (187) |

**Why the Tesseract rows moved: the fixtures, on one profile.** Per profile, arm
(b) went `handheld_light` 12.4% → **98.2%** word accuracy (CER 0.627 → **0.000**,
line exact 2.1% → 91.5%, fields 0% → 100%) and arm (c) 23.6% → **98.6%** (0.536 →
**0.000**, 4.3% → 95.7%, 16.7% → 100%), while every other profile is unchanged to
the digit (`studio_clean` 94.3%/0.011, `off_axis` 98.2%/0.015, `jpeg_social`
96.5%/0.003 for arm (b) — the old and new numbers are the same numbers). The old
`handheld_light` profile was the one rendered 17–67× harsher than a real
photograph (`docs/14_HACKATHON_BENCHMARK.md`); the current one is not, and a
Tesseract that is not reading through a broken binarisation mask is a different
engine on that page.

**Why our rows moved: the retrain** (plus that same fixture correction on
`handheld_light`). On the three profiles whose pixels are provably identical, our
raw-page arm moved `studio_clean` 59.9% → 62.8% (+2.9), `off_axis` 43.0% → 49.9%
(+6.9), `jpeg_social` 73.4% → 77.8% (+4.4) — that is the model, and only the
model. On `handheld_light` our arm went 68.4% → 78.4%, which is model **and**
fixtures together. Our field recovery is the one regression: 20/24 → 18/24, from
`prescription/studio_clean` — its only recovered field, `400mg`, is lost, because
`RX IBUPROFEN 400MG TABLETS` now reads `RX IBUOROFEN 400MO TABLES` — and one
field each on `prescription/handheld_light` and `legal/jpeg_social`, against a
gain on `legal/handheld_light` and a swap on `prescription/off_axis`
(`400mg` → `500mg`). Those are clean-pixel losses on `studio_clean`, so they
belong to the retrain, not to the fixtures.

## 4 · Error profile — what each engine actually gets wrong

### Ours: wins digits, loses punctuation and prose

Structural, by contract:

- **It cannot emit punctuation at all.** The colon in `Account Holder: John
  Smith`, the `-` in `40-11-04`, the `/` in `01/03`, the `.` in `1,204.33` — the
  37-symbol alphabet
  has none of them, so 120 of the 188 scored rows can never be a *string*-exact
  match however right the read is. That is the whole distance between 10.6% and
  48.9% line exact, and it is why `SORT CODE 401104` (digits perfect, field
  recovered) counts as a miss while `Sort Code: 40-11-04` counts as a hit.
- **Oversized prose lines degenerate.** On the clean studio page,
  `Take one capsule three times daily with food` came back as
  `BK OME ADSUE IVE IMES AL I1000`, and `Rx  Amoxicillin 500mg capsules` as
  `KX AMOXCICIN UUMO CBOSUIES`. Lowercase sentence-case prose in a
  proportional font is out of distribution for a model trained on uppercase
  receipt crops — the recognised **digits inside the same line survive**
  (`PATIENT JANE DOE DOB 04021979` is now character-perfect), which is
  exactly why word accuracy (67.2%) and field recovery (75–83%) look far better
  than line exactness (10.6%).
- **The retrain traded one clean-page read for others.** `RX IBUPROFEN 400MG
  TABLETS` → `RX IBUOROFEN 400MO TABLES` on `studio_clean`, where the pixels did
  not change at all: a `G`→`O` substitution inside the dosage, costing that
  field. Report it as a regression on that row, not as noise.
- Alphabet-level confusions dominate: by GT character, the most-missed are
  `A` (152), `E` (147), `R` (137), `T` (132), `S` (125), `O` (124) — the classic
  receipt-font confusables (`J/A`, `F/T`, `S/5`, `O/0`), i.e. the failure mode
  is letter shapes, not layout.
- It never invents punctuation (0.0% of its output characters are outside the
  alphabet) and never emits an empty line: `dropped lines 0` for every profile.
- Line count is not the bottleneck here: 193 detected bands for 188 GT lines.

### Tesseract: near-perfect everywhere on the current fixtures, including the hand-held one

On the clean page (arm (b), the honest input) it produces the statement
essentially verbatim, with these characteristic errors:

- `Rx  Amoxicillin 500mg capsules` → `RX AMOXICILLIN 300MG CAPSULES`
  — **a dosage misread**: `500mg` → `300mg`. A plausible-word substitution on
  the metric this product is *for*, on a clean scan, on pixels that did **not**
  change between the two fixture generations.
- `Take one capsule three times daily with food` → `... DALLY WITH FOODQ`,
  `Every 6 to 8 hours` → `EVERY 6 TO 8 NOURS`, `Complete the full 7 day course`
  → `COMPLETE THE FULL / DAY COURSE`, `Maximum 1200mg in 24 hours` →
  `MAXIMUM 1200MGQ IN 24 HOURS`, `Keep out of reach of children` →
  `KEEP OUT OF REACH OF CNILAREN`. Its language model produces real-looking words
  (`DALLY`, `NOURS`), so a wrong read still looks like text — there is no signal
  in the output that says which of these is the invention.
- ~5% of its output characters are punctuation; that is by design and is not
  scored against it here.

**On `handheld_light` the previous version of this page reported a collapse.
That collapse was the fixture, and it is withdrawn here rather than caveated.**
On the *previous* generation, arm (b) read 12.4% of words on this profile and
emitted word salad (`Account Holder: John Smith` → `a act wee et See`,
`Account Number: 40218877` → `&ccouet Num@er S02 1887?`). On the current
generation, the *same three documents re-rendered*, arm (b) reads **98.2%** of
words at CER 0.000 and arm (c) **98.6%**, and every field comes back verbatim:

```
GT   Account Holder: John Smith        tess (b)  Account Holder: John Smith
GT   Account Number: 40218877          tess (b)  Account Number: 40218877
GT   Sort Code: 40-11-04               tess (b)  Sort Code: 40-11-04
GT   Opening Balance        GBP 1,204.33   tess (b)  Opening Balance GBP 1,204.33
GT   01/03 SALARY CREDIT   GBP 2,900.00    tess (b)  01/03 SALARY CREDIT GBP 2,900.00
```

Ours on the same pixels: 78.4% word accuracy, CER 0.073, 83.3% of fields. The
mechanism claimed before was leptonica's binarisation breaking up on a defocused
noisy page, and that is presumably what happened — the point is that the
over-harsh simulate made it happen on a page a camera does not actually produce.
**The 5.5× hand-held advantage this page used to claim was a property of the
simulator. On pixels that model a real capture, Tesseract wins that profile too,
and this is the number to quote from now on.**

### Paired, line by line, on the same GT lines

| comparison | both right | ours only | Tesseract only | both wrong |
|---|---|---|---|---|
| identical 32 px pixels, strict | 14 | 6 | 29 | 139 |
| identical 32 px pixels, punct-blind | 43 | **49** | 17 | 79 |
| ours (CLAHE) vs tesseract (b), strict | 19 | 2 | 144 | 23 |
| ours (CLAHE) vs tesseract (b), punct-blind | 87 | 4 | 88 | 9 |
| ours (CLAHE) vs tesseract (c) page, strict | 21 | 0 | 159 | 8 |
| ours (CLAHE) vs tesseract (c) page, punct-blind | 91 | 0 | 96 | 1 |

Given the *same* pixels and a punctuation-blind comparison, ours takes 49 lines
that Tesseract misses and loses 17 — the crop-set control behaves as expected,
which is the check that the harness is measuring the engine and not the crops.
Against Tesseract's fair inputs we lose that contest on this bench (4 vs 88
line crops, 0 vs 96 pages). Both-wrong is down to 23–9 lines: the residual
ceiling shrank as the pages got realistic, for everyone.

### The app's own end-to-end number, and the gap to this harness

`artifacts/pipeline_raw_shipped.json` (the shipped browser pipeline — restorer,
then the three-candidate election, then its own detection) reads **57.2% word
accuracy, CER 0.217, 33.3% field recovery** on these same 12 documents on the
current fixtures (`ml/src/bench_hackathon.py` definitions). This harness reads
**67.2% / 0.152 / 75.0%** with the same model on the raw deskewed candidate.

So the gap is **10.0 points of word accuracy and 41.7 points of field recovery**,
and its direction is the same as before: the app's own path gives away more than
the engine comparison does. The difference is preprocessing plus the app's
evidence gate (`artifacts/pipeline_raw_deskew.json` was the previous fixture
generation and read 45.9% / 0.303 / 20.8%; the 15.3-point gap this section used
to describe was that generation's, app 45.9% against this harness's 61.2%).

**What this section can and cannot say.** It can say, from two artifacts, that
the harness rows are not the app's end-to-end rows and never were. It cannot say
*which* stage of the app costs the 10.0 points — that is an inference, and the
repo's own A/B work is where it belongs (`docs/10_ACCURACY_GAP.md`, the candidate
list fix, the deskew sign fix). One correction to an earlier claim on this page:
the gap did **not** collapse to ~4 points when the fixtures were regenerated. It
moved 15.3 → 10.0, because both sides moved — the app by +11.3 points
(fixture + retrain) and this harness's raw-page row by +6.0 (retrain, §3a). Only
the app-side number was expected to move; the harness row was assumed to be a
constant, and it is not.

## 5 · Decision table

| condition | ship | why |
|---|---|---|
| clean flat scan, one language, a vendored engine is acceptable | **Tesseract (b)/(c)** | On the current fixtures it is 86.7–95.7% line-exact at CER 0.007–0.000 **on every profile, `handheld_light` included**, and we do not beat it there — quoting an aggregate or a profile that pretends otherwise is the cherry-pick this page exists to prevent. |
| offline web/iOS app, one bundle, no vendor engine | **ours** | 3.6 MB ONNX + onnxruntime-web, already vendored. Tesseract means a second WASM/JS runtime **plus its language files** — on the box this was measured on, `eng.traineddata` is 4,113,088 B and `osd.traineddata` another 10,562,727 B — and the build gate fails the build if Tesseract returns to the bundle. |
| one model that must run offline on a phone with no second runtime | **ours** | This is now a packaging and privacy argument, not an accuracy one: on this bench Tesseract reads better on every profile. What rests on our side is one 3.6 MB graph, no thresholding stage to break, and an inference that never leaves the device. |
| structured fields only (dosage, account number, sort code) | **ours + the field policy** | Its alphabet cannot emit punctuation, so `500mg` cannot return as `500 mg.` and `40218877` cannot return as `S02 1887?`; the cost is paid in CER on prose, not on the fields the product extracts. Tesseract's counter-example is measured and real: `500mg` → `300mg` on a clean page. |
| input too poor to read — tell the user | **ours** | Per-line CTC confidence feeds the app's speak/refuse policy (`app/js/field_guard.js`); the direct Tesseract integration measured here has no equivalent gate wired in. |
| arbitrary languages, handwriting, tables, 100+ docs/scan | **Tesseract (or cloud OCR)** | We ship **one** 37-symbol English charset trained from scratch, no punctuation, no lexicon. That is declared scope, not a capability claim. |

## 6 · Limits of this measurement

- **12 synthetic renders, one font, three templates.** Not photographs. A
  render bench is what the repo has for end-to-end documents; the recogniser
  never trained on these renders (its corpus is SROIE + hand-photographed
  receipts), so this is a domain-gap measurement in both directions, and the
  app's template text is also its ground truth.
- **The bench's own inputs changed between the two published generations, on one
  profile.** §0 and §3a say which rows that affects. Any number on this page
  must be quoted with its generation; the old artifact is kept beside the new one
  for exactly this reason, and `ml/tests/test_fixture_freshness.py` now fails if
  the fixtures drift from the capture profiles again.
- **Our own engine changed between the two generations too** (retrained export).
  The three profiles whose pixels are identical price the model change only;
  `handheld_light` mixes model and fixtures. Nothing here isolates the fixture
  effect inside our own rows.
- **Tesseract 4.1.1 with `eng` only**, default OEM (LSTM). A Tesseract 5, a
  different `traineddata`, `--oem 0`, a character whitelist, `--psm 6`, or a
  lexicon-constrained decode are all different engines and would need a re-run.
  Nothing here generalises across versions.
- **One detector for the line-level arms** (`ours`, `tess_c32`, `tess_nat`), so
  their ceilings are shared and capped by it; the page arm (c) uses Tesseract's
  own layout analysis and is not capped by ours. Detection was healthy on this
  bench (193 bands / 188 GT lines, in both generations) — that is a property of
  clean renders and will not hold on real photos.
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
  restorer, candidate election and evidence gate are not reproduced here.
- The deskew is a cv2 re-expression of the app's `estimateSkew` (rotation sweep
  instead of the JS shear sweep), not the same code path; the detector is a
  faithful port of `js/layout.js`, not bit-identical to it.

## 7 · What this lane changed, and what it deliberately did not

Added: `ml/src/tesseract_compare.py`, this page, and
`artifacts/engine_head_to_head.json`. On the fixture re-measurement, this lane
kept the previous generation as `artifacts/engine_head_to_head_oldfixtures.json`
rather than overwriting the record, and changed no harness logic. **Nothing in
`app/` was touched and Tesseract was not added to the bundle** — the build gate
that forbids that is deliberate, and this is a measurement of a baseline, not a
product change.
