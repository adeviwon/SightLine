# JS module boundaries — and the Python each one mirrors

This app is a port, not a reimplementation. Every module names the Python file
it corresponds to, and where behaviour was deliberately changed, the comment
says so and `tools/` contains the test that proves the change was safe.

If the Python and the JS disagree, **the app is wrong, not the test** — that is
the whole point of the vectors in `selftest.html` being copied verbatim out of
`ml/src/`.

---

## Module map

| JS module | Mirrors | Role |
|---|---|---|
| `js/pipeline.js` | `ml/src/evaluate.py`, `ml/src/corpus.py` | The fallback path. Metrics, preprocessing, OCR, understanding. **Must always work.** |
| `js/restorer.js` | `ml/src/model.py` (SightLineNet), `evaluate.py:arm_restorer` / `arm_classical` / `arm_restorer_clahe` | Optional ONNX restoration with a classical fallback. |
| `js/classifier.js` | `ml/src/train_classifier.py` (ClassifierHead, embed_corpus), `ml/src/export_onnx.py` | Optional MiniLM encoder + head with a keyword fallback. |
| `js/tts.js` | (no Python — platform speech) | Sentence-chunked `speechSynthesis`. |
| `js/app.js` | (no Python — DOM) | UI state, capture sources, accessibility wiring. |

---

## `js/pipeline.js` — the fallback that must always work

Zero dependencies beyond a `<canvas>` and tesseract.js. Everything else is an
accelerator layered on top of this, and the app is fully functional if every
optional file is deleted.

### Ported from `ml/src/evaluate.py`

| JS | Python | Note |
|---|---|---|
| `norm(s)` | `norm(s)` | exact |
| `word_accuracy(gt, hyp)` | `word_accuracy` | exact, **including** the quirk that splitting happens on whitespace *before* normalisation — so `word_accuracy("40-11-04", "40 11 04")` is `0.0` in both. |
| `fields_found(text, dt)` | `fields_found` | exact |
| `fields_complete(text, dt)` | the `len(found) == len(FIELDS[dt])` check in `run()` | extracted so the UI and tests can reuse the all-or-nothing rule |
| `exact_dosage_rate` | the `dosage` metric in `run()` | extracted |
| `classify_doc(text)` | `classify_doc` | exact, including the `> 0` guard that falls back to `general` |
| `classify_doc_scores(text)` | the `scores` dict | extracted for explainability |
| `FIELDS` | `FIELDS` | verbatim |
| `CATEGORIES` | `corpus.CATEGORIES` | order is the ONNX head's index map — do not reorder |

### Preprocessing arms — canvas versions of `evaluate.ARMS`

| Python arm | JS | Used by |
|---|---|---|
| `arm_raw` | (input canvas as loaded) | `preprocess` stage 0 |
| `arm_classical` | `median3` + `unsharp` | `restorer.js:runClassical` — the fallback |
| `arm_restorer` | `restorer.js:runONNX` — tiled 64×256, reflect-pad, overlap-average | optional |
| `arm_restorer_clahe` | `restorer.js:armRestorerClahe` | optional |
| `createCLAHE(clipLimit=2.0, 8×8)` | `clahe(canvas, 2.0, 8)` | both paths |
| `cv2.threshold(OTSU)` | `otsu(hist, total)` / `binarize` | both paths |

Two deliberate differences, both documented at the function:

1. **`unsharp` uses separable box blurs, not a true Gaussian.** Two 1-D box
   passes approximate a Gaussian closely enough for text-edge restoration and
   cost O(n) rather than O(n·k²), which is the difference between usable and
   not on a phone. `arm_classical` in Python uses
   `addWeighted(g, 1.8, blur, -0.8, 0)` with `GaussianBlur(sigma=2.0)`; the
   constants (1.8 / 0.8) are preserved.

2. **`assessQuality` and `estimateSkew` were reimplemented, not ported.** The
   Python side has no blur/noise gate — those thresholds live in the desktop
   repo, and the JS values (`lapVar < 100` blurry, `lapVar > 800` noisy, mean
   `< 50` dark) are carried over from the tested SightLine-Mobile port.

### `normalizeOCRText` — two fixes, both proven

The reference implementation had two bugs that silently cost **dosages** — the
one failure mode this app cannot have.

1. **Rule order.** The leading-`S` rule ran before `m9 → mg`. Its lookahead
   needs the token to already end in `mg`, so on raw `S00m9` it could not fire
   and the output was `S00mg` — not a dose. The comment claimed
   `S00m9 → 500mg`; it never did. Fixed by resolving `m9`/`m1` first.

2. **Unreachable de-duplication.** `/\s*\b(mg|ml)\s+\1/` can never match the
   tail of `50mg mg`, because there is no word boundary between the `0` and the
   `m`. Its stated purpose never fired. Simply removing the `\b` would emit
   `50 mg`, which changes the dose token and breaks `fields_found`. Fixed by
   anchoring on the leading digits: `(\d+)\s*(mg|ml)\s+\2 → $1$2`.

`node tools/normalizer_diff.js` proves the change against
`evaluate.FIELDS`: **42 repairs, 0 regressions, 231 inputs unchanged.**

### Field extraction — two deliberate widenings

Both are cases where the reference pattern could not extract a field that
`evaluate.FIELDS` treats as ground truth, verified against `capture.py`'s own
document text.

| Field | Reference | Now | Why |
|---|---|---|---|
| `Patient` | `/(?:patient\|name)/` — case-sensitive, so it never matched the capitalised `Patient:` label | `/(?:[Pp]atient\|[Nn]ame\|[Aa]ccount [Hh]older)/` | a character class, **not** `/i` — `/i` would relax the captured name and turn `Take one capsule` into a person |
| `Case Number` | `(?:no\|number)?` after `case` | `(?:no\|number\|reference\|ref)?` + `[:#.]?` | `capture.py` emits `Case Reference: 2024-CV-00456`, which the old pattern could not extract at all |

Both only widen what is found. Neither changes a value.

---

## `js/restorer.js` — optional

```
evaluate.arm_restorer            restorer.js
------------------------------   ------------------------------------------
th, tw = 64, 256                 TILE_H, TILE_W  (identical)
np.pad(..., mode="reflect")      reflectPadPlane()  (BORDER_REFLECT, not
                                 symmetric — mirror-without-repeat)
out += r;  cnt += 1              out[] / cnt[] accumulate
out / maximum(cnt, 1)            per-pixel divide, then crop the pad
```

`evaluate.py` exports the network with `dynamic_axes` on H and W, so one tile
shape covers the whole image. `runONNX` reuses a single input tensor buffer
across tiles, which is what keeps a 3200px scan from thrashing memory.

`armRestorerClahe` reproduces the production arm: restorer → CLAHE(2.0, 8×8) →
light unsharp → Otsu, and returns `[restored, clahe, binary]` as OCR candidates.

**Degradation:** every failure path returns the classical result and records the
reason in `status()`. `restore()` cannot throw.

### The `globalThis` handle

`restorer.js` and `classifier.js` reach the pipeline through
`globalThis.SightLine`, not a bare `SightLine` identifier. `pipeline.js` is a
`const` in its own script scope, so under CommonJS a bare reference is
genuinely `undefined` even though the module is loaded — which made the
classical fallback silently no-op. Found by running the code, not by reading it.

---

## `js/classifier.js` — optional

| Python | JS |
|---|---|
| `all-MiniLM-L6-v2` (frozen) | `minilm_encoder.onnx` |
| `mean-pool over attention_mask` | `meanPoolL2(h, mask, seq, dim)` |
| `F.normalize(emb, p=2, dim=1)` | same function, L2 block |
| `ClassifierHead` Linear(384→128)+ReLU+Dropout+Linear(128→4) | `minilm_head.onnx` |
| `F.softmax(logits, 1)` | `softmax()` — max-subtracted |
| `corpus.CATEGORIES` | `CATEGORIES` — **the index map** |
| `max_length=256` | `MAX_TOKENS` |

`encode()` implements BERT wordpiece (basic tokenise → greedy longest-match
`##` continuation → `[CLS]`/`[SEP]`) directly from `tokenizer.json`, rather
than pulling in `@xenova/transformers`. That is ~60 lines instead of ~1 MB in
the precache, and the app classifies one document at a time.

`classify()` returns `null` — never throws — when the model, the tokenizer or
the runtime is missing. `null` means "use the keyword classifier", which is a
complete result, not an error.

---

## The availability manifest

Both optional modules read `models/ort.json` **before** touching
onnxruntime-web:

```json
{ "ort": false, "restorer": false, "classifier": false }
```

A stock install ships with all three `false` and makes **zero** network
requests for models. Without this gate, `init()` fires four doomed dynamic
imports on every page load and logs four 404s — the browser smoke test caught
exactly that.

`app/js/app.js` calls `init()` once at startup and auto-ticks the two
"Skip…" checkboxes when the corresponding path is unavailable, unticking them
when it becomes available. The user never has to know which path is running,
but the result panel always reports it (`restorer: onnx`, `MiniLM model` vs
`keyword classifier`).

---

## `js/app.js` — orchestration order

```
1. SightLine.preprocess(source)        unconditional, canvas
2. SightLineRestorer.armRestorerClahe  optional; returns classical on any failure
3. SightLine.runOCR(candidates)        multi-pass PSM 6/3/11, early exit ≥75%
4. SightLineClassifier.classify(text)  optional; null → keywords
5. SightLine.understand(text, override)  identical code path either way
```

Step 5 is the important one: **both** classification routes produce the same
result shape, so nothing downstream can tell whether the model ran. That is
what makes the optional path genuinely optional rather than a special case.

### Accessibility wiring

Two live regions, not one:

- `#live` — `aria-live="polite"`, never mutated. Progress and results.
- `#live-assertive` — `aria-live="assertive"`. Errors only.

`announce()` originally flipped the attribute on a single node, which left it
assertive for every subsequent progress message. Found by the browser smoke
test, which asserted the region was still polite after an error had fired.

Also: all tap targets ≥48px, confidence always rendered as number **and** word,
`prefers-reduced-motion` and `prefers-contrast` honoured, and the recognised
text exposed as a focusable `region`.

---

### The restore mode is chosen by whether the model exists, not by the image

`preprocess(source, maxDim, { restoreMode })` takes one of three modes:

| Mode | Behaviour | Mirrors |
|---|---|---|
| `"adaptive"` *(default)* | noisy → 3×3 median, blurry → unsharp, **nothing** when neither is detected | the desktop port, which assumes a neural restorer covers the rest |
| `"classical"` | median + unsharp, unconditionally | `evaluate.arm_classical` |
| `"none"` | deskew only | — |

`app.js` picks `"classical"` when `restorer.onnx` is absent and `"adaptive"`
when it is present. That is not a stylistic choice — it is measured:

| restoreMode | classification | field recovery |
|---|---|---|
| `adaptive` (no model) | 10/12 | 5/12 |
| `classical` (what ships) | **11/12** | 5/12 |

A moderately blurred capture lands around `lapVar ≈ 133`, just *above* the
`blurry` threshold of 100, so adaptive mode applies no restoration at all and
sends the raw blur into OCR. `arm_classical` has no such gate.

Note the two modes are not always different: on a **noisy** input they coincide
exactly, because adaptive does median+unsharp and so does classical. They
diverge on a merely **blurry** input (adaptive skips the median) and on a clean
one (adaptive is a no-op). `tools/selftest_image.js` asserts all three cases,
because a test that only checked "the modes differ" would pass on the one input
where they legitimately agree and tell you nothing.

### ⚠ There are TWO different quality gates, and they do not agree

This is a real inconsistency, documented here because it is easy to mistake
one for the other.

| | `pipeline.js` `assessQuality` | `gating.js` `degradation_weight` |
|---|---|---|
| metric | `lapVar`, variance of Laplacian on the **full image** | same metric on a **64×256 tile** |
| scale | full-image | ~3.5× smaller — it is a per-pixel statistic |
| threshold | `lapVar < 100` blurry | `SHARP_CLEAN 0.080` / `SHARP_BAD 0.008` |
| unit | full-image lapVar | tile-scale sharpness |
| purpose | decide *whether* to median/unsharp | decide *how strongly* to restore |

`app.js` calls `preprocess(..., {restoreMode: "adaptive"})`, which uses the
first gate, and then `SightLineRestorer.armRestorerClahe(...)`, which uses the
second. They were written at different times against different units, and
neither was calibrated against the other.

Consequence: a capture that `assessQuality` calls *clean* (lapVar 133, above
the 100 threshold) can still be fully restored by the neural path, and one it
calls *blurry* may be passed through untouched by the gate. The measured
`adaptive` row above is exactly this — the `lapVar ≈ 133` case that the neural
path then does restore, which is why `classical` won when no model was
present.

**Not yet reconciled.** The clean fix is to have `assessQuality` call
`gating.weightFromImageData` so there is one gate, one scale, one threshold —
`ml/src/gating.py` and `app/js/gating.js` are already cross-language
parity-tested (30/30 on real capture tiles). Until then, treat the two gates as
independent and do not reason from one to the other.

---

## The honesty gate — `js/app.js`

`preprocess()` only knows image statistics. A capture can look fine
(`lapVar 133`) and still OCR at 26%, and on a prescription that is the
difference between "take 500mg" and "take 300mg".

The pipeline's defence is that it **never guesses a missing field** — an
unreadable dosage is omitted, not invented. The remaining risk is a user
believing a confident-looking card that is quietly incomplete, so `runScan()`
adds explicit warnings:

| Condition | Warning |
|---|---|
| OCR confidence < 0.40 | "Text was hard to read (N% confidence)…" |
| OCR confidence < 0.60 | "Text confidence is only N%…" |
| zero fields extracted | "No key details could be extracted…" |
| keyword classification, confidence < 0.6 | "The document type is uncertain…" |

Measured over the 12 rendered samples, **6 scans warned and 6 did not**, and
the split follows OCR confidence exactly: every scan at ≥ 55% and with fields
was un-warned, every scan at ≤ 55% or with zero fields warned. Verified in
real Chromium by driving `SightLineApp.runScan` — not a copy of it — and
reading the warning back out of the DOM. `tools/warning_gate.js`.

### The invariant, stated precisely

**The summary must never assert a figure absent from the recognised text.**

The first version of that test compared against `fields_found()` and reported
two "fabrications" that were not fabrications: `1200mg` of calcium and `490mg`
are genuinely in those images. `fields_found()` is Python's deliberately strict
all-or-nothing *evaluation metric* — it rejects plenty of figures the app
correctly shows, because it only accepts patterns matching its ground-truth
label conventions. The correct oracle is the OCR text's own DOSAGE entities.
Asserting against the metric would have "fixed" honest behaviour and hidden the
real bug.

---

## Test layers

| Layer | Command | What it proves | Count |
|---|---|---|---|
| Text battery | `node tools/selftest_text.js` | the port matches `evaluate.py` on the Python vectors | 146 |
| Image battery | `node tools/selftest_image.js` | the canvas path works (via `tools/canvas_shim.js`) | 34 |
| Normalizer | `node tools/normalizer_diff.js` | the normalizer fixes > the reference, loses nothing | 273 cases |
| Browser | `node tools/browser_smoke.js` | boots in real Chromium, service worker caches, **works offline**, no console errors, no 404s | 37 |
| Honesty gate | `node tools/warning_gate.js` | low-confidence scans are warned; no dosage is invented | 25 |
| End-to-end | `node tools/e2e_ocr.js` | real OCR on real degraded documents, scored all-or-nothing | 12 documents |

`selftest.html` is the browser-facing version of the first two batteries; the
`tools/selftest_*.js` files execute the same assertions headlessly.

### Why the shim

`tools/canvas_shim.js` implements exactly the Canvas 2-D operations the
pipeline calls — `getImageData`, `putImageData`, `drawImage` with
nearest-neighbour scaling, `fillRect`, `fillText` with a 5×7 bitmap font — and
throws on anything else. A test passing through the shim is exercising the real
`js/pipeline.js` code, not a mock of it, and the 5×7 font means the blur
estimator sees genuine text-like edge structure rather than noise.

---

## Rules for changing this code

1. **The fallback path may not gain a dependency.** If `js/pipeline.js` starts
   needing an optional file, the offline guarantee is gone.
2. **Never reorder `normalizeOCRText`'s rules** without re-running
   `tools/normalizer_diff.js`. The order is load-bearing and the failure is
   silent.
3. **Never reorder `CATEGORIES`.** It is the ONNX head's output index map.
4. **`fields_found` is the safety oracle.** A change that loses a field fails
   the medical case, whatever it does to the string output.
5. **Optional paths return `null`, not errors.** A missing model is a normal
   operating state, not an exception.
