# Architecture

How a photo becomes speech. Every stage below runs **on the device** — there
is no server, no API call, and no network request after the app is installed.

---

## 1. The pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT                                                              │
│   camera frame · gallery image · PDF page · camera roll            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│ STAGE 1 — RESTORE                    ml/src/model.py · SightLineNet│
│                                                                     │
│   Purpose: undo what the camera did to the document.                │
│                                                                     │
│   camera frame                                                        │
│      │  motion blur (directional — shaky hand)                      │
│      │  defocus (circular bokeh — close focusing distance)          │
│      │  Poisson-Gaussian sensor noise (scales with ISO)             │
│      │  hand shadow (hard edge — the hand holding the phone)        │
│      │  perspective (off-axis camera pose)                           │
│      │  JPEG (the share sheet's compression)                        │
│      ▼                                                                │
│   SightLineNet (PyTorch → ONNX, 126,097 params)                     │
│      Conv3x3(1→48)+ReLU                                              │
│      5 × [Conv3x3(48→48)+BatchNorm+ReLU]                             │
│      Conv3x3(48→48)+ReLU → Conv3x3(48→1)                           │
│      output = input + residual      ← zero-init, starts as identity │
│      ▼                                                                │
│   CLAHE contrast equalisation + Otsu binarise (OpenCV)              │
│                                                                     │
│   Output: an image Tesseract can read.                              │
│   Confidence: restoration gain measured as PSNR vs. the clean render │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│ STAGE 2 — READ                                    Tesseract (WASM)   │
│                                                                     │
│   Purpose: pixels → text.                                            │
│                                                                     │
│   Tesseract 5, --oem 3 (LSTM), multi-pass PSM 6 → 3 → 11.           │
│   Emits per-word confidence. Runs entirely in WASM — no network.    │
│                                                                     │
│   Confidence: mean word confidence from Tesseract itself            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│ STAGE 3 — UNDERSTAND                                                 │
│                                                                     │
│   3a. OCR-confusion normaliser     "S00mg" → "500mg"                │
│   3b. Document classifier         MiniLM-L6-v2 (frozen) + MLP head  │
│                                 → banking │ medical │ legal │ general│
│   3c. Field extraction            dosages, amounts, account numbers, │
│                                    IBANs, sort codes, dates, clauses │
│   3d. Named entities              drugs, money, dates, organisations │
│                                                                     │
│   Confidence: softmax over the 4 classes + field-presence check      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│ STAGE 4 — SPEAK                                 Web Speech API / TTS  │
│                                                                     │
│   A plain-language summary is composed from the extracted fields and  │
│   spoken:                                                            │
│     "This is a prescription. Amoxicillin, 500 milligrams, one capsule  │
│      three times daily, for seven days."                            │
│                                                                     │
│   IF CONFIDENCE < THRESHOLD → do not guess. Say:                    │
│     "I could not read this clearly. Please retake the photo with     │
│      more light and hold the phone steadier."                        │
└─────────────────────────────────────────────────────────────────────┘
```

**The confidence gate is the most important line in the file.** A blind user
who hears "take 400 milligrams" when the label says 500 is worse off than one
who is asked to retake the photo. Accuracy is a safety property here, not a
vanity metric.

---

## 2. Why offline is an architectural requirement, not a feature

Every existing alternative uploads the document:

| App | Where the OCR runs | Privacy consequence |
|---|---|---|
| Google Lens | Google servers | Document image leaves the device |
| Microsoft Seeing AI | Azure servers | Same |
| Be My Eyes | Volunteer's phone (remote volunteer sees the image) | A stranger sees your medical document |
| **SightLine** | **The user's own phone** | **Nothing is transmitted. Ever.** |

This matters most for exactly the documents SightLine targets — prescriptions
(reveals medical conditions), bank statements (reveals financial distress),
legal correspondence (reveals disputes).

### The offline guarantee is verifiable in three ways

1. **Read the source.** `app/sw.js` precaches every asset and never falls
   back to the network after install.
2. **Break the network.** Turn on airplane mode. The app keeps working.
3. **Capture the proof.** macOS: `sudo tcpdump -i en0 | grep -i sightline`
   during a scan produces no packets. Show this on a slide.

There is no service worker update path, no analytics, no telemetry, no crash
reporting. That is a feature.

---

## 3. Module map

| Component | Python (training/eval) | JavaScript (device) | Model artifact |
|---|---|---|---|
| Capture simulation | `ml/src/capture.py` | — | — |
| Corpus + augmentation | `ml/src/corpus.py` | — | — |
| Restorer | `ml/src/model.py` | `app/js/restorer.js` | `restorer.onnx` |
| Classifier | `ml/src/train_classifier.py` | `app/js/classifier.js` | `minilm_encoder.onnx`, `minilm_head.onnx` |
| OCR | `pytesseract` (eval only) | `tesseract.js` | `eng.traineddata` |
| Field extraction | `ml/src/evaluate.py` | `app/js/pipeline.js` | — |
| Evaluation | `ml/src/evaluate.py` | `app/selftest.html` | — |

Each JavaScript module mirrors a Python file. The mapping is documented in
`app/js/ORCHESTRATION.md`.

---

## 4. The fallback ladder

No single dependency is load-bearing. Each stage degrades instead of failing:

```
restorer.onnx present?  ──yes──► SightLineNet inference
        │ no                          │ fails at runtime
        ▼                             ▼
classical median+unsharp  ──► CLAHE+Otsu  ──► raw image
                                                │
minilm_encoder.onnx present? ──yes──► ONNX classifier
        │ no
        ▼
keyword-density classifier  ──► "general" (safe default)
```

The app **never** shows a broken state. If every model file is missing, it
still reads documents with classical preprocessing and keyword
classification. The models make it much better; they are not required for it
to function.

---

## 5. Data flow and memory budget

A 12 MP iPhone frame is ~48 MB as raw RGB. The pipeline avoids ever holding
that:

```
camera frame (48 MB)
  └─► downscale to 1600px long edge      (~3 MB)
        └─► grayscale                     (~1.5 MB)
              └─► 64×256 tiled inference  (restorer, ~4 KB/tile)
                    └─► restore + binarise
                          └─► OCR text string (a few KB)
```

Peak resident memory for the ML path is under 40 MB. Both ONNX models total
under 1.5 MB, so the whole app — OCR engine, language data, both models, UI —
fits comfortably in a phone browser's cache.

---

## 6. Threading

| Work | Thread | Why |
|---|---|---|
| Camera capture | main | browser requirement |
| Restoration | Web Worker | keeps the UI at 60fps during a scan |
| ONNX inference | Web Worker (WASM, threaded) | `ort.env.wasm.numThreads = min(4, cores-1)` |
| Tesseract | its own WASM worker | already worker-based upstream |
| Speech | main | browser requirement |

A 1600px image restores in well under a second on a modern phone, so even on
the main thread the UI would survive — but the worker is what makes it feel
instant.

---

## 7. What happens when it fails

| Failure | Detection | Response |
|---|---|---|
| Image too blurry | Stage-1 PSNR gain below threshold | "Hold the phone steadier and retake" |
| Too dark | Mean luminance below threshold | "Move somewhere with more light" |
| Low OCR confidence | Tesseract mean confidence below threshold | "I could not read this clearly — retake" |
| Unknown document type | Softmax max below threshold | Reads it anyway, but says "I am not sure what kind of document this is" |
| Wrong field | Field regex fails to parse | That field is **omitted from the spoken output**, never guessed |
| Model file corrupt | ONNX load throws | Silent fallback to classical path |

That last column is the contract: **the app degrades, it never guesses.**
