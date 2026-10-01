# SightLine — on-device document reader (PWA)

Point a phone at a prescription, bank statement or lease. SightLine reads the
text, pulls out the key details, and **reads them aloud** — entirely inside the
browser. No account, no server, no network call after first load.

Built for the users a policy address actually names: someone who cannot read a
printed label, or whose vision makes small print unusable. That constraint is
why the whole pipeline is on-device — sending a bank statement to a server to
read it aloud is a privacy problem, not a feature.

---

## What it does

| Stage | What happens | Where |
|---|---|---|
| Capture | Camera, photo library, or PDF (pdf.js, page 1) | `index.html` |
| Preprocess | Downscale → grayscale → blur/dark quality gate → deskew → adaptive restore → tile-CLAHE | `js/pipeline.js` |
| Restore *(optional)* | SightLineNet via ONNX Runtime Web, 64×256 tiles | `js/restorer.js` |
| OCR | tesseract.js 5.x, multi-pass PSM 6/3/11, early exit at ≥75% confidence | `js/pipeline.js` |
| Normalise | OCR-confusion repair (`S00m9` → `500mg`) | `js/pipeline.js` |
| Understand | Classify, extract fields, NER, build a spoken summary | `js/pipeline.js` |
| Classify *(optional)* | MiniLM encoder + head via ONNX Runtime Web | `js/classifier.js` |
| Speak | Platform speech synthesis, sentence-chunked | `js/tts.js` |

The **optional** stages are accelerators. If `models/ort.json` says `false`, or
the files are missing, or ONNX Runtime Web is not vendored, the app uses the
classical path and works exactly the same. Nothing in the required path can
throw its way out of a scan.

---

## Requirements

- Any static file server. No backend, no build step, no npm install.
- iOS 15+ Safari, or Android 8+ Chrome.
- ~42 MB of vendored wasm + language data (this is the offline guarantee).

---

## Serving it locally

Service workers need a real origin. `file://` will **not** work — the browser
refuses to register a service worker there, and so does the camera API.

```bash
cd app
python3 -m http.server 8080
# then open http://<your-computer-ip>:8080 on the phone
```

Binding to `0.0.0.0` is what makes the phone on the same Wi-Fi able to reach it:

```bash
python3 -m http.server 8080 --bind 0.0.0.0
```

> **HTTPS is required for the camera outside `localhost`.** Browsers treat
> `http://192.168.x.x` as an insecure origin, so `getUserMedia` is blocked and
> only the photo/PDF path will work. Options, easiest first:
>
> 1. **Use the photo path** — "Choose a Photo or PDF" works over plain HTTP.
>    This is the quickest way to try the app on a LAN.
> 2. **Tunnel to HTTPS** — `cloudflared tunnel --url http://localhost:8080` or
>    `ngrok http 8080`, then open the https URL. The service worker installs
>    normally and the camera works.
> 3. **Self-signed cert** — pass one to `http.server`:
>    `python3 -m http.server 8443 --bind 0.0.0.0 --cert cert.pem --key key.pem`,
>    then accept the certificate warning on the phone once.

---

## Installing on an iPhone (Safari)

1. Open the app's URL in **Safari** (not Chrome — iOS only installs PWAs from
   Safari).
2. Tap the **Share** button (square with an arrow pointing up).
3. Scroll down and tap **Add to Home Screen**.
4. Confirm the name **SightLine** and tap **Add**.
5. Launch SightLine from your home screen. It opens full-screen with no browser
   chrome.

The first launch caches all 28 assets. The header badge reads
`OFFLINE-READY — 28 FILES CACHED` when it is done. After that, airplane mode
changes nothing.

## Installing on Android (Chrome)

1. Open the app's URL in **Chrome**.
2. Either tap **Install app** in the address bar / bottom sheet, or
   **⋮ → Add to Home screen → Install app**.
3. Confirm. SightLine launches as a standalone app.

The `file_handlers` block in `manifest.json` also lets Android offer SightLine
as a handler for shared images and PDFs.

---

## Verifying it works offline

After installing and letting the badge report the cache is populated:

1. Turn on **Airplane Mode** (iOS) or **Airplane mode** (Android).
2. Fully close SightLine and relaunch it from the home screen.
3. Scan a document.

It should behave identically. The service worker is cache-first and
same-origin only, so there is nothing to fall back to and nothing to fail.

The automated equivalent is `node tools/browser_smoke.js`, which loads the app
in a real Chromium, cuts the network at the protocol level, and asserts the app
still boots.

---

## Self-test

`selftest.html` runs the browser port against the **same vectors the Python
side uses** — the document text is copied verbatim from `ml/src/capture.py`,
and the ground-truth fields from `ml/src/evaluate.py`. If the two
implementations disagree, the test fails, which is the point.

Open `selftest.html` in a browser. It reports PASS/FAIL per assertion.

Two batteries:

- **Text** (auto-runs on load) — the metrics ported from `evaluate.py`:
  `norm`, `word_accuracy`, `fields_found`, `classify_doc`; the OCR-confusion
  normalizer; field extraction; NER; garbage rejection; robustness against
  `null`/`undefined`/empty input.
- **Image** (press *Image processing only*) — the canvas path, which needs a
  real 2-D context: quality assessment, deskew, median, unsharp, CLAHE, Otsu,
  and the restorer's classical fallback.

Both batteries also run headlessly:

```bash
node tools/selftest_text.js     # 146 assertions
node tools/selftest_image.js    #  26 assertions (uses tools/canvas_shim.js)
node tools/normalizer_diff.js   # proves the normalizer fixes > the reference
node tools/browser_smoke.js     # 31 checks in real Chromium, incl. offline
```

### A note on the normalizer

`node tools/normalizer_diff.js` exists because the reference normaliser had two
bugs that silently cost dosages — the exact failure mode this app must not
have. It scores the old and new implementations against
`evaluate.FIELDS` and fails if any change loses a field. Current result:
**42 repairs, 0 regressions, 231 inputs unchanged.**

---

## Enabling the optional ONNX models

ONNX Runtime Web **is** vendored (`app/vendor/ort/ort.min.mjs` +
`ort-wasm-simd-threaded.wasm`), so `models/ort.json` currently reads:

```json
{ "ort": true, "restorer": false, "classifier": false }
```

`ort: true` means the runtime is present. `restorer`/`classifier` are `false`
because no model files are bundled, so `init()` returns before importing
anything — **zero** network requests for models, and the classical path runs.
This is also why the runtime is deliberately *not* precached: nothing loads it.

To turn the neural path on:

1. Export the models:
   ```bash
   cd ..                       # project root
   bash run.sh export          # writes models/onnx/*.onnx
   ```
2. Copy them into the app:
   ```bash
   cp ../models/onnx/restorer.onnx        app/models/
   cp ../models/onnx/minilm_encoder.onnx  app/models/
   cp ../models/onnx/minilm_head.onnx     app/models/
   cp ../models/onnx/tokenizer.json       app/models/
   ```
3. Set `restorer` and/or `classifier` to `true` in `app/models/ort.json`.
4. **Add the model files to `PRECACHE` in `app/sw.js` and bump `VERSION`.**
   This step is not optional. The runtime is currently un-precached on purpose,
   so flipping a flag without doing this gives you a neural path that silently
   works online and fails offline — the exact failure this app exists to avoid.
5. Verify with `node tools/precache_check.js` (every entry must exist) and
   `node tools/browser_smoke.js` (the offline reload must still pass).

Never load ONNX Runtime Web from a CDN — that breaks the offline guarantee.

The in-app checkboxes (*Skip neural restoration*, *Skip model classifier*)
auto-tick themselves when the corresponding path is unavailable, and untick
once it is.

---

## Layout

```
app/
├── index.html          UI: capture, progress, results, speech
├── selftest.html       PASS/FAIL per assertion, in the browser
├── manifest.json       PWA manifest (standalone, icons, file handlers)
├── sw.js               cache-first service worker, 28-entry precache
├── js/
│   ├── pipeline.js     metrics port + preprocess + OCR + understanding
│   ├── restorer.js     ONNX inference, tiled 64×256; classical fallback
│   ├── classifier.js   MiniLM encoder + head; keyword fallback
│   ├── tts.js          sentence-chunked speech synthesis
│   └── app.js          DOM controller and accessibility wiring
│   └── ORCHESTRATION.md  which Python file each module mirrors
├── models/ort.json     availability manifest for the optional models
├── assets/             icons (180/192/512)
├── vendor/             tesseract.js + wasm cores, eng.traineddata.gz, pdf.js
└── tools/              headless test harnesses
```

`vendor/` is **copied, never installed** — no npm install, no CDN, no network at
runtime. See `js/ORCHESTRATION.md` for the module-to-Python mapping.

---

## Accessibility

- Every control is a real `<button>`/`<input>` with an accessible name; all
  tap targets are ≥48px (WCAG 2.5.5).
- Two live regions: progress and results are **polite**; errors are
  **assertive**. They are separate nodes because flipping `aria-live` on one
  node leaves it escalated for every later message.
- Confidence is always shown as a number *and* a word (*high* / *moderate* /
  *low*) — never colour alone.
- Body text is 18.1:1 contrast on the default theme; `prefers-contrast: more`
  and `prefers-color-scheme: light` are both honoured.
- `prefers-reduced-motion` disables the scan-line sweep and transitions.
- The recognised text is a focusable `region` so a screen-reader user can
  review it without hunting.

## Privacy

The service worker intercepts same-origin `GET` requests only and passes
cross-origin requests straight through without storing them. There are no
analytics, no fonts from a CDN, and no telemetry. After the first load the app
works indefinitely with the network off.

## Licence

MIT — see the repository root.
