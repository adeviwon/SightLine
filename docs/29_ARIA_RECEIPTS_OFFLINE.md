# Aria receipt integration and offline diagnosis

24 October 2026. Implemented in `<repo>`, branch
`codex/ppocr-ui`. Nothing pushed or deployed. Aria Reader's application/source
and the original SightLine checkout were preserved.

## What was merged

SightLine already had Aria-derived PP-OCRv5 integration. Its default remains
**PP-OCRv5 mobile FP32, ONNX Runtime Web 1.20.1**, using the exact detector,
recognizer and dictionary hashes in `app/models/ppocr-provenance.json`.
The original trained CRNN is still selectable. No model weights or runtime
binaries changed in this fix.

Aria's shipping default is **Tesseract.js 7.0.0**, whose live core reports
`5.1.0-288-g2a9c1`. Testing the same real receipt showed no reason to replace
SightLine's stronger PP reader with Tesseract. Instead we brought over Aria's
pure weighted document classifier, receipt extraction, taxonomy and 0.75 safety
floor. `app/js/aria/README.md` records source provenance. Graphify's supported
AST extraction/query traced Aria's `performPipeline → extractAll → classifyDocument`
relationships; its output is under Aria's `artifacts/receipt-offline-audit/`.
This was a targeted integration, preserving the existing Taste-designed UI;
no new visual redesign or unverified third skill is claimed.

## Receipt root cause and real test

SightLine's classifier has only banking, medical, legal and general classes.
The UI passed its opinion directly into extraction. Receipt evidence could
therefore never select receipt fields, regardless of OCR confidence. MiniLM's
fixed category indices remain unchanged: the new receipt adapter runs after
recognition/medical safety processing and supplies a separate receipt result.
Strong statement-of-account/opening-and-closing-balance evidence prevents bank
statements from being overridden merely because they contain totals and shops.

The actual public scanned receipt is SROIE **X00016469612**, 463×1013, from
[jsdnrs/ICDAR2019-SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE),
revision `bffe40c26759f3376ec2b3ae9031dbba54cd587c`, train row 0.
The image, public dataset row, reference text and entities are saved in
`artifacts/aria-merge/`. This is a real receipt scan, not a generated preview.
It is not the user's exact image, which was not supplied.

| Same receipt, same Chromium environment | SightLine PP-OCRv5 | Aria Tesseract |
|---|---:|---:|
| Live OCR confidence | 88.46% | 71% |
| Page CER | 23.09% | 29.85% |
| Baseline scan time | 2628 ms | 1609 ms |
| Baseline classification | banking, no fields | receipt |

CER uses uppercase, whitespace-normalised text in the dataset's word order.
One image is **not** a general accuracy benchmark, and confidence is not accuracy.
The previously reported 94% cannot be attributed to a fixture or engine without
that original scan. This reproduction obtained its confidence from live model
inference, not a constant or preview fixture.

Before: MiniLM probabilities banking .3224 / medical .2067 / legal .2194 /
general .2515. After: Aria weighted scores **receipt 7, prescription 0**;
recognised category **receipt**. Shop heading and cash payment render as cards.
Date is omitted: PP text contains `25/12/20188`. Total is deliberately omitted:
the amount reads `9.00`, but the associated word `Total` scored **.72784**, below
.75. No field is fabricated to match the ground truth. Full recognised text and
source image remain available. Structured values are candidates to verify, not
a guarantee of correct recognition.

The adapter rejects unlabelled phone-like barcodes, conflicting labelled totals,
unclear source lines, unknown confidence and low local confidence. Decimal and
currency spelling remain literal; missing currency is not invented. `fieldText`
exposes complete clear lines for receipt cards while prescription `speechText`
retains the existing whole-document completeness requirement. Medical extraction,
dose association and read-aloud gates were not relaxed. The receipt-specific
reminder replaces pharmacist wording on receipt results.

Exact public recognised text, local word confidences, classifier probabilities,
weighted scores, accepted fields and rejection reasons are in
`sightline-before.json`, `sightline-final.json` and `aria-before.json`.

## Offline root cause and fix

The exact string **“Offline saving is unavailable”** comes from
`app/js/app.js → registerSW()` when `serviceWorker` is absent from `navigator`.
It is different from **“Offline saving did not finish”**, emitted on registration
or required cache-install failure. The native WKWebView did not configure the
app-bound service-worker route, yet the UI assumed browser cache availability
was necessary for offline reading. The native application already bundles its
models, WASM, workers and PDF code and serves them over a secure loopback origin.
Service-worker availability is a separate WebKit capability; see
[WebKit's app-bound domain documentation](https://webkit.org/blog/10882/app-bound-domains/).

`ios/Sources/OfflineAssets.swift` now reads the PWA's actual required asset list
and verifies **every entry is readable and nonempty** before starting the reader.
Missing files fail explicitly with a reinstall message. Only a verified bundle
receives a read-only, main-frame marker scoped to its exact loopback origin.
The web controller uses that verified native installation as its offline source.
It does not pretend that a service worker registered or that browser storage
succeeded. No entitlement workaround, unstable fixed port or duplicate native
model cache is needed. Native bundled files are available on every new launch.

Browser installation still requires successful service-worker/cache verification.
The cache version is now **v1.8.3-receipts**, and the adapter plus three imported
Aria modules are required precache assets. Required asset failures still reject
installation and produce a warning/retry; normal browser warnings were not
suppressed. Browser secure contexts remain necessary: localhost/127.0.0.1 or
HTTPS; arbitrary LAN HTTP does not provide the same APIs.

46 precache entries exist. Required bytes are approximately **38.78 MiB**,
already above the historical 25 MiB goal before these fixes. That release blocker
is not solved here; no lossy weight replacement was made to hide it. Optional
MiniLM/restorer files are separate. Storage failures continue to report failure.
The previous legacy CDP profile cache-add failure was not rerun in this fix;
its historical limit remains separate from these successful fresh-context tests.

## Measured offline execution

Fresh browser contexts installed online **before any scan**, switched offline,
reloaded, then scanned the actual receipt twice with PP-OCRv5:

| Browser | First complete scan | Warm complete scan | First OCR inference | Warm OCR inference |
|---|---:|---:|---:|---:|
| Chromium | 2280 ms | 1740 ms | 1771 ms | 1631 ms |
| Playwright WebKit | 2219 ms | 1650 ms | 1683 ms | 1570 ms |

These are end-to-end/inference timings, **not isolated model initialisation
measurements**. No memory saving or significant speed improvement is claimed.
Both produced receipts with the same conservative omission policy and no page
errors. JSON and screenshots are in `chromium-offline.*` / `webkit-offline.*`.

A separate **native macOS WKWebView** used the production loopback server and
bundle verifier, a fresh nonpersistent store, and a compiled content rule list
blocking external HTTP(S), while permitting only loopback bundle requests.
A no-CORS external fetch failed and the actual PP-OCRv5 receipt scan succeeded
with “Ready offline”. This verifies the native bundle strategy; it is **not a
physical-iPhone or iOS-Simulator execution claim**. The native verifier's positive
and missing-file negative controls are saved alongside the logs. iOS simulator
SDK compilation passes; physical-device airplane-mode, camera and speech remain
unverified. Xcode was opened on `ios/SightLine.xcodeproj`.

## Validation and reproduction

Set development-only `PW_CORE=/tmp/sightline-ui-tools/node_modules/playwright-core`,
`CHROME_BIN=/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`, and
`AXE_CORE=/tmp/sightline-ui-tools/node_modules/@axe-core/playwright` on this Mac.

- `node --test app/tools/receipt_regression.test.mjs app/tools/ppocr_contract.test.mjs`:
  **20/20**. Receipts, prescriptions, unrecognised text, bank statements, local
  confidence, unknown confidence, barcodes and conflicting totals are covered.
- `bash run.sh test-ui`: **88/88**, including actual OCR, responsive layouts,
  accessibility, cancelled/stale scans, runtime recovery and cache failure/retry.
- `bash run.sh test-offline-receipt`: **both browser engines pass**, four real
  receipt scans after offline cold reload.
- `bash run.sh test-native-offline`: **pass**, real native WKWebView receipt
  inference with external requests blocked and verified installed assets.
- `bash run.sh bundle-check`: **pass**, every shipped model loads and runs.
- `node app/tools/precache_check.js`: **46 entries, zero missing**.
- `node app/tools/ios_compat_scan.js`: **zero Safari parse-time blockers**.
- `xcodebuild -project ios/SightLine.xcodeproj -scheme SightLine -sdk iphonesimulator
  -configuration Debug -derivedDataPath /tmp/sightline-merge-build
  CODE_SIGNING_ALLOWED=NO build`: **BUILD SUCCEEDED**.
- `bash run.sh test`: **blocked**, shared existing venv has no `pytest`.
  No Python suite pass is claimed and the shared Aria environment was not modified.

The first wider browser regression found the bank-statement misclassification.
The guard was added, a regression test was added, and the unchanged browser
assertions were rerun. The final unchanged browser assertions passed **69/69 WebKit and 69/69 Chromium**
using `bash run.sh ios-full --engine=both '--suites=boot,scan,refusal,offline,cold install'`.
This is the selected relevant regression battery, not the entire historical suite.

## Screenshots and changed files

Real rendered results were opened and inspected at 320px and 1440px with no
horizontal clipping. Screenshots are not painted mockups:

- `artifacts/aria-merge/sightline-before.png`: original real-receipt failure.
- `artifacts/aria-merge/sightline-final.png`: 390px corrected receipt result.
- `artifacts/aria-merge/receipt-final-320.png`: narrow full-scroll result.
- `artifacts/aria-merge/receipt-final-1440.png`: desktop result.

Important changes: `receipt.mjs` + `aria/*` add the document adapter; `ppocr_core.js`
returns safe complete-line text without changing inference or medical speech;
`app.js` connects receipt results, diagnostics and verified native offline state;
`index.html` clarifies recognition confidence; `sw.js` versions/caches the new
modules; `OfflineAssets.swift` and `SightLineApp.swift` validate and expose bundled
offline availability. `WebAssetServer.swift` only corrects its capability comment.
New receipt/browser/native audit tools, runner targets and this handoff make the
work reviewable. Existing redesign, model files and unrelated unfinished work
were preserved.
