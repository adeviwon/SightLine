# PP-OCR and interface integration — 24 October 2026

**Later frontend work:** [Consumer UI redesign](28_CONSUMER_UI_REDESIGN.md)
supersedes this report's browser-verification and screenshot limitations with
real Chrome/WebKit scans and captures. Its remaining limits are stated separately.

## Scope and provenance

This local implementation is in `<repo>`, branch
`codex/ppocr-ui`. It starts from the GitHub main archive at
`e78c90e34e34ab3a11187b06e1b62bbeb98dfc9d` (local snapshot commit `2aa6ddd`).
The older, dirty `~/sightline` checkout was not modified. Nothing was
pushed, published or deployed. Historical evaluation numbers elsewhere in this
repository describe the original models, not this integration.

The new default **Enhanced reader** uses pretrained PP-OCRv5 mobile, FP32 detector
and FP32 recognizer, Apache-2.0. Exact export revision, sizes, hashes, dictionary
and input/output contracts are in `app/models/ppocr-provenance.json`.
SightLine's existing ONNX Runtime Web **1.20.1** is retained. **Original reader**
preserves the trained CRNN and its previous pipeline. There is no Tesseract.
Engine choice is explicit and stored locally; failure offers retry or Original
reader, without silently substituting engines.

## Measured model results

Evidence: `artifacts/integration/precision-comparison.json` and
`artifacts/integration/native-fp32.json`. These are actual native Node executions
of the shipped ORT Web WASM, weights and adapter on 12 fixed synthetic SightLine
images. The native harness feeds original grayscale pixels; it does **not** test
browser canvas preprocessing, camera capture or the entire browser controller.
No real patient or financial documents were used.

| Measurement | Result |
|---|---:|
| Quantized recognizer mean CER, same 12 images | 79.11% |
| Selected FP32 recognizer mean CER | 50.41% |
| Clean prescription CER | 2.74% |
| Clean banking CER | 2.34% |
| Clean legal CER | 1.38% |
| Latest native runtime module import | 24 ms |
| Latest first model-session preparation | 592 ms |
| Reusing already prepared sessions | 0.004 ms |
| Native recognition, 12 images | 731–1,482 ms |
| Three repeated clean prescription reads | 1,243 / 1,241 / 1,241 ms |
| Literal expected fields present in recognized text | 10 of 24 |
| Expected fields remaining in speech-eligible text | 3 of 24 |
| Expected fields retained by native extraction | 3 of 24 |

CER above is the existing benchmark normalization: uppercase alphanumeric text
with whitespace collapsed. The latest JSON additionally records punctuation-
preserving CER: clean prescription **4.24%**, banking **4.14%**, legal **1.55%**.
Field counts measure literal presence, **not** correct extraction, medical
interpretation, precision or calibrated confidence. Browser extraction accuracy
and a matched CRNN-versus-PP-OCR accuracy comparison remain unverified. The remote
original-controller baseline completed 5 scans and 13 functional checks, but its
scored JSON and screenshots were lost when the remote sandbox reset.

Off-axis and social-JPEG images remain very poor (roughly 87–99% CER). These
results do not establish medicine-label capability, clinical safety, or a general
accuracy percentage. There is no verified real medicine corpus here. Small text,
low light, genuine receipts and real prescription accuracy need additional data.

The native process RSS before the last three repeats was about **252 MiB**;
repeat samples were about **257 / 258 / 252 MiB**. Other runs varied substantially.
These are whole Node process snapshots influenced by garbage collection,
not browser peak memory or proof against leaks. Mobile WebKit memory and UI
responsiveness still require measurement. Session scratch memory patterns are
disabled; every input/output tensor is disposed; partial loads release sessions.
Worker termination cancels timed-out or abandoned inference. Native timing
excludes camera acquisition, preprocessing, classification and rendering; it
must not be described as scan-to-results latency or physical-device cold start.
Session preparation varied from about 403 to 592 ms across runs. The final run
also rejected an invalid input, returned no words/speech at zero confidence on
a real white-image inference (134 ms), then completed three valid reads on the
same sessions. This verifies native invalid-input recovery and a blank negative
control, not recovery from a physical-device runtime crash.

FP32 was selected because the quantized candidate damaged recognition on the
same fixtures. The selected recognizer is 16,517,247 bytes rather than 4,302,156
bytes. The quantized comparison binary is outside the shipped app.

## Safety findings and fixes

The clean prescription contains two medicines. A less confident second medicine
line was omitted from the trusted read while its maximum daily amount remained.
A fragment-based summary could incorrectly label that maximum as another dose.

The enhanced engine now refuses structured fields and spoken summaries when any
detected word/crop fails the existing **0.75** safety threshold. Recognition text
is retained for inspection. A daily maximum appearing only in a limit sentence
is presented with that complete sentence as a warning, not an isolated dosage.
Literal-value checks prevent normalization repairs such as `S00m9` becoming an
asserted `500mg`, or `500` becoming `5.00`. Cantonese receives the same eligible
text; an incomplete read produces a plain-language refusal before translation.

This deliberately conservative policy refuses the clean prescription fixture's
read-aloud despite low page CER. It cannot detect text that the detector misses
entirely and is not a calibrated medical safety guarantee. The medication-reading
release is therefore **not ready to sign off**. Original reader behavior remains
available; its historical limitations still apply.

## Architecture and interface

The path is `app.js → pipeline.runOCR → ocr_engine → ppocr → module worker →
PaddleReader → detector/recognizer`. Recognition executes in a dedicated worker;
the existing optional classifier and restoration paths remain separate. Optional
sessions and camera permission are no longer eagerly requested at startup.
Measured word counts drive progress; unmeasured phases are indeterminate.
Screen changes cannot start an overlapping enhanced recognition request.

The interface keeps SightLine's blue identity and system typography, with a
quiet light reading sheet, dark/high-contrast variants, 20–24 px reading text,
64 px controls, visible focus, reduced-motion handling and semantic result
groups. Results receive focus and scroll into view, the mobile image preview is
hidden after reading, and read-aloud controls move beside the result. Raw text
and technical metadata are separate disclosures. Dynamic low-confidence warning
strings now use plain language. No confidence threshold was lowered.

Skills actually used:

- **design-taste-frontend**: app-screen direction draw, framed-sheet composition,
  typography/spacing rules and scoped source lint (0 failures, 0 warnings).
- **graphify**: structural extraction of the integration path; graph and explain
  output in `artifacts/integration/graphify-out` and `architecture-path.log`.
- **review-agent**: built-in read-only defect-first workflow, which identified
  stale worker errors, literal-field safety and partial-document speech risks.
  Its installation could be verified, but it could **not** be established as the
  user's third recently installed skill. No third recent UI skill was invented.

The Figma SwiftUI routing skill was also read for the iOS work; no Figma
translation was applicable and no Figma file was created.

## Offline and iOS

Service-worker version is `v1.7.0-ppocr`. Required models, worker modules,
dictionary, CSS and runtime are listed. Partial required installs fail, and the
offline badge checks actual cache contents instead of equating online status
with offline readiness. All 42 listed paths exist. Required precache totals
**40,537,586 bytes / 38.66 MiB** at final validation; rerun
`node app/tools/precache_check.js` for exact current bytes. Optional MiniLM and
restorer bytes are additional. This **exceeds the requested 25 MiB target**;
no budget exception is approved here. The inherited SightLine bundle had no
executable 25 MiB gate. The size/accuracy/offline-loading decision remains a
release blocker, rather than weakening recognition to claim compliance.

`ios/SightLine.xcodeproj` is a new iOS 17+ SwiftUI/WKWebView shell bundling the
same app folder. A read-only loopback server provides module-worker MIME types
and a local HTTP origin. It does not serve the app over Wi-Fi. Camera permission
is requested by WebKit on user action. The project stores no signing team.
An unsigned simulator SDK build succeeded. Opening the new project in Xcode was
attempted, but UI automation remained stuck in its file chooser; it was **not
verified open**. Open that project directly to continue device testing.

No physical iPhone, mobile Safari, actual simulator launch, VoiceOver, installed
Cantonese voice, camera, airplane-mode cold start or service-worker update/reload
test passed during this integration. Native bundled offline loading is an
architectural property; actual WKWebView worker/offline behavior remains to test.

## Validation and concrete limits

| Check | Result |
|---|---|
| `bash run.sh test-reader` | 9 passed, 0 failed |
| field_spot_contract | 2,175 passed |
| field_guard_contract | 898 passed |
| field_engine_contract | 22 passed |
| ocr_input_contract | 4 passed |
| crop_contract | 9 passed |
| gating_parity | 30 passed |
| `bash run.sh bundle-check` | Passed; all original and new graphs load and execute |
| precache_check | 42 entries, 0 missing; size target exceeded |
| ios_compat_scan | 19 shipped files, 0 Safari parse blockers |
| scoped Taste source lint | 0 failures / 0 warnings |
| `git diff --check` | Passed |
| unsigned Xcode simulator SDK build | Passed |
| `bash run.sh test` | Blocked: project Python environment lacks pytest |
| normalizer_diff | Blocked: required Node dependency unavailable |
| full local browser gates | Blocked: listening on localhost returns EPERM |
| remote new-engine controller check | Inconclusive: transport closed / sandbox reset |

The global Taste lint also flags pre-existing synthetic names, old presentation
styling and test symbols; the clean count above is scoped to the changed UI.
The environment's CoreSimulator service connection is invalid, so build success
does not establish simulator execution. No tests were deleted or weakened.
G2/G3/G5/G7/G8/G9 names belong to the Aria repository, not this project's runner;
they must not be reported as SightLine gates that passed.

Full screenshots at 320×568, 390×844 and desktop were requested by the browser
harness. Remote before checks passed at these widths; after idle geometry and
64 px target checks ran before the sandbox was lost. No durable before/after
PNG pair was recovered. Local browser capture returned a blank window, so there
is **no completed visual screenshot sign-off**. `ui-preview.html` is generated
from real native inference output for layout inspection; `worker-preview.html`
is an exploratory embedded-asset harness, not verified browser/offline evidence.

## Reproduction and changed files

Generate fixtures with `bash run.sh py app/tools/render_samples.py`, then run
`bash run.sh bench-reader`, `bash run.sh test-reader`, `bash run.sh bundle-check`
and `bash run.sh test`. Browser runs use `node app/tools/integration_browser.js`
with Playwright installed and permission to bind localhost. That harness uses
the real controller but isolates optional MiniLM on the bounded test host; it
does not replace the full classifier browser tests.

Important files:

- `app/js/ppocr*.js`: model math, sessions, worker and cancellable RPC adapter.
- `app/js/ocr_engine.js`, `pipeline.js`, `app.js`: explicit engine integration,
  literal/context safety, state handling, lazy startup and result rendering.
- `app/index.html`, `app/assets/reader.css`: accessible reading-sheet interface.
- `app/models/ppocr*`, license and `THIRD_PARTY.md`: exact weights and provenance.
- `app/sw.js`, `app/tools/precache_check.js`: cache completeness/version/size.
- `ml/src/bundle_check.py`: new hash and actual inference checks, original gates retained.
- `app/tools/ppocr*`, preview/browser harnesses: measurements and regression cases.
- `ios/`: generated Xcode project, native shell and build instructions.
- `run.sh`: `test-reader` and `bench-reader` commands.

This is a reviewable implementation with explicit release blockers, not a claim
of production readiness or a completed mobile-device validation.
