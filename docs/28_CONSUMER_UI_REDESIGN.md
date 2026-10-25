# SightLine consumer interface redesign

24 October 2026. Local working copy: `<repo>`, branch
`codex/ppocr-ui`. No push or deployment. The separate original checkout and the
OCR integration work were preserved.

## What changed

The old combined capture/results workbench has been replaced by distinct ready,
camera, photo-review, processing, results and recovery screens.

- **Ready:** one primary **Scan with Camera**, one secondary **Upload Photo or
  PDF**. No source tabs or duplicate camera controls. The SightLine bracket mark,
  warm off-white canvas, slate actions and sky-blue positioning guide establish
  a consistent visual identity.
- **Camera:** a dedicated mobile view keeps the shutter in reach, with an actual
  live preview, review-before-reading, Retake and Back. Permission is requested
  after a tap. Cancelled permission requests cannot reopen a discarded camera.
- **Processing:** honest phase labels, measured progress when available,
  indeterminate activity otherwise, and cancellation. Cancelled jobs cannot
  restore old results. Selecting the same file after cancellation works.
- **Results:** a separate reading page, with recognized medicine names first,
  clearly labeled values, warnings and restrained read-aloud controls. The
  paragraph summary is no longer the visual centerpiece. Original-image
  comparison is alongside desktop results and expandable on mobile. Multi-page
  PDFs explicitly show and announce the first-page-only scope.
- **Safety:** unassociated doses/directions are omitted from structured cards
  when several medicines or dose values are found; the complete recognized text
  remains available. Cantonese refuses to combine their directions. English
  reads the original recognized words instead of constructing a dose list.
  Existing OCR confidence thresholds and literal-field guards remain intact.
- **States:** clear text without extracted fields is distinguished from uncertain
  OCR. Speech API unavailability has its own message. Offline preparation is
  separate from reading progress. No demonstration scan is embedded in the app.
- **Preferences/help:** native accessible dialogs hold language, appearance,
  optional reading settings, privacy and positioning help. Light is the default;
  dark or device appearance can be selected and persisted.

No frontend framework, web font, icon package or production dependency was
added. Icons and positioning guidance are inline SVG. Existing models, math,
classifier, restorer, recognition worker and image processing were preserved.

## Skills and design decisions

**design-taste-frontend** was read and applied through its app-screen workflow.
The consumer direction is **Pastel consumer**, seed **1624706442**, with existing
palette/type retained. Its consumer surfaces replace the previous B2B framed
sheet. The product-specific face is a paper label inside SightLine's scanning
brackets, repeated purposefully in the brand mark and processing indicator.

Each screen answers one question: how do I start; is my photo readable; is the
reader working; what words were found? A 1120 px desktop composition and a
separate mobile flow replace squeezed workspace columns. Reading text is 20 px,
with larger field values; controls meet the project's 48 px minimum. Large
consumer headings intentionally use Taste's `app-title-size` allowance, rather
than its dense-dashboard title cap. No other inspector rule was waived.

Taste's source lint and real-browser inspector both passed. Actual screenshots
were opened and reviewed at 320, 390, 768 and 1440 px, including full results
pages. The review corrected heading measure, crowded results, medicine-name
priority, narrow-phone field visibility and camera/processing control placement.
A compositor artifact during repeated viewport resizing was resolved by fresh
browser contexts and settled software-rendered captures; screenshots were not
painted or edited afterward.

**Graphify** was used for the supported AST extraction and graph query workflow,
tracing `setStage`, `goHome`, `startCamera`, `runScan` and `render` into the
existing OCR boundary. Outputs are in `artifacts/redesign/graphify-out` and
`architecture-query.log`. Architecture analysis did not substitute for visual
review. Its CLI notes the extractor's older node-ID scheme; this graph is scoped
to the uniquely named frontend files.

**Third recent skill: not verified.** The project/global skill directories and
installation metadata identify Taste and Graphify, plus older unrelated
Higgsfield skills and bundled system skills. They do not establish a third
recently installed design skill. None was invented or substituted silently.

## Real screenshots

These are browser captures of the actual application. Result screenshots use
clearly labeled synthetic images processed by the shipped OCR and classifier;
they are not injected result objects or clinical validation.

| Screen | Mobile, 390×844 | Desktop, 1440×1000 |
|---|---|---|
| Ready | [Home](../artifacts/redesign/home-390.png) | [Home](../artifacts/redesign/home-1440.png) |
| Processing | [Processing](../artifacts/redesign/processing-390.png) | [Processing](../artifacts/redesign/processing-1440.png) |
| Results | [Results](../artifacts/redesign/results-390.png) | [Results](../artifacts/redesign/results-1440.png) |
| Full results | [Full page](../artifacts/redesign/results-full-390.png) | [Full page](../artifacts/redesign/results-full-1440.png) |

Additional captures: `home-320.png`, `results-320.png`, `processing-320.png`,
`home-768.png`, `camera-test-device-320.png`, `camera-review-390.png`,
`results-multiple-medicines-390.png`, `results-unstructured-390.png`,
`results-needs-checking-390.png`, `results-text-200-390.png` and
`error-reader-390.png`. Camera captures use the browser's **test camera**.

Genuine baseline captures are in `artifacts/redesign/before`: home at three
widths and real OCR results at 390/1440, using the pre-redesign app snapshot and
the same single-label fixture. Before/after backend identity was preserved.
The final UI check records the first recognized field's position and requires
it to be completely visible in the initial 320×568 and 390×844 result viewports.

## Model integration and validation

PP-OCRv5 mobile FP32 and ORT Web 1.20.1 remain the selected default. Original
CRNN remains selectable. No Tesseract was added; its exclusion gate still passes.
The actual file input → controller → worker → detector/recognizer → classifier
→ safety → results path completed in real Chrome and WebKit. Offline cold-page
and never-scanned cold-install reads also succeeded in those browser suites.

| Validation | Result / evidence |
|---|---|
| New UI regression | **88/88**, `artifacts/redesign/ui-check.json` and `ui-check-final.log`; real OCR, responsive geometry, 200% text, dialogs, speech dispatch, camera lifecycle, cancellation, retry, unsupported text, multiple medicines and failed offline installation |
| Full existing browser battery | **97/97 WebKit + 97/97 Chrome**, `full-browser-final.log`; 24 benchmark records in `full-browser-records.json` |
| Final offline/version boot recheck | **16/16 per engine**, `final-offline-boot.log` |
| WebKit self-test | **196/196**, including 12 actual CRNN inference assertions |
| Cantonese browser safety checks | **14/14** |
| Extraction/crop/gating/PP contracts | **3,147 checks passed** across the seven contract runners |
| Bundle check | Passed; original and new models load and execute |
| Safari syntax scan | 19 shipped files, 0 blockers |
| Taste lint / browser inspector | 0 failures, 0 warnings, with the stated heading-size allowance |
| Axe WCAG A/AA automated checks | Zero violations in the tested views; JSON reports included |
| Unsigned iOS simulator SDK build | `BUILD SUCCEEDED`, `xcode-build.log` |
| Two-page PDF scope | Real PP-OCR verified: first page only; **Page 1 of 2** visible in results, `pdf-scope.json` |
| Required cache paths | 42 entries, 0 missing; about **38.70 MiB** |
| Diff whitespace check | Passed |

Existing tests were aligned to the real contract, not weakened:

- Test servers now serve CSS with the correct MIME type.
- CRNN session-disposal and old-device SIMD tests explicitly select Original
  reader, rather than inspecting an unused CRNN while the default PP worker runs.
- Cold installation must now **succeed** on its first offline scan, with PP-OCR
  engine identity verified; the old expected missing-model failure was removed.
- Geometry uses the CRNN-selected candidate and exported crop dimensions, not
  an unused CLAHE convenience image. The actual detected/cropped lines pass.
- Empty OCR is checked against the result's real text, rather than treating an
  on-screen empty-state sentence as successful recognition.
- The PDF fixture now uses a real 46 px installed font on macOS/Linux and fails
  clearly if none exists. Its old silent bitmap fallback invalidated the test.
- Cantonese preferences are tested for actual visibility inside the new dialog,
  and refusal information is checked through its Help disclosure.

## Remaining limits

The legacy low-level **CDP smoke runner still has 5 failures**: large model/runtime
`Cache.add()` requests intermittently fail with a network error in its persistent
Chrome profile, preventing that runner's install/offline checks. Diagnostics
are saved in `browser-smoke.log`; its 36 other checks pass. This is not erased
by the successful independent browser suites, and its precise cause remains
unresolved. Required assets get one sequential retry after initial failure;
persistent failures are never marked Ready offline. The new UI suite verifies a
real HTTP 503 installation failure and recovery through Help's saving retry.

The 38.70 MiB required cache exceeds the earlier 25 MiB target. This redesign
preserves model precision and the existing offline asset set; it does not approve
a budget exception or trade recognition quality for size. Model accuracy and
medical corpus limitations from [the integration report](27_PP_OCR_UI_INTEGRATION.md)
remain. Clean synthetic demonstration scans do not establish clinical accuracy.

`bash run.sh test` remains blocked by missing pytest in the inherited Python
environment. `normalizer_diff.js` requires an unavailable external reference at
`/tmp/ad/SightLine-Mobile/js/pipeline.js`. These failures are retained in logs.

No physical iPhone camera, actual iOS 17 Safari, VoiceOver session, real installed
Cantonese audio output, hearing/volume experience or physical-device memory limit
was verified. Desktop WebKit and simulated camera/speech transport checks must
not be described as physical-device tests. Xcode's native path chooser remained
unresponsive to project opening; the generated project and successful build are
available at `ios/SightLine.xcodeproj`.

## Run and reproduce

`bash run.sh serve 8080` launches the application. Read it at
`http://localhost:8080`. Open `ios/SightLine.xcodeproj` for the bundled iOS shell;
its next build copies the current frontend.

Browser test tools are development-only, installed outside the product:

```sh
npm install --prefix /tmp/sightline-ui-tools playwright-core @axe-core/playwright
node /tmp/sightline-ui-tools/node_modules/playwright-core/cli.js install webkit
bash run.sh py app/tools/render_samples.py
bash run.sh py app/tools/make_fixtures.py
PW_CORE=/tmp/sightline-ui-tools/node_modules/playwright-core \
CHROME_BIN='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
AXE_CORE=/tmp/sightline-ui-tools/node_modules/@axe-core/playwright \
bash run.sh test-ui
```

Use the same `PW_CORE`/`CHROME_BIN` variables with `bash run.sh ios-full
--engine=both`. Other checks: `bash run.sh ios-check`, `bash run.sh test-reader`,
`bash run.sh bundle-check`, `node app/tools/precache_check.js` and the contract
scripts named in the artifact directory. Substitute your local Chrome path.

Important changed files: `app/index.html` (screens and dialogs),
`app/assets/reader.css` (design system and responsive capture/reading layouts),
`app/js/app.js` (view state, focus, cancellation and presentation safety),
`app/sw.js` (version and bounded required-asset retry), browser test helpers,
`app/tools/ui_redesign_check.cjs` and `run.sh` (`test-ui`). The backend-integrity
artifact compares OCR/model files with the saved pre-redesign snapshot.
