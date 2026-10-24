# Read Aloud repair and Device Check UX

24 October 2026. Implemented in `<repo>` on local
branch `codex/ppocr-ui`. This is the SightLine app and Xcode project used in the
preceding integration. `~/aria-reader` supplied the reference brief,
elder/navigation/motion requirements and parser; its application source remains
untouched. Existing unrelated dirty work was preserved. Nothing pushed/deployed.

## Root cause: receipt eligibility, before speech synthesis

The real public SROIE receipt X00016469612 was recognised by **PP-OCRv5 FP32**:

| Signal | Measured baseline |
|---|---|
| OCR confidence | 0.8845656961 |
| Receipt classification confidence | 1, weighted receipt score 7 vs prescription 0 |
| Accepted shop field confidence | 0.86 |
| Accepted cash payment confidence | 0.80 |
| Whole-document `speechText` | empty |
| Read Aloud button | hidden |
| Local browser voice catalog | 180 voices on this Mac |

`ppocr_core.js` intentionally empties whole-document speech when a line is
uncertain, protecting medicine/dose associations. The receipt total label scored
0.72784, below 0.75. Although receipt extraction correctly kept two clear fields,
`app.js` rendering and playback still required the **medical whole-page gate**.
The failure was reproduced before changing code; it was not an absent speech API.
See `artifacts/speech-repair/before.json` and the before screenshot.

There were additional reliability defects in `tts.js`: handlers were attached
only to the last queued utterance, after calling `speak`; earlier chunk errors
could therefore be ignored. A timer could report successful completion without
any end event. Stop cancelled the platform queue without settling the caller's
promise. Voice selection could fall back to an unrelated language or remote
voice. Sentence splitting could insert whitespace into decimal amounts.

## Implemented repair

`js/speech_plan.js` is the shared rendering/playback eligibility policy:

- **Receipts:** speak only existing accepted fields with finite confidence at
  least **0.75** and source-line evidence. Never fall back to the entire noisy OCR
  transcript. Explain that unclear details were omitted. The actual test receipt
  speaks the shop heading and cash payment, never the uncertain `9.00` total.
- **Prescriptions:** retain the PP-OCR whole-document completeness gate and
  original medicine/dose/warning order. Existing verbatim, multi-medicine and
  Cantonese refusal rules remain intact. No medical threshold was reduced.
- `lastResult.speechEligibility`, existing classifier diagnostics, local word and
  field confidences make these four distinct signals inspectable. No production
  document text is sent to a server or logged by the new speech controller.

`tts.js` now discovers voices using `voiceschanged` plus a bounded polling window,
selects an installed **local** voice of the correct language, and attaches
start/end/error handlers before each utterance is queued. English can use another
installed English locale; Mandarin never substitutes for Cantonese. Empty catalogs
report a discovery delay/retry instead of immediately claiming no voice is installed.
Device Check also refreshes when a late catalog arrives.

A ready voice starts synchronously within the user's tap. Cantonese's lexicon is
prepared when choosing its language; if still unavailable, a fresh tap is requested
rather than starting speech after a network await. Queue sessions retain their
utterances, ignore stale events, settle Stop immediately and permit replay. Every
chunk must end; timeouts now fail honestly rather than inventing completion.
Error messages distinguish a missing offline voice, delayed voice list, gesture
restriction, audio-device failure and a stalled engine. Decimal punctuation stays
intact. The actual speech lifecycle is exposed through `TTS.getState/subscribe`.

The prominent 24px Read Aloud button has a minimum 64px target and changes to
**Stop reading** during preparation/playback. An atomic status beside it reports
starting, reading, completion, stop or the specific failure. Stop/replay are offered;
pause/resume are not promised without target-device reliability evidence.

Relevant platform references: [voice discovery](https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis/voiceschanged_event),
[utterance events](https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesisUtterance),
[error reasons](https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesisErrorEvent/error),
and [WebKit's iOS gesture restriction](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/speech/SpeechSynthesis.cpp).

## Device Check navigation and honest checks

Help now has a directly visible **Check Device** action. About retains its route.
The existing page opens in an accessible dialog above the reader, so closing it
preserves the actual scan, source image and result state. Its top-left **Back to
SightLine** stays outside the scrolling check content. A bottom Return action
remains. Direct standalone access has its own sticky, safe-area-aware top exit.
Keyboard Escape works with focus in either the parent dialog or embedded page;
message-based exits validate both origin and source. Dark appearance follows the
reader when reopening the check.

The page groups camera/permission status, executable browser/SIMD compatibility,
verified offline installation, model/runtime file availability, and local voices.
No camera permission prompt is triggered by checking. Missing capabilities get
plain recovery steps. Models being available is explicitly not an OCR accuracy
or successful inference test. File readiness is separate from voice readiness.
Native mode uses the verified installed-bundle contract; browser mode asks the
active service worker to verify its actual required cache.

**Test voice** is user initiated, with Stop and genuine start/end/error handling.
An end event asks **“Did you hear the voice?”**. Audible playback remains
unverified until a person confirms it. Starting a new test clears that confirmation;
a metadata recheck does not invent human confirmation. Detailed software checks
remain available behind a disclosure, preserving the original 196 assertions.
The WebKit runner now opens that disclosure before pressing the existing Run button;
no assertion or expected result was weakened.

## Skill and visual review

The newly verified third skill is **ui-ux-pro-max**, located at
`~/.codex/skills/ui-ux-pro-max/SKILL.md`. Its actual supported targeted
`--domain ux` searches supplied predictable back behavior, sticky navigation and
contextual live-status feedback. Queries and verified results are saved as
`skill-navigation.txt` / `skill-feedback.txt`. The mobile pre-delivery checklist
was reviewed. Its generic smaller typography does not override elder readability.

The existing Taste **Pastel consumer** direction (seed 1624706442), SightLine mark,
warm ground, slate actions and sky accents are preserved. Taste source lint and
real-browser inspection of Device Check each report **0 FAIL / 0 WARN**.
`app-title-size` retains the already documented consumer/elder heading allowance.
Device Check uses 22px body text, 20px minimum technical text, 64px controls,
explicit words for status, visible focus and existing reduced-motion behavior.
The new body copy uses the high-contrast text token. No framework, icon library,
web font, online TTS API or production dependency was added.

Real screenshots were opened and inspected at phone, tablet and desktop sizes,
including full-scroll receipt/device pages. A Chrome compositor artifact during
repeated viewport changes was corrected by fresh contexts/software rendering;
images were recaptured, not edited. Playback status was moved near the main
button, and Escape/focus handling corrected after the browser checks exposed it.

## Executed tests and evidence

All artifacts below are under `artifacts/speech-repair/`.

| Command / scope | Actual result |
|---|---|
| `node --test app/tools/speech_contract.test.cjs app/tools/receipt_regression.test.mjs app/tools/ppocr_contract.test.mjs` | **32/32** |
| `bash run.sh test-ui` | **88/88**, real OCR; legacy speech transport stub only in its marked assertion |
| `bash run.sh ios-full --engine=both '--suites=boot,scan,refusal,offline,cold install,tts'` | **75/75 WebKit + 75/75 Chromium** |
| `bash run.sh ios-full --engine=both --suites=tts` after speech cleanup | **6/6 per engine** |
| `bash run.sh ios-check` | **196/196** self-test assertions, zero page/console errors |
| `bash run.sh test-speech-ui` | **30/30** journey/layout checks plus actual speech-event assertions |
| `node app/tools/device_check.cjs` | **8/8 per engine**, including missing SIMD/API, Escape, theme, doubled text |
| `bash run.sh test-native-offline` | **pass**, native macOS WKWebView real PP scan, Device Check and return with external HTTP(S) blocked |
| `bash run.sh bundle-check` | **pass**, all shipped models load/run |
| `node app/tools/precache_check.js` | **49 required entries, zero missing** |
| `node app/tools/ios_compat_scan.js` | **zero Safari parse-time blockers** |
| `xcodebuild … -sdk iphonesimulator … CODE_SIGNING_ALLOWED=NO build` | **BUILD SUCCEEDED** |
| Taste lint / rendered Device Check inspector | **0 FAIL, 0 WARN** each |

The speech journey performs real receipt OCR, clicks the real Read Aloud button,
records the actual platform `speak` calls without replacing them, observes genuine
start/end events using the local **Daniel / en-GB** voice, stops and replays. It
also scans a synthetic prescription fixture and confirms the real utterance
contains Amoxicillin and its recognised 500 mg strength before stopping it.
The prescription fixture is synthetic, not a real patient document.

A fresh offline page then recognises the real receipt again and receives a real
speech end event while browser networking is disabled. Device checks and its voice
test run offline; an end event alone leaves `audioVerified: false`. Navigation
preserves the receipt, and layouts are tested at **320×568, 390×844, 768×1024 and
1440×1000**, with doubled text, no horizontal overflow and zero automated WCAG
A/AA violations in the device dialog. Negative-control mocks test failure states;
they are not the evidence for real speech. See `actual-speech-events.json`,
`prescription-speech.json`, `journey-checks.json` and `device-negative-controls.json`.

The initial WebKit self-test runner failed because it tried to click a button
inside the new closed disclosure. Its navigation was corrected and all 196
original assertions rerun. The initial Escape test exposed a real embedded-focus
bug; that was fixed and both browsers passed the stronger dialog-closed check.

## Screenshots

- Receipt before: `before-receipt-390.png`; after: `after-results-390.png`.
- Actual reading state: `after-reading-390.png`.
- Device page before: `before-device-390.png`; after: `after-device-full-390.png`.
- Embedded persistent exit: `after-device-320.png` / `after-device-390.png`.
- Responsive full-scroll results/device: `after-results-{320,390,768,1440}.png`
  and `after-device-full-{320,390,768,1440}.png`.
- Taste independent captures: `taste-device/` (desktop, tablet, mobile).

## Important files and limits

`js/tts.js` handles local voice discovery and reliable speech sessions;
`js/speech_plan.js` separates receipt and prescription eligibility;
`js/app.js` connects that policy, speech state and preserved-scan device navigation;
`index.html` and `assets/reader.css` present prominent speech and persistent exits;
`selftest.html`, `js/device_check.js` and `assets/device.css` implement the readable
check page while retaining its original software battery. `sw.js` versions the
new modules/styles as **v1.8.4-speech**. Runner/QA tools reproduce these checks.
Model weights, OCR math, image processing, extraction thresholds and the original
selectable CRNN remain intact. The native audit tool adds a real device-check
round trip; production native engine/server behavior is unchanged in this task.

**Actual audible playback is unverified**: no person listened on a physical
iPhone. Genuine speech events prove platform progression, not acoustic output,
volume, routing or VoiceOver experience. Physical-iPhone tests still needed:
Scan/upload → listen → stop/replay, volume/silent mode/Bluetooth, Cantonese voice,
background/resume, VoiceOver, camera permissions and airplane-mode app restart.
The iOS result above is a simulator-SDK build, not a device execution claim.

The historical 25 MiB precache target remains exceeded (about **38.81 MiB**);
no lossy model substitution was made. The inherited Python suite remains blocked
by missing `pytest` in the shared existing environment; it was not modified and
no Python pass is claimed. The old CDP-profile cache failure is outside these
fresh-context/native checks and was not declared fixed. No push or deployment.
