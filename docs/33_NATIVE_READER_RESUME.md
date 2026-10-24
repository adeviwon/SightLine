# Native reader loading and foreground recovery — 24 October 2026

The user reported a physical-iPhone error after returning to SightLine:
`could not load ocr_engine.js: Importing a module script failed`.
This occurs before inference, independently of image quality or OCR confidence.
The original device's underlying socket error was not captured.

The native shell retained a loopback listener across iOS suspension without a
foreground restart. OCR JavaScript was first imported during scanning, after a
photo picker or app switch could suspend the app. Clearing the application's
rejected promise did not reliably recover a failed module fetch in WebKit.

## Repair

- ReaderRoot stops the asset server on background and creates a new listener
  on active. The web view and stable port 18192 remain, preserving history's origin.
- Startup warms the OCR JavaScript graph and enhanced-reader wrapper. It does
  not load weights, create workers, compile WASM or run inference.
- Typed READER_LOAD_FAILED errors offer Reload reader, resetting the document's
  module state. Local storage remains; users choose the photo again.
- English/Traditional Chinese recovery replaces developer jargon and incorrect
  photo-quality advice. Technical cause remains in a helper disclosure. A known
  failed module load overrides Ready offline.
- Service-worker cache is now v1.9.4-visible-read-aloud. Model weights and safety
  thresholds are unchanged.

## Executed validation

- test-reader-loading: 14 checks across Chromium/WebKit, including a deliberately
  aborted module fetch, no startup weight/WASM requests, localized reload,
  persistent language and real PP-OCRv5 receipt recognition after recovery.
- test-native-resume: real native WKWebView retains its page while the actual
  Swift listener restarts on the same port. A real receipt scans with external
  network blocked. native-resume.json records serverRestarted=true.
- Existing test-reader: 9/9. Existing test-ui: 88/88.
- Safari compatibility: 26 JS files, zero blockers. Precache: 53 entries, zero
  missing; 38.91 MiB remains above the inherited 25 MiB release target.
- Signed iOS Debug build: BUILD SUCCEEDED. Nine relevant bundled assets match
  current source by SHA-256 (iphone-bundled-assets.json).
- Installed and launched com.sightline.reader on the connected physical iPhone
  13 Pro. iphone-install.json and iphone-launch.json record successful operations.

Evidence is under artifacts/bilingual-history/. Playwright commands use the
development-only PW_CORE installation documented in report 31.

The physical phone's scan after switching away/returning and actual speaker
audio still need user confirmation. The simulator had first-boot WebKit delays
and blank rendering and is not counted as a successful iOS scan. The native
WKWebView receipt test above ran on macOS, not a physical iPhone.

## Read Aloud discoverability follow-up

The user subsequently reported Read Aloud disappearing on recognized results.
The controller intentionally hid it when speech was unavailable or the selected
summary failed safety eligibility. Its explanation was below the result fields.
The button now remains at the top of results, disabled when playback is not
permitted, with the reason immediately below and linked by aria-describedby.
Eligible English/Cantonese playback and the underlying safety policy are unchanged.

The 88 UI checks pass, including visible/disabled unavailable speech and both
English/Cantonese medical refusals. The test assertions were changed from hiding
the button to checking that it cannot play and explains why. Actual platform
receipt start/end, Stop/replay, prescription and offline receipt speech checks
passed. An earlier platform replay attempt failed; an isolated rerun passed,
with added replay event diagnostics and no TTS runtime change. Acoustic output
on a physical phone is still not claimed. The updated signed bundle was built
and its seven speech/UI assets verified against source before installation.
