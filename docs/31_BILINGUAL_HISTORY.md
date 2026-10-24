# Bilingual reading and local scan history — 24 October 2026

Implemented locally in `<repo>`, branch
`codex/ppocr-ui`, the existing SightLine integration/Xcode working copy.
`~/aria-reader` remains the reference repository and was not edited.
Nothing was pushed or deployed. Existing unrelated unfinished changes remain.

## Audit and causes

English receipt speech already used accepted individual fields. Cantonese
receipt speech was explicitly refused in the controller. The old prescription
mapper covered only a limited vocabulary; selecting Cantonese did not translate
the interface. Its punctuation/substring transformations were unsuitable for
the new literal, confidence-gated summaries. The old mapper remains for its
historical tests; the current controller uses the new deterministic summary
module instead.

There was no scan-history database. Preferences alone were in localStorage.
The iOS shell also chose a new loopback port each launch: browser storage belongs
to an origin, so an IndexedDB history at one port would disappear from view at
the next. Production now uses stable `http://127.0.0.1:18192` with the existing
persistent WKWebView data store. A port collision is a startup error, not a
silent switch to a different, apparently empty history.

The earlier “Offline saving is unavailable” warning concerned service-worker
availability in native WKWebView, not a history write. The existing validated
native bundled-assets path remains. History has its own real IndexedDB error
reporting; it does not treat a service-worker badge as storage permission.

## What changed

- Visible English / 繁體中文（廣東話） control on home, results and history.
  Interface copy, navigation, accessibility labels and deterministic spoken
  summaries change together. The choice persists; neither rescan nor inference
  is performed. Extracted names, prices, source text and confidence stay intact.
- Actual local Cantonese voice required; `zh-HK` utterances select installed
  `zh-HK` / Cantonese `yue` voices. Mandarin is never substituted. Missing-voice
  instructions and an explicit English alternative are provided. Existing
  asynchronous discovery, user-tap playback and Stop/session handling remain.
- Receipts use only accepted fields, including exact decimal money handling.
  HKD is spoken as 蚊／毫／仙; ambiguous `$` is not assumed to be HKD. Malformed
  comma groupings are rejected. Unknown brands remain original text.
- Medical summaries retain the original medicine name. The whole-page safety
  gate still applies; multiple medicines or unmatched doses are refused.
  Supported strength/dose units and exact full-line directions/warnings have
  fixed Cantonese templates. Negated/unsupported instructions are not converted
  by substring. Partial translations explicitly require checking the original
  and confirming with a pharmacist. PDF scope remains the recognized page.
- History is directly accessible from main navigation, with a top Back action,
  chronological records, original title, scan date, trusted amount when present,
  reopen, and confirmed individual/all deletion. Reopened results retain their
  safety data and can switch languages without OCR.
- Saving defaults off. Consent explicitly covers this and future scans; images
  are optional. IndexedDB stores structured results, source text, confidences
  and metadata locally. “Saved” appears only after transaction completion.
  Quota/permission errors have actionable messages and preserve the current
  result. Delete All also turns future saving off. Disabling saving alone does
  not erase existing records; this is explained in preferences.
- Stored images are copied into an owned Blob when reopening. This fixes a real
  WebKit failure where resaving an IndexedDB-backed object URL produced “Load
  failed” before storage was reached. Async opening checks navigation generation
  before committing, so leaving History cannot restore a stale screen.
- New scripts are precached under `v1.9.0-bilingual-history`. Installation writes
  two assets concurrently instead of issuing the entire model-cache burst.
  Required-asset failures still fail installation and offer retry.

## UI/UX skill applied

Read and used `~/.codex/skills/ui-ux-pro-max/SKILL.md` and its local
search workflow. Verified UX guidance on screen-reader structure, predictable
navigation, destructive-action confirmation and success feedback informed the
implementation. An initial design-system search was off-topic and was not
adopted. Search evidence is under `artifacts/bilingual-history/skill-*.txt`.

The approved Taste palette and accessible typography are preserved. Added a
CJK system-font stack, large touch targets, semantic navigation/articles,
readable privacy consent, responsive history cards, honest loading/empty/error
states, and consistent disclosure controls. Long Chinese labels were checked at
320, 390, 768 and 1440px, including increased text size through the existing UI
suite. Existing Check Device top navigation remains. Taste source lint and its
real-browser home inspector report zero failures/warnings; screenshots were
visually inspected, not accepted on measurements alone.

## Reproduction and evidence

Development-only browser tooling was installed outside the repository:

```sh
export PW_CORE=/tmp/sightline-bilingual-tools/node_modules/playwright-core
export AXE_CORE=/tmp/sightline-bilingual-tools/node_modules/@axe-core/playwright
export CHROME_BIN='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
bash run.sh test-bilingual
bash run.sh test-speech
bash run.sh test-receipts
bash run.sh test-reader
bash run.sh test-bilingual-ui
bash run.sh test-history
bash run.sh test-native-history
bash run.sh test-ui
bash run.sh ios-check
bash run.sh ios-full --engine=both '--suites=boot,scan,refusal,offline,cold install'
bash run.sh bundle-check
```

Use `node app/tools/bilingual_visual_check.cjs` for the bilingual screenshot/axe
checks; it starts an isolated static server. `bash run.sh serve 8151` opens a
manual preview. Logs and JSON evidence are in
`artifacts/bilingual-history/`. Tests deliberately injected storage and runtime
failures only in the named negative controls; OCR and platform speech were real.

The public SROIE receipt `artifacts/aria-merge/sroie-X00016469612.jpg` was scanned
through the actual upload/PP-OCRv5 path. Its shop/payment fields were accepted;
its uncertain total remained omitted. Switching language did not turn that
amount into a confirmed value. Prescription end-to-end evidence uses the
explicitly synthetic single-medicine label through real OCR, not a fixture
result substituted into the controller.

`actual-speech.json` records native platform calls and start/end state: English
Daniel and local Cantonese Sinji (`yue-HK`, utterance `zh-HK`). Platform events
are stronger than mocked success but do not prove audio heard on an iPhone.

Browser persistent-profile tests close/relaunch Chromium and WebKit, load
offline, reopen saved fields and optional original images with OCR disabled,
exercise a real transaction abort, inject denied storage, retry, and erase the
database. The native history test uses two separate macOS WKWebView processes
with persistent data and stable origin; the second process blocks external
HTTP(S), reopens the real receipt with OCR forbidden, then removes its isolated
public test records. Its audit bundle identifier differs from the actual app.

Verified regression results (individual logs retained):

| Check | Result |
| --- | --- |
| Bilingual + existing speech/receipt/OCR contracts | 45/45 |
| Real OCR, platform speech, language and History journey | 42/42 |
| Existing complete UI suite | 88/88 |
| Bilingual visual checks, four widths | 18/18 |
| Chinese receipt and History axe scans | Zero violations in the checked WCAG A/AA rules |
| Persistent-profile storage/offline/failure controls | 8/8 Chromium, 8/8 WebKit |
| Native persistent WKWebView write then offline restart | Both processes pass |
| Existing browser boot/scan/refusal/offline/cold-install battery | 69/69 Chromium, 69/69 WebKit |
| WebKit self-tests | 196/196 assertions |
| Safari parse compatibility | 25 shipped JS files, zero blockers |
| Required asset existence | 52 entries, zero missing |
| Bundle model loading/inference check | Pass |
| Unsigned iOS simulator SDK build | BUILD SUCCEEDED |
| Taste source and real-browser home inspection | Zero failures/warnings |
| Python pytest | Blocked: pytest is absent from the configured environment |

The final actual-speech journey is recorded separately in `checks.json` and
`journeys-verified.log`. Earlier failed screenshot runs are retained as diagnostic
logs: their external Python preview server reset connections. The isolated Node
server completed all four viewports. An additional English medicine assertion
was corrected to accept the actual uppercase OCR name while still requiring the
exact medicine and strength. Production text was not changed to satisfy the test.

Screenshots of the implemented interface:

- [Chinese mobile home](../artifacts/bilingual-history/review-home-zh-390.png)
- [Processing](../artifacts/bilingual-history/review-processing-zh-390.png)
- [Real-receipt results, full scroll](../artifacts/bilingual-history/review-receipt-zh-390.png)
- [Mobile History](../artifacts/bilingual-history/review-history-zh-390.png)
- [Desktop History](../artifacts/bilingual-history/review-history-zh-1440.png)
- [320px consent](../artifacts/bilingual-history/review-consent-zh-320.png)
- [English desktop home](../artifacts/bilingual-history/taste-home/desktop-fold.jpg)

The new History screens have no preexisting baseline. Historical pre-redesign
screenshots are retained under `artifacts/redesign/before/`; they are not presented
as a new baseline for this task.

## Limitations and iPhone checklist

- Actual iPhone speaker output, camera, downloaded-voice airplane-mode behavior,
  storage eviction and OS upgrade behavior remain unverified. Test both languages
  from a real tap, Stop/replay, force-quit/reopen History, airplane-mode reopen,
  and missing Cantonese voice recovery. Install a Cantonese/Hong Kong voice in
  iOS accessibility speech settings if absent; return and retry, or use English.
- Templates are deterministic, not arbitrary document translation or medical
  advice. They still need native Cantonese and clinical review before patient use.
- One synthetic Traditional Chinese image was recognized verbatim by PP-OCRv5,
  but existing classification/extraction did not yield receipt fields. See
  `chinese-ocr-probe.json`. General Chinese-document understanding is **not**
  claimed. The selectable original CRNN remains Latin-only.
- Earlier unsaved scans cannot be recovered automatically. Clearing browser/app
  data can erase history. Anyone with access to the device/app can read saved
  records; this is disclosed before consent. No server upload is introduced.
- Required precache is 38.85 MiB (52 entries, zero missing), still above the
  inherited 25 MiB release target. Model weights/precision were not reduced to
  satisfy a presentation task. Existing release blockers remain.
- `bash run.sh test` cannot execute: the configured shared Python environment
  lacks pytest. This is reported, not counted as a pass or fixed by mutating the
  other project's shared environment.

## Important files

`app/js/cantonese_summary.js`: safe deterministic bilingual content;
`i18n.js`: interface-only reversible localization; `history.js`: IndexedDB
transactions; `app.js`: language, consent/history and speech integration;
`tts.js`: explicit Cantonese locale and language-aware chunk handling;
`index.html` / `assets/reader.css`: navigation, cards and dialogs;
`device_check.js` / `selftest.html`: matching interface language;
`sw.js`: new required modules/cache version; `ios/Sources/{SightLineApp,
WebAssetServer}.swift`: stable storage origin. New test tools cover actual
speech, history failure/restart/offline, native persistence and bilingual layout.
