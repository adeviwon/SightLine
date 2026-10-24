# Read Aloud voice retry and sectioned summary preview

24 October 2026. Implemented in the SightLine Xcode/PWA source. The OCR model,
recognized source text, confidence thresholds, and prescription safety policy
were not changed.

## Voice failure found

The Cantonese results handler checked for a `zh-HK` voice, awaited asynchronous
voice discovery when the list was initially empty, and then unconditionally
reported that no Cantonese voice existed. It never checked the refreshed list.
This made a delayed iOS voice catalog look like an unsupported voice even when
the device subsequently returned one.

Native voice discovery now retries while its catalog is empty. On iOS, Read
Aloud waits for the native list and checks for a Cantonese voice again before
speaking. The browser path still requires the user to tap again after delayed
voice discovery so mobile Safari's gesture behavior is respected. Mandarin is
never used as a Cantonese substitute. If the refreshed catalog has no Hong Kong
Cantonese voice, SightLine reports that and leaves English available.

## Summary layout

The expandable Read Aloud preview now renders structured field cards rather
than one long paragraph. Medicine name/type, label strength, amount each time,
timing, warnings and other accepted fields remain in separate cards. Receipt
preview cards group shop, purchase details, totals, payment and other fields.
Changing to Traditional Chinese updates supported labels and values without
rerunning OCR. The visible result cards have clearer borders, spacing and light
elevation. Safety eligibility is unchanged; an ineligible page continues to
show the reason instead of a misleading read-aloud preview.

## Verification

- `node --test app/tools/speech_contract.test.cjs app/tools/receipt_regression.test.mjs` — **26/26**.
- `bash run.sh test-ui` with local Playwright and Axe — **94/94**, including real PP-OCRv5 fixture scans, separate English/Traditional Chinese preview cards, 320/390/tablet/desktop layouts, text enlargement, and WCAG A/AA checks.
- Xcode Debug device build — **BUILD SUCCEEDED**.
- The updated signed build is installed/launched on the connected iPhone 13 Pro.

Native event and browser tests do not prove sound was physically heard. The
latest app reports the iOS output route and media volume during playback. The
user should still confirm audible output and report that line if the voice is
silent.
