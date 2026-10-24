# Physical iPhone speech verification

24 October 2026. SightLine integration branch, iPhone 13 Pro.

The native bridge previously had no WKWebView reference. Voice requests reached
Swift, but catalogs and playback events could not return to JavaScript. The
host now assigns the view before loading the page.

The corrected signed build was installed and actually launched on the unlocked
phone. A debug-only `--speech-audit` launch path exercises the production bridge
and emits no scanned text. The user confirmed hearing the native English test
sentence. iOS selected Daniel (Enhanced), with output Speaker and media volume
1.0, and delivered start and finish events.

A second audit used public SROIE receipt X00016469612, real PP-OCRv5 inference,
the production results controller, and the actual Read Aloud button handler.
It classified the page as receipt, retained two eligible fields, enabled the
button, and completed native speaker playback with real start/end events.
Evidence: `artifacts/speech-repair/iphone-native-receipt-speech.json`.

The receipt audit passes its public fixture through a debug launch environment
variable, temporarily disables automatic history saving during the test, and
restores the setting afterward. It is excluded from Release builds. Normal app
launches do not run it. The app was relaunched normally after verification.

Validation: Xcode device Debug build succeeded; `bash run.sh test-speech`
16 passed; `bash run.sh test-receipts` 11 passed; `git diff --check` passed.
The user's audible confirmation covers the English test sentence. Receipt
playback events are verified on physical hardware; this run did not establish
Cantonese audibility or offline airplane-mode speech.
