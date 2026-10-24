# Live Read Aloud tap diagnosis

24 October 2026, connected iPhone 13 Pro. The user confirmed the test sentence
was audible but their own scan's button remained greyed out.

A debug-only pointer listener captured actual user taps without logging OCR
text, medicines, patient details, or image content. Each tap reported: medical
category, zero accepted fields, English voice ready, speech state idle, and
button disabled because the speech plan was ineligible. No speak request was
sent. The native voice was not the failure in this case.

Unverified results now offer **Hear reading help** (Traditional Chinese:
**聽取閱讀協助**). This action plays only the fixed safety/recovery message.
Eligible results retain **Read Aloud**. Receipt confidence thresholds, medical
whole-document verification, OCR output, and language safety policies are
unchanged. A missing speech API still disables playback with its explanation.

Validation: real PP-OCRv5 browser UI regression 95 passed, zero failed; speech
contracts 16 passed; receipt contracts 11 passed; Xcode device Debug build
succeeded and was installed/launched. The build's controller matches the source.
Updated UI tests explicitly verify the precise English refusal and Cantonese
medicine-association refusal rather than permitting uncertain document speech.

The new help action awaits the user's live iPhone tap confirmation. Earlier
native English test speech was audibly confirmed by the user. Diagnostic mode
is only enabled by `--speech-observe` in Debug builds, with no automatic speech.
