# Native iPhone Read Aloud

24 October 2026. SightLine's Xcode build now sends Read Aloud through iOS
`AVSpeechSynthesizer` and `AVAudioSession`; browser speech remains the fallback
for web builds. The prior iPhone shell used WebKit's `speechSynthesis` queue
directly. Queue completion is not evidence that sound reached the speaker, and
that route did not configure an iOS playback audio session. This explains how
the button could be present and the web tests could pass while the user heard
nothing. Device-specific output routing had not previously been inspected, so
the exact prior mute/route state is unknown.

The native bridge enumerates voices available to iOS and returns them to the
same language selector. It only matches English voices for English and `zh-HK`
or `yue-HK` for Cantonese; Mandarin is never substituted. A tap sends the
existing confidence-checked English or Cantonese summary directly to the native
synthesizer. The native layer sets a spoken-audio playback session, volume 1,
and deactivates the session after finish/stop. Real iOS start, finish, cancel,
and failure events drive the existing visible control state. It adds no network
service and does not change OCR, translated values, prescription gates, or
recognized fields. The PWA retains its existing Web Speech implementation.

## Validation

- `node --test app/tools/speech_contract.test.cjs app/tools/receipt_regression.test.mjs` — **25/25**. Includes native bridge voice discovery, English/Cantonese dispatch, event state, Mandarin refusal and stop/retry behavior.
- `bash run.sh test-speech-ui` with the repository's Playwright/Axe tools — real receipt and synthetic prescription OCR, real Chromium speech events, Stop/replay, offline reading and device navigation; output is recorded in `artifacts/speech-repair/`. Browser acoustic output is not used to claim iPhone speaker playback.
- Signed iPhone build: **BUILD SUCCEEDED**.
- `devicectl` installed and launched `com.sightline.reader` on the connected iPhone 13 Pro without uninstalling the app, retaining its local app data.

The app launch and build are verified. A person must still confirm audible sound on
the iPhone; remote build tools cannot hear its speaker. If it is still silent,
the results screen reports the output route and current media volume after the
native start event. Xcode's device-control interface for this iPhone does not
implement its audio-route query capability, so SightLine now reads those values
from `AVAudioSession` locally. The user confirmed the previous build's status
reached “Reading aloud” without audible output. This confirms the synthesizer's
start callback, but does not by itself prove sound reached the speaker. The
diagnostic build was installed and launched again with route/volume reporting.
The next tap will distinguish a muted output, a headphone/Bluetooth route, or a
normal iPhone-speaker route; the last case would require checking the installed
voice and physical speaker path directly.
