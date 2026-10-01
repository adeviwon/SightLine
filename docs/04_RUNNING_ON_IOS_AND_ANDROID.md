# Running SightLine on iOS and Android

SightLine is a **Progressive Web App (PWA)**. That is a deliberate choice over
a native build, and it is the right one for a 36-hour hackathon:

| | Native (React Native / Swift / Kotlin) | PWA (this) |
|---|---|---|
| Time to build | 8–14 h, plus app-store setup | ~2 h |
| Judge installs it | App Store review, or sideloading | **A URL.** Opens instantly. |
| Runs on both platforms | Two builds | **One build.** |
| Offline | Bundle with the binary | Service worker precache |
| ONNX inference | Native module (build risk) | `onnxruntime-web` WASM |
| Camera access | Native permission flows | Standard `getUserMedia` |
| OCR offline | Bundled native lib | Tesseract WASM (vendored) |

The trade-off is honest: a native build would be faster and could use the
phone's NPU. Neither matters for a document scanner, which is
latency-tolerant and already fast enough.

---

## 1. Serve the app

SightLine is a static site. Any static file server works.

```bash
bash run.sh serve          # defaults to port 8080
bash run.sh serve 3000     # or pick a port
```

Output looks like:

```
Serving ./app on http://0.0.0.0:8080
  phone on the SAME wifi: http://192.168.1.42:8080
  iOS:     Safari -> Share -> Add to Home Screen
  Android: Chrome -> menu -> Install app
```

**Your phone and the computer must be on the same Wi-Fi network.** This is the
single most common reason a demo fails.

Other static hosts that work identically:

```bash
# From anywhere in the world (needs internet, not offline)
npx serve app
python3 -m http.server 8080 --directory app

# GitHub Pages: push app/ to a repo, enable Pages in settings
```

---

## 2. iOS (iPhone / iPad)

1. **Open Safari** on the phone (Chrome on iOS will not install a PWA).
2. Go to `http://<computer-ip>:8080`.
3. Wait for the status line to read **`ENGINE READY`**. First load downloads
   the OCR engine and language data (~15–40 MB depending on connection).
4. Tap the **Share** button (square with an arrow pointing up).
5. Tap **Add to Home Screen** → **Add**.
6. Open SightLine from your home screen.

### iOS specifics you must know before the demo

- **Airplane mode works** after the first successful load. Verify this during
  your rehearsal, not during the presentation.
- **The service worker cache can be evicted** by iOS if you go weeks without
  opening the app. Reload once online before a demo.
- **Camera permission** is requested on first use. If you pre-seeded the demo
  with a gallery image, avoid triggering it unexpectedly.
- **HTTPS is required for the camera on iOS.** `http://localhost` and
  `http://<LAN-IP>` both work in current iOS for `getUserMedia`; older iOS
  (< 14) will refuse. If in doubt, test on the actual demo device.
- **Do not use private browsing** for the demo — service workers are disabled.

---

## 3. Android (Chrome / Edge)

1. Open **Chrome** on the phone.
2. Go to `http://<computer-ip>:8080`.
3. Wait for **`ENGINE READY`**.
4. Tap the **⋮ menu** → **Install app** (or **Add to Home screen**).
5. Launch from your home screen.

### Android specifics

- Chrome prompts to install once the app meets the PWA criteria. You can also
  force it from **⋮ → Install app**.
- **Airplane mode works** after the first load.
- Camera permission is requested on first use.
- If the install prompt does not appear, `chrome://flags` is not required —
  the app is served over plain HTTP on a LAN IP, which is sufficient for
  installation.

---

## 4. Verifying it works offline

Do this on both phones during rehearsal:

1. Load the app, wait for `ENGINE READY`.
2. Scan one document successfully.
3. **Turn on airplane mode.**
4. Force-quit the app (swipe it away from recents).
5. Reopen from the home screen.
6. Scan again. It must work.

If step 6 fails, the service worker did not precache something. Check the
browser console for failed `cache.addAll` entries — every asset must be in the
precache list in `app/sw.js`.

---

## 5. Using it

1. **Frame the document.** The app draws corner guides. Fill the frame.
2. **Tap Scan.** Restoration and OCR take under a second on a modern phone.
3. **Read the result.** SightLine announces the document type, then the
   extracted fields, then offers to read the full text.
4. **Tap Listen** to hear the plain-language summary.
5. **Not readable?** SightLine says so and asks you to retake. Move somewhere
   brighter, or steady the phone against the table.

---

## 6. Adding your own test documents

Use documents that resemble what a user would actually photograph:

- A **prescription** from a pharmacy (medication name + dosage + frequency)
- A **bank statement** (account number + sort code + balance)
- A **letter** (correspondence, dates, reference numbers)

Print them. Photograph them badly on purpose — that is the whole point.

**Do not use real documents containing real account numbers or medical data**
in a public demo or a recorded video.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Camera does not open | Not HTTPS, or permission denied | Reload; grant permission in Settings; use `localhost` or a LAN IP |
| `ENGINE` never reaches READY | Precache failed | Hard-reload; check console for failed `cache.addAll` |
| Blank white page | Serving from the wrong directory | Must serve `app/`, not the repo root |
| Works online, fails offline | Service worker not installed | Confirm HTTPS/localhost; unregister old SW in DevTools |
| OCR very slow first run | Downloading language data | Wait once, then cache persists |
| Install prompt absent | iOS needs Safari; Android needs Chrome | Not Firefox, not in-app browsers |
| No sound | Device muted / TTS not downloaded | Check media volume; some Android devices download voices on first use |

See [`08_TROUBLESHOOTING.md`](08_TROUBLESHOOTING.md) for the Python side.
