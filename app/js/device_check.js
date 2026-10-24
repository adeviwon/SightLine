/** Plain-language checks. API presence never means audible playback passed. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const embedded = window.parent !== window;
  if (embedded) {
    document.body.dataset.embedded = "true";
    try {
      document.documentElement.dataset.theme =
        parent.document.documentElement.dataset.theme || "light";
    } catch {}
  }
  function leave(event) {
    if (embedded) {
      event.preventDefault();
      TTS.stop();
      parent.postMessage({ type: "sightline-device-close" }, location.origin);
    }
  }
  $("device-back").addEventListener("click", leave);
  $("device-return").addEventListener("click", leave);
  window.addEventListener("message", (e) => {
    if (
      e.origin === location.origin &&
      e.source === parent &&
      e.data?.type === "stop-device-voice"
    )
      TTS.stop();
  });
  window.addEventListener("message", (e) => {
    if (
      e.origin === location.origin &&
      e.source === parent &&
      e.data?.type === "device-theme"
    )
      document.documentElement.dataset.theme = e.data.theme;
  });
  document.addEventListener("keydown", (e) => {
    if (embedded && e.key === "Escape") {
      e.preventDefault();
      TTS.stop();
      parent.postMessage({ type: "sightline-device-close" }, location.origin);
    }
  });
  if (window.SightLineI18n) {
    let language = "en-GB";
    try {
      language = embedded
        ? parent.SightLineI18n?.language
        : localStorage.getItem("sightline.language");
    } catch {}
    SightLineI18n.set(language);
    SightLineI18n.observe();
    window.addEventListener("message", (e) => {
      if (
        e.origin === location.origin &&
        e.source === parent &&
        e.data?.type === "device-language"
      )
        SightLineI18n.set(e.data.language);
    });
  }
  function row(title, state, explanation, ok) {
    return { title, state, explanation, ok };
  }
  function draw(rows) {
    $("device-checks").replaceChildren();
    for (const check of rows) {
      const section = document.createElement("section");
      section.className = "device-check";
      const title = document.createElement("h2");
      title.textContent = check.title;
      const status = document.createElement("p");
      status.className = "check-state" + (check.ok ? "" : " attention");
      status.textContent = check.state;
      const detail = document.createElement("p");
      detail.textContent = check.explanation;
      section.append(title, status, detail);
      $("device-checks").append(section);
    }
  }
  async function offlineStatus() {
    try {
      const native =
        window.SightLineNativeOffline ||
        (embedded ? parent.SightLineNativeOffline : null);
      if (
        location.hostname === "127.0.0.1" &&
        native?.verified &&
        native.assetCount > 0 &&
        native.bytes > 0
      )
        return true;
      if (!navigator.serviceWorker) return false;
      const reg = await navigator.serviceWorker.getRegistration();
      const worker = navigator.serviceWorker.controller || reg?.active;
      if (!worker) return false;
      return await new Promise((resolve) => {
        const timer = setTimeout(() => done(false), 3000);
        const message = (e) => {
          if (e.source === worker && e.data?.type === "precache-report")
            done(
              e.data.total > 0 &&
                e.data.failed === 0 &&
                e.data.ok === e.data.total,
            );
        };
        function done(value) {
          clearTimeout(timer);
          navigator.serviceWorker.removeEventListener("message", message);
          resolve(value);
        }
        navigator.serviceWorker.addEventListener("message", message);
        worker.postMessage({ type: "precache-status" });
      });
    } catch {
      return false;
    }
  }
  async function checkModels(installed) {
    if (installed) return true; // The verified required set includes BOTH PP weights and WASM.
    const paths = [
      "models/ppocrv5-mobile-det.onnx",
      "models/ppocrv5-mobile-rec.onnx",
      "models/ppocrv5_dict.txt",
      "vendor/ort/ort-wasm-simd-threaded.wasm",
    ];
    try {
      return (
        await Promise.all(
          paths.map(async (path) => {
            if (
              "caches" in window &&
              (await caches.match(new URL(path, location.href)))
            )
              return true;
            const controller = new AbortController(),
              timer = setTimeout(() => controller.abort(), 4000);
            try {
              return (
                await fetch(path, { method: "HEAD", signal: controller.signal })
              ).ok;
            } finally {
              clearTimeout(timer);
            }
          }),
        )
      ).every(Boolean);
    } catch {
      return false;
    }
  }
  async function run() {
    $("device-recheck").disabled = true;
    $("device-status").textContent =
      "Checking reading files and available voices…";
    try {
      const camera = !!navigator.mediaDevices?.getUserMedia;
      let denied = false;
      try {
        denied =
          (await navigator.permissions?.query({ name: "camera" }))?.state ===
          "denied";
      } catch {}
      // This is the same executable SIMD probe used by the WebKit gate.
      let simd = false;
      try {
        simd = WebAssembly.validate(
          new Uint8Array([
            0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10,
            10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
          ]),
        );
      } catch {}
      const supported =
        typeof WebAssembly === "object" &&
        typeof Worker === "function" &&
        isSecureContext &&
        simd;
      const installed = await offlineStatus();
      const models = await checkModels(installed);
      await TTS.loadVoices();
      const english = TTS.hasVoice("en-GB"),
        cantonese = TTS.hasVoice("zh-HK");
      const rows = [
        row(
          "Camera",
          camera && !denied
            ? "Available when you choose Scan"
            : "Needs attention",
          !camera
            ? "Camera access is unavailable here. Upload a photo instead."
            : denied
              ? "Allow camera access in your device settings, or upload a photo."
              : "SightLine asks permission when you start scanning. No camera was opened by this check.",
          camera && !denied,
        ),
        row(
          "Reading software",
          supported ? "Supported" : "Needs attention",
          supported
            ? "The browser supports on-device reading. Use a real scan to check recognition."
            : "Open the app or a supported browser over HTTPS. You can also try an updated Safari.",
          supported,
        ),
        row(
          "Offline reading",
          installed ? "Reading files saved" : "Not ready yet",
          installed
            ? "The required files are available on this device."
            : "Open SightLine online and wait for Ready offline. Then run this check again.",
          installed,
        ),
        row(
          "Recognition files",
          models ? "Available" : "Files missing",
          models
            ? "The reading models and their runtime files are available. This is not an accuracy test."
            : "Return to SightLine online to finish saving the reading files.",
          models,
        ),
        row(
          "Read Aloud",
          TTS.available() && english
            ? "Offline English voice available"
            : "Needs attention",
          !TTS.available()
            ? "This browser does not support Read Aloud. Try the SightLine app or Safari."
            : !english
              ? TTS.getVoices().length
                ? "Download an English voice in your device speech settings. Then run these checks again."
                : "The voice list is not ready. Run the checks again in a moment. If it stays empty, check your device speech settings."
              : `An installed English voice is available. ${cantonese ? "Cantonese is available too." : "Cantonese needs a separately installed voice."} Test the voice below to check sound.`,
          TTS.available() && english,
        ),
      ];
      draw(rows);
      $("test-voice").disabled = !(TTS.available() && english);
      $("device-status-title").textContent =
        supported && models
          ? "Ready to try reading"
          : "Some checks need attention";
      $("device-status").textContent =
        supported && models
          ? english
            ? "Your reading files are available. Test the voice before listening."
            : "Reading is available. Install an offline English voice to listen."
          : "Review the steps below. Uploading a photo may still be available.";
      window.SightLineDeviceReport = {
        rows,
        installed,
        models,
        english,
        cantonese,
        audioVerified:
          english && window.SightLineDeviceReport?.audioVerified === true,
        checkedAt: Date.now(),
      };
    } catch {
      $("device-status-title").textContent = "Checks could not finish";
      $("device-status").textContent =
        "Try the checks again. You can return to SightLine at any time.";
    } finally {
      $("device-recheck").disabled = false;
    }
  }
  $("device-recheck").addEventListener("click", run);
  $("test-voice").addEventListener("click", async () => {
    if (TTS.speaking()) {
      TTS.stop();
      return;
    }
    $("voice-confirmation").hidden = true;
    if (window.SightLineDeviceReport)
      window.SightLineDeviceReport.audioVerified = false;
    try {
      const r = await TTS.speak(
        "This is SightLine. I can read clear details aloud.",
        { lang: "en-GB" },
      );
      if (r.status === "ended") $("voice-confirmation").hidden = false;
    } catch (e) {
      $("voice-status").textContent = TTS.errorMessage(e);
    }
  });
  TTS.subscribe((s) => {
    $("test-voice").textContent = [
      "preparing",
      "starting",
      "speaking",
    ].includes(s.status)
      ? "Stop voice test"
      : "Test voice";
    const text = {
      preparing: "Finding an offline voice…",
      starting: "Starting the voice…",
      speaking: "The voice is reading. Can you hear it?",
      ended: "The voice test finished. Confirm below whether you heard it.",
      stopped: "Voice test stopped. You can try again.",
    };
    $("voice-status").textContent =
      s.status === "error"
        ? TTS.errorMessage({ code: s.error })
        : text[s.status] || "";
  });
  $("voice-heard").addEventListener("click", () => {
    $("voice-status").textContent = "You confirmed that you heard the voice.";
    $("voice-confirmation").hidden = true;
    if (window.SightLineDeviceReport)
      window.SightLineDeviceReport.audioVerified = true;
  });
  $("voice-unheard").addEventListener("click", () => {
    $("voice-status").textContent =
      "Check the volume, silent mode and connected headphones. Then try Test voice again.";
    $("voice-confirmation").hidden = true;
    if (window.SightLineDeviceReport)
      window.SightLineDeviceReport.audioVerified = false;
  });
  window.addEventListener("pagehide", () => TTS.stop());
  // Voice catalogs can arrive after the initial bounded discovery window.
  window.speechSynthesis?.addEventListener("voiceschanged", () => {
    if (!$("device-recheck").disabled && !window.SightLineDeviceReport?.english)
      run();
  });
  run();
})();
