/** Local platform speech. Completion means an end event, never a timer guess.
 * Voices are discovered early; a ready voice speaks synchronously in the tap.
 * Only local voices are selected: speech must not require an online service.
 */
"use strict";
const TTS = (() => {
  let session = null,
    voiceLoad = null;
  let nativeVoices = null;
  let state = { status: "idle", error: null };
  const listeners = new Set();
  const available = () =>
    typeof window !== "undefined" &&
    (!!nativeBridge() ||
      (!!window.speechSynthesis &&
        typeof window.SpeechSynthesisUtterance === "function"));
  function nativeBridge() {
    if (typeof window === "undefined") return null;
    return window.webkit?.messageHandlers?.sightlineSpeech || null;
  }
  function getVoices() {
    if (nativeBridge()) return nativeVoices || [];
    try {
      return available() ? window.speechSynthesis.getVoices() || [] : [];
    } catch {
      return [];
    }
  }
  function pickVoice(lang = "en-GB") {
    const local = getVoices().filter((v) => v.localService === true);
    const exact = (v) => v.lang?.toLowerCase() === lang.toLowerCase();
    if (lang === "zh-HK")
      return (
        local.find(exact) ||
        local.find((v) => /^yue(?:-|$)/i.test(v.lang)) ||
        null
      );
    return (
      local.find(exact) ||
      local.find(
        (v) =>
          v.lang?.split("-")[0].toLowerCase() ===
          lang.split("-")[0].toLowerCase(),
      ) ||
      null
    );
  }
  function loadVoices(waitMs = 1500) {
    const bridge = nativeBridge();
    if (bridge) {
      // An empty first catalog is common before iOS finishes loading voices.
      // Keep retrying on demand until at least one voice is available.
      if (nativeVoices && nativeVoices.length) return Promise.resolve(nativeVoices);
      if (voiceLoad) return voiceLoad;
      voiceLoad = new Promise((resolve) => {
        const timer = setTimeout(() => {
          voiceLoad = null;
          resolve(nativeVoices || []);
        }, waitMs);
        nativeVoiceResolve = (voices) => {
          clearTimeout(timer);
          nativeVoiceResolve = null;
          voiceLoad = null;
          resolve(voices);
        };
        bridge.postMessage({ action: "voices" });
      });
      return voiceLoad;
    }
    if (getVoices().length || !available()) return Promise.resolve(getVoices());
    if (voiceLoad) return voiceLoad;
    voiceLoad = new Promise((resolve) => {
      const synth = window.speechSynthesis;
      let timer, poll;
      const done = () => {
        clearTimeout(timer);
        clearInterval(poll);
        synth.removeEventListener?.("voiceschanged", changed);
        voiceLoad = null;
        resolve(getVoices());
      };
      const changed = () => {
        if (getVoices().length) done();
      };
      synth.addEventListener?.("voiceschanged", changed);
      poll = setInterval(changed, 100);
      timer = setTimeout(done, waitMs);
    });
    return voiceLoad;
  }
  let nativeVoiceResolve = null;
  function emit(status, detail = {}) {
    state = { ...state, ...detail, status };
    for (const listener of listeners) {
      try {
        listener({ ...state });
      } catch {}
    }
    try {
      session?.options.onState?.({ ...state });
    } catch {}
  }
  function errorMessage(error) {
    const code = error?.code || error?.error || "";
    return (
      {
        unavailable:
          "Read-aloud is not supported here. Try Safari, or ask someone to read the document with you.",
        "voices-not-ready":
          "The voice list is not ready yet. Tap Read Aloud again in a moment. If this continues, open Check Device.",
        "no-local-voice":
          "No offline voice is available for this language. Download a voice in your device speech settings, then try again.",
        "not-allowed": "Tap Read Aloud again to allow speech to start.",
        "audio-busy":
          "Another app is using the sound output. Stop its audio, then try again.",
        "audio-hardware": "Check your speaker or headphones, then try again.",
        "voice-unavailable":
          "The selected voice is unavailable. Check your device speech settings and try again.",
        "language-unavailable":
          "Install a voice for this language in your device speech settings.",
        "start-timeout":
          "The voice did not start. Check your volume and speech settings, then try again.",
        "end-timeout":
          "Reading stopped responding. Tap Read Aloud to try again.",
        network:
          "This voice could not work offline. Download an offline voice in your device speech settings.",
      }[code] ||
      "Reading could not finish. Check your device speech settings, then try again."
    );
  }
  function stop() {
    const old = session;
    if (old) {
      old.closed = true;
      clearTimeout(old.timer);
      session = null;
      old.resolve({ status: "stopped" });
    }
    try {
      if (nativeBridge()) nativeBridge().postMessage({ action: "stop" });
      else if (available()) window.speechSynthesis.cancel();
    } catch {}
    if (old)
      emit("stopped", {
        error: null,
        voice: null,
        lang: null,
        chunk: 0,
        chunks: 0,
      });
  }
  function speak(text, options = {}) {
    const clean = String(text || "").trim();
    if (!available())
      return Promise.reject(
        Object.assign(new Error(errorMessage({ code: "unavailable" })), {
          code: "unavailable",
        }),
      );
    if (!clean)
      return Promise.reject(
        Object.assign(new Error("There are no clear words to read."), {
          code: "empty-text",
        }),
      );
    stop();
    return new Promise((resolve, reject) => {
      const job = {
        resolve,
        reject,
        options,
        text: clean,
        closed: false,
        timer: null,
        utterance: null,
      };
      session = job;
      const current = () => session === job && !job.closed;
      function finish(status, error) {
        if (!current()) return;
        job.closed = true;
        clearTimeout(job.timer);
        if (error) {
          try {
            if (job.nativeId && nativeBridge())
              nativeBridge().postMessage({ action: "stop" });
            else window.speechSynthesis.cancel();
          } catch {}
        }
        emit(status, { error: error?.code || null });
        session = null;
        if (error) reject(error);
        else {
          try {
            options.onEnd?.();
          } catch {}
          resolve({ status });
        }
      }
      function fail(code) {
        finish(
          "error",
          Object.assign(new Error(errorMessage({ code })), { code }),
        );
      }
      job.finish = finish;
      job.fail = fail;
      if (nativeBridge()) {
        const voice = pickVoice(options.lang || "en-GB");
        if (!voice) {
          fail(nativeVoices ? "no-local-voice" : "voices-not-ready");
          return;
        }
        const id = String(Date.now()) + "-" + Math.random().toString(36).slice(2);
        job.nativeId = id;
        emit("starting", {
          error: null,
          voice: voice.name,
          lang: voice.lang,
          chunks: 1,
          chunk: 0,
        });
        job.timer = setTimeout(
          () => fail("start-timeout"),
          options.startTimeoutMs ?? 8000,
        );
        try {
          nativeBridge().postMessage({
            action: "speak",
            id,
            text: clean,
            lang: options.lang || "en-GB",
            voice: voice.name,
            voiceLang: voice.lang,
            rate: options.rate ?? 0.95,
          });
        } catch (error) {
          fail(error?.code || "synthesis-failed");
        }
        return;
      }
      function begin() {
        if (!current()) return;
        const voice = pickVoice(options.lang || "en-GB");
        if (!voice) {
          fail(getVoices().length ? "no-local-voice" : "voices-not-ready");
          return;
        }
        const chunks = chunkText(clean, 180);
        emit("starting", {
          error: null,
          voice: voice.name,
          lang: voice.lang,
          chunks: chunks.length,
          chunk: 0,
        });
        function next(index) {
          if (!current()) return;
          if (index === chunks.length) {
            finish("ended");
            return;
          }
          const utterance = new window.SpeechSynthesisUtterance(chunks[index]);
          job.utterance = utterance; // Retain the active utterance until its event.
          utterance.voice = voice;
          utterance.lang = options.lang === "zh-HK" ? "zh-HK" : voice.lang;
          utterance.rate = options.rate ?? 0.95;
          utterance.pitch = options.pitch ?? 1;
          utterance.volume = 1;
          utterance.onstart = () => {
            if (!current()) return;
            clearTimeout(job.timer);
            emit("speaking", { chunk: index + 1 });
            job.timer = setTimeout(
              () => fail("end-timeout"),
              Math.max(
                30000,
                chunks[index].length * (options.lang === "zh-HK" ? 800 : 300),
              ),
            );
          };
          utterance.onend = () => {
            if (current()) {
              clearTimeout(job.timer);
              next(index + 1);
            }
          };
          utterance.onerror = (e) => {
            if (current()) fail(e.error || "synthesis-failed");
          };
          // Handlers and deadline exist BEFORE queuing, including the first chunk.
          job.timer = setTimeout(
            () => fail("start-timeout"),
            options.startTimeoutMs ?? 8000,
          );
          try {
            window.speechSynthesis.resume?.();
            window.speechSynthesis.speak(utterance);
          } catch (e) {
            fail(e.error || e.code || "synthesis-failed");
          }
        }
        next(0);
      }
      if (getVoices().length)
        begin(); // Keep this on the original user gesture.
      else {
        emit("preparing", { error: null });
        loadVoices(options.voiceWaitMs ?? 1500).then(begin, () =>
          fail("voice-unavailable"),
        );
      }
    });
  }
  function nativeEvent(event) {
    if (event?.type === "voices") {
      nativeVoices = Array.isArray(event.voices) ? event.voices : [];
      nativeVoiceResolve?.(nativeVoices);
      return;
    }
    const job = session;
    if (!job || !job.nativeId || event?.id !== job.nativeId) return;
    if (event.type === "started") {
      clearTimeout(job.timer);
      emit("speaking", {
        chunk: 1,
        outputKind: event.outputKind || null,
        outputName: event.outputName || null,
        voiceName: event.voiceName || null,
        outputVolume: Number.isFinite(event.outputVolume)
          ? event.outputVolume
          : null,
      });
      job.timer = setTimeout(
        () => job.fail("end-timeout"),
        Math.max(
          30000,
          job.text.length * (job.options.lang === "zh-HK" ? 800 : 300),
        ),
      );
    } else if (event.type === "finished") {
      job.finish("ended");
    } else if (event.type === "stopped") {
      job.finish("stopped");
    } else if (event.type === "error") {
      job.fail(event.code || "synthesis-failed");
    }
  }
  /** Sentence-aware chunker with a hard wrap for run-on strings. */
  function chunkText(text, maxLen = 180) {
    maxLen = Math.max(1, Math.min(180, Math.floor(maxLen) || 180));
    const src = String(text || "")
      .replace(/\s+/g, " ")
      .trim();
    if (!src) return [];
    // Sentence split WITHOUT a lookbehind. Sentence chunking used to split on
    // a lookbehind-anchored whitespace pattern, and regex lookbehind is a
    // PARSE-TIME SyntaxError on Safari/iOS before 16.4. tts.js is a classic
    // script, so that error took the whole file down: window.TTS never existed
    // and every read-aloud call threw on exactly the older phones this product
    // targets. A global match is equivalent here and keeps the terminator with
    // its sentence; chunkText trims, so the leading space is irrelevant.
    // (See tools/ios_compat_scan.js, which now fails the build on this class
    // of syntax — do not reintroduce it.)
    const sentences = src.match(/[\s\S]+?(?:[.!?](?:\s+|$)|[。！？]|$)/g) || [
      src,
    ];
    const chunks = [];
    let cur = "";
    for (const s of sentences) {
      if ((cur + " " + s).trim().length <= maxLen) {
        cur = (cur + " " + s).trim();
      } else {
        if (cur) chunks.push(cur);
        if (s.length <= maxLen) {
          cur = s;
        } else {
          for (let i = 0; i < s.length; i += maxLen)
            chunks.push(s.slice(i, i + maxLen));
          cur = "";
        }
      }
    }
    if (cur) chunks.push(cur);
    return chunks.length ? chunks : [src.slice(0, maxLen)];
  }
  return {
    speak,
    nativeEvent,
    stop,
    stop,
    available,
    nativeBackend: () => !!nativeBridge(),
    loadVoices,
    getVoices,
    hasVoice: (lang) => !!pickVoice(lang),
    pickVoice,
    chunkText,
    errorMessage,
    speaking: () => !!session,
    getState: () => ({ ...state }),
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
})();
if (typeof module !== "undefined" && module.exports) module.exports = TTS;
if (typeof window !== "undefined") window.TTS = TTS;
