/**
 * SightLine TTS — offline speech output.
 *
 * Uses the platform Web Speech API (speechSynthesis), which iOS Safari and
 * Android Chrome both run fully on-device. No audio bytes are bundled, so this
 * costs 0 KB of precache and works in airplane mode.
 *
 * iOS truncates utterances beyond ~200 characters, so text is split on
 * sentence boundaries and queued. This mirrors the tested reference
 * implementation in SightLine-Mobile/js/tts.js.
 */

"use strict";

const TTS = (() => {

  let _current = null;

  function available() {
    return typeof window !== "undefined" && "speechSynthesis" in window;
  }

  /**
   * speak(): read text aloud, resolved when finished.
   * @param {string} text
   * @param {{rate?:number, pitch?:number, lang?:string, onEnd?:Function}} [opts]
   */
  function speak(text, opts) {
    const o = opts || {};
    if (!available()) return Promise.reject(new Error("speechSynthesis unavailable"));
    const clean = String(text || "").trim();
    if (!clean) return Promise.resolve();
    stop();
    return new Promise((resolve, reject) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        if (o.onEnd) o.onEnd();
        resolve();
      };
      const fail = (e) => {
        if (settled) return;
        settled = true;
        reject(new Error("TTS error: " + ((e && e.error) || "unknown")));
      };
      try {
        const chunks = chunkText(clean, 180);
        const voices = pickVoice(o.lang || "en-GB");
        let last = null;
        for (const chunk of chunks) {
          const u = new SpeechSynthesisUtterance(chunk);
          u.rate = o.rate != null ? o.rate : 1.0;
          u.pitch = o.pitch != null ? o.pitch : 1.0;
          u.volume = 1.0;
          u.lang = o.lang || "en-GB";
          if (voices) u.voice = voices;
          last = u;
        }
        _current = last;
        if (last) {
          last.onend = finish;
          last.onerror = fail;
        }
        for (const chunk of chunks) {
          // Re-create utterances: the queue holds the object, not the string.
          const u = new SpeechSynthesisUtterance(chunk);
          u.rate = o.rate != null ? o.rate : 1.0;
          u.pitch = o.pitch != null ? o.pitch : 1.0;
          u.volume = 1.0;
          u.lang = o.lang || "en-GB";
          if (voices) u.voice = voices;
          window.speechSynthesis.speak(u);
        }
        // Some WebViews fire no events at all — never leave the UI stuck.
        setTimeout(finish, 400 + clean.length * 90);
      } catch (e) {
        fail(e);
      }
    });
  }

  function pickVoice(lang) {
    if (!available()) return null;
    const voices = window.speechSynthesis.getVoices() || [];
    if (!voices.length) return null;
    const want = lang.slice(0, 2).toLowerCase();
    return voices.find((v) => v.lang && v.lang.toLowerCase() === lang.toLowerCase())
      || voices.find((v) => v.lang && v.lang.toLowerCase().indexOf(want) === 0)
      || voices.find((v) => v.lang && v.lang.toLowerCase().indexOf("en") === 0)
      || voices[0];
  }

  /** Sentence-aware chunker with a hard wrap for run-on strings. */
  function chunkText(text, maxLen) {
    const src = String(text || "").replace(/\s+/g, " ").trim();
    if (!src) return [];
    const sentences = src.split(/(?<=[.!?])\s+/);
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
          for (let i = 0; i < s.length; i += maxLen) chunks.push(s.slice(i, i + maxLen));
          cur = "";
        }
      }
    }
    if (cur) chunks.push(cur);
    return chunks.length ? chunks : [src.slice(0, maxLen)];
  }

  function stop() {
    if (available()) {
      try { window.speechSynthesis.cancel(); } catch (e) { /* nothing queued */ }
    }
    _current = null;
  }

  function speaking() {
    return available() && window.speechSynthesis.speaking;
  }

  return { speak, stop, speaking, available, chunkText };
})();

if (typeof module !== "undefined" && module.exports) module.exports = TTS;
if (typeof window !== "undefined") window.TTS = TTS;
