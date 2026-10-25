/**
 * SightLine App — UI controller.
 *
 * Owns the DOM: capture sources (camera / file / PDF), the scan orchestration,
 * the result panel, and the accessibility announcements. It contains no image
 * processing and no classification logic — those live in pipeline.js (always)
 * with optional acceleration from restorer.js / classifier.js.
 *
 * ACCESSIBILITY CONTRACT (the users this app is for cannot read the screen):
 *   - every control is a real <button>/<input> with an accessible name
 *   - min 48x48 CSS px tap targets
 *   - progress and results are announced through a polite aria-live region
 *   - the whole result panel is keyboard-reachable and does not rely on colour
 *     alone (confidence is always rendered as a number + word)
 *   - honours prefers-reduced-motion and prefers-contrast
 */

"use strict";

(function () {

  const $ = (id) => document.getElementById(id);
  const els = {};
  let stream = null;
  let busy = false;
  let lastResult = null;

  // ── Announcements ─────────────────────────────────────────────────────

  /**
   * Screen-reader announcements.
   *
   * Two separate live regions rather than one region whose aria-live is
   * mutated. Flipping the attribute on a single node leaves it assertive for
   * every later update, so routine progress messages end up interrupting the
   * user — and a real screen reader has already committed to the polite queue.
   * Errors go to an assertive region; progress and results to a polite one
   * that never changes.
   */
  function announce(msg, assertive) {
    const el = assertive ? els.liveAssertive : els.live;
    if (!el) return;
    // Clearing first forces re-announcement when the text is unchanged.
    el.textContent = "";
    window.setTimeout(() => { el.textContent = msg; }, 30);
  }

  function setProgress(pct, label, measured) {
    const known = measured && measured.total > 0;
    const p = known ? Math.round(measured.done / measured.total * 100) : pct === 100 ? 100 : null;
    if (els.progressFill) els.progressFill.style.width = p === null ? "100%" : p + "%";
    if (els.progressTrack) {
      if (p === null) els.progressTrack.removeAttribute("aria-valuenow");
      else els.progressTrack.setAttribute("aria-valuenow", String(p));
      els.progressTrack.setAttribute("aria-valuetext", label || "");
      els.progressTrack.dataset.indeterminate = String(p === null);
    }
    if (els.progressLabel) els.progressLabel.textContent = label || "";
  }

  function showProgress(on) {
    if (els.progressWrap) els.progressWrap.hidden = !on;
  }

  function setBadge(text, state) {
    if (!els.badge) return;
    els.badge.textContent = text;
    els.badge.dataset.state = state || "ok";
  }

  function showError(msg) {
    if (!els.errorBanner) return;
    if (!msg) { els.errorBanner.hidden = true; els.errorBanner.textContent = ""; return; }
    els.errorBanner.hidden = false;
    els.errorBanner.textContent = msg;
    announce("Error: " + msg, true);
  }

  // ── Camera ────────────────────────────────────────────────────────────

  async function startCamera() {
    showError("");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("This browser does not expose a camera. Use Choose a Photo instead.");
      return false;
    }
    stopCamera();
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 3840 }, height: { ideal: 2160 } },
        audio: false,
      });
      els.video.srcObject = stream;
      els.video.hidden = false;
      els.preview.hidden = true;
      els.viewfinder.dataset.mode = "camera";
      setTab("cam");
      const placeholder = document.getElementById("camera-placeholder");
      if (placeholder) placeholder.hidden = true;
      els.btnCapture.textContent = "Read this label";
      announce("Camera ready. Hold the label inside the frame, then tap Read this label.");
      return true;
    } catch (e) {
      const name = e && e.name;
      showError(name === "NotAllowedError"
        ? "Camera permission denied. Use Choose a Photo to pick an existing picture."
        : "Could not start the camera (" + (name || "unknown") + "). Use Choose a Photo.");
      return false;
    }
  }

  function stopCamera() {
    if (stream) {
      for (const t of stream.getTracks()) { try { t.stop(); } catch (e) { /* already stopped */ } }
      stream = null;
    }
    if (els.video) els.video.srcObject = null;
  }

  function captureFrame() {
    if (!stream || !els.video.videoWidth) {
      showError("No camera frame available. Start the camera or choose a photo.");
      return null;
    }
    const c = document.createElement("canvas");
    c.width = els.video.videoWidth;
    c.height = els.video.videoHeight;
    c.getContext("2d").drawImage(els.video, 0, 0);
    els.preview.src = c.toDataURL("image/jpeg", 0.9);
    els.preview.hidden = false;
    els.viewfinder.dataset.mode = "preview";
    return c;
  }

  // ── File + PDF ────────────────────────────────────────────────────────

  function pickFile() {
    if (els.fileInput) els.fileInput.click();
  }

  async function onFile(e) {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    showError("");
    lastResult = null;
    els.results.hidden = true;
    els.btnSpeak.hidden = true;
    document.body.classList.remove("has-results");
    stopCamera();
    if (f.type === "application/pdf" || /\.pdf$/i.test(f.name)) {
      await scanPDF(f);
    } else {
      const url = URL.createObjectURL(f);
      els.preview.src = url;
      els.preview.hidden = false;
      const placeholder = document.getElementById("camera-placeholder");
      if (placeholder) placeholder.hidden = true;
      els.video.hidden = true;
      els.viewfinder.dataset.mode = "preview";
      try {
        await new Promise((res, rej) => {
          if (els.preview.complete && els.preview.naturalWidth) return res();
          els.preview.onload = res;
          els.preview.onerror = () => rej(new Error("decode"));
        });
        await runScan(els.preview, f.name);
      } catch (err) {
        showError("Could not read that image file.");
      } finally {
        // Revoke late: the scan is still reading pixels off the element.
        window.setTimeout(() => URL.revokeObjectURL(url), 60000);
      }
    }
    els.fileInput.value = "";
  }

  /**
   * scanPDF(): render page 1 at 3x scale to a canvas, then scan.
   * pdf.js is vendored; the worker is same-origin so it stays offline.
   */
  async function scanPDF(file) {
    const pdfjs = window.pdfjsLib;
    if (!pdfjs) {
      showError("PDF reading is unavailable. Try choosing a photo instead.");
      return;
    }
    setProgress(5, "Opening PDF…");
    announce("Opening PDF.");
    try {
      const buf = await file.arrayBuffer();
      const doc = await pdfjs.getDocument({ data: buf }).promise;
      const page = await doc.getPage(1);
      const base = page.getViewport({ scale: 1 });
      const scale = Math.min(4, 3200 / Math.max(base.width, base.height));
      const vp = page.getViewport({ scale });
      const c = document.createElement("canvas");
      c.width = Math.round(vp.width);
      c.height = Math.round(vp.height);
      await page.render({ canvasContext: c.getContext("2d"), viewport: vp }).promise;
      els.preview.src = c.toDataURL("image/jpeg", 0.92);
      els.preview.hidden = false;
      const placeholder = document.getElementById("camera-placeholder");
      if (placeholder) placeholder.hidden = true;
      els.video.hidden = true;
      els.viewfinder.dataset.mode = "preview";
      const total = doc.numPages;
      doc.destroy();
      await runScan(c, file.name, total > 1 ? "Page 1 of " + total : null);
    } catch (e) {
      showError("Could not render that PDF: " + ((e && e.message) || "unknown"));
      setProgress(0, "");
      showProgress(false);
    }
  }

  // ── Scan orchestration ────────────────────────────────────────────────

  async function runScan(source, name, note) {
    if (busy) return null;
    busy = true;
    setScanEnabled(false);
    showError("");
    showProgress(true);
    setProgress(2, "Getting ready to read.");
    announce("Scanning. This usually takes a few seconds.");
    els.scanLine.classList.add("on");
    els.scanFlash.classList.add("on");
    window.setTimeout(() => els.scanFlash.classList.remove("on"), 200);

    const t0 = performance.now();
    try {
      // Which restoration to use is decided by whether the model exists, not
      // by the image. With no restorer.onnx, "classical" (evaluate.py's
      // arm_classical: median + unsharp, unconditionally) beats "adaptive",
      // which applies nothing to a moderately blurred capture that sits just
      // above the blurry threshold. Measured in tools/e2e_ocr.js.
      const readingMethod = els.ocrModel ? els.ocrModel.value : "crnn";
      const onnxRestorer = readingMethod === "crnn" && !els.skipRestore.checked;
      const restoreMode = readingMethod === "ppocrv5" ? "none" : onnxRestorer ? "adaptive" : "classical";

      // Stage 1: preprocessing is unconditional. The restorer and the
      // classifier are accelerators layered on top, never prerequisites.
      const pp = SightLine.preprocess(source, 3200, { restoreMode });
      setProgress(15, "Making the picture clearer.");
      await frame();

      // Stage 2: optional neural restoration. Falls back silently.
      let ocrCandidates = pp.candidates;
      let engine = { restorer: restoreMode, classifier: "skipped" };
      if (onnxRestorer) {
        setProgress(20, "Restoring image…");
        const r = await SightLineRestorer.armRestorerClahe(pp.canvas, {
          onProgress: (m) => {
            if (m && m.total) setProgress(20 + (m.done / m.total) * 10, "Restoring " + m.done + "/" + m.total);
          },
        });
        engine.restorer = r.engine;
        // PREPEND THE UNPROCESSED IMAGE. This used to REPLACE the list, and
        // armRestorerClahe's three candidates contain no raw slot, so the ranking
        // could only ever choose among degraded variants -- it never saw the deskewed
        // grayscale at all. Measured by holding the crop set, the crop geometry and
        // the model fixed and varying ONLY the page transform, on the app's own 12
        // documents (145 paired crops):
        //     raw          exact 45.5%  lineCER 0.161  word 49.8%  fields 12/24
        //     tile_clahe   exact 40.7%  lineCER 0.180  word 46.8%  fields 12/24
        //     clahe_otsu   exact 13.8%  lineCER 0.444  word 25.0%  fields  5/24
        //     restored     exact 11.0%  lineCER 0.610  word 20.2%  fields  3/24
        // and the shipped ranking picked one of the three RESTORED arms on 12 of 12
        // documents, costing 34.5 points of exact line match, 29.6 points of word
        // accuracy and 9 of 24 fields -- while not even winning on structure_score,
        // the ranking's OWN objective (103 tokens against raw's 146).
        //
        // The ranking rule is deliberately NOT touched, and does not need to be: raw
        // wins on structure_score outright, so the existing rule selects it as soon as
        // it is allowed to see it. Changing the candidate LIST does not weaken the
        // guard; changing the RULE would.
        //
        // This is not a new idea -- pipeline.js's own comment documents exactly this
        // fix for the non-restorer path ("THE UNPROCESSED IMAGE IS A CANDIDATE, and it
        // used to be unreachable ... Ranking is unchanged; structure_score still
        // chooses, now over three candidates instead of two"). app.js's ONNX branch is
        // what silently re-introduced the omission by substituting the restorer's list.
        if (r.candidates) ocrCandidates = [pp.candidates[0]].concat(r.candidates);
      }
      await frame();

      // Stage 3: OCR.
      setProgress(32, "Getting ready to read.");
      const ocr = await SightLine.runOCR(ocrCandidates, (m) => {
        setProgress(null, readingMethod === "ppocrv5" ? (m.status || "Reading the words.") : "Reading the words.", m);
      }, { engine: readingMethod });
      engine.ocr = ocr.engine;
      setProgress(null, "Finding the useful details.");

      // Stage 4: classification — model if present, keywords otherwise.
      let cls = null;
      if (!els.skipClassify.checked) {
        setProgress(null, "Finding the useful details.");
        cls = await SightLineClassifier.classify(ocr.text);
        engine.classifier = cls ? "onnx" : "keywords";
      } else {
        engine.classifier = "keywords";
      }
      await frame();

      // Stage 5: understanding — identical code path in both cases.
      const u = SightLine.understand(ocr.text, cls ? {
        category: cls.category, confidence: cls.confidence, source: "minilm",
      } : undefined);

      // Enhanced reading keeps complete lines only when every word clears
      // 0.75. Use that same conservative read for fields AND Cantonese speech;
      // never stitch a dose out of the remaining words of an uncertain line.
      if (readingMethod === "ppocrv5") {
        const trusted = SightLine.understand(ocr.speechText || "", {
          category: u.category, confidence: u.classificationConfidence, source: u.classificationSource,
        });
        u.fields = SightLine.retainVerbatim(trusted.fields, ocr.speechText || "");
        if (u.category === "medical") u.fields = SightLine.medicalFieldContext(u.fields, ocr.speechText || "");
        u.entities = SightLine.retainVerbatim(trusted.entities, ocr.speechText || "", "text");
        u.summary = SightLine.buildSummary(u.category, u.fields);
      }

      // Honesty gate on the RECOGNISED text, which is where the safety risk
      // lives. preprocess() only knows about image statistics; a capture can
      // look fine (lapVar 133) and still OCR at 26%, and on a prescription that
      // is the difference between "take 500mg" and "take 300mg".
      //
      // The pipeline deliberately never guesses a missing field, so the risk is
      // not a fabricated dosage — it is a user believing an incomplete one.
      // Say so, rather than letting a confident-looking card imply certainty.
      const warnings = pp.warnings.slice();
      if (ocr.confidence < 0.40) {
        warnings.push("This picture was hard to read. "
          + "Some details may be missing or wrong — check the original document, "
          + "and retake the photo in better light if a figure matters.");
      } else if (ocr.confidence < 0.60) {
        warnings.push("Some words are unclear. Check important figures against the original document.");
      }
      if (u.fields.length === 0) {
        warnings.push("No key details could be extracted. Try a closer, "
          + "straighter photo with the whole page in frame.");
      }
      if (readingMethod === "ppocrv5" && !ocr.speechText) {
        warnings.push("Not clear enough to read aloud. Try another photo in better light.");
      }
      // Fires for BOTH classification paths: a MiniLM result below 60% is
      // just as uncertain as a keyword one — the user must hear the caveat,
      // not read it off a meta line. (Keyword-path confidence is floored at
      // 0.5, so this changes nothing for the fallback.)
      // TWO SIGNALS, TAKE THE STRONGER — a MiniLM score under 60% is not on its
      // own grounds to nag. The keyword evidence is a second opinion, and on a
      // clean read it is frequently the stronger one: with the export width fixed,
      // four clean documents read at >=85% confidence WITH fields present while
      // MiniLM still came back under 60%, and the app told the user the type was
      // uncertain about a scan it had just read cleanly. Nag only when the read is
      // genuinely ambiguous -- both signals weak. (The document-evidence count is
      // what pipeline.understand computes; 0 means no keyword matched at all,
      // which is also the case where it refuses to assert fields.)
      if (u.classificationConfidence < 0.6 && (u.documentEvidence || 0) < 2) {
        warnings.push("The type is uncertain — this may not be a medicine label.");
      }

      const ms = Math.round(performance.now() - t0);
      lastResult = Object.assign(u, {
        ocrConfidence: ocr.confidence,
        speechText: readingMethod === "ppocrv5" ? (ocr.speechText || "") : ocr.text,
        warnings: warnings,
        quality: pp.quality,
        imageDims: pp.width + "×" + pp.height,
        skew: pp.skew,
        engine: engine,
        sourceName: name || "camera",
        pageNote: note || null,
        ms: ms,
        modelProbs: cls ? cls.probs : null,
        timestamp: new Date().toISOString(),
      });

      render(lastResult);
      document.getElementById("results-title")?.focus({ preventScroll: true });
      els.results.scrollIntoView({ block: "start", behavior: "auto" });
      precacheModelsOnce();
      setProgress(100, "Done in " + (ms / 1000).toFixed(1) + "s");
      announce("Scan complete. " + lastResult.categoryLabel + ". " +
        lastResult.fields.length + " key details found. Press Read Aloud to hear the summary.");
    } catch (e) {
      const msg = (e && e.message) || String(e);
      showError("Scan failed: " + msg);
      setProgress(0, "Failed");
      // The panel still shows the PREVIOUS scan. Never let Read Aloud speak
      // a stale document after a failed one — a blind user cannot see that
      // the card did not update.
      lastResult = null;
      els.results.hidden = true;
      document.body.classList.remove("has-results");
      els.btnSpeak.hidden = true;
    } finally {
      els.scanLine.classList.remove("on");
      showProgress(false);
      window.setTimeout(() => setProgress(0, ""), 1200);
      busy = false;
      setScanEnabled(true);
    }
    return lastResult;
  }

  /** Yield to the compositor so the progress bar actually paints. */
  function frame() {
    return new Promise((r) => window.requestAnimationFrame(() => window.setTimeout(r, 0)));
  }

  function setScanEnabled(on) {
    for (const b of [els.btnCapture, els.btnPick, els.ocrModel]) if (b) b.disabled = !on;
  }

  // ── Render ────────────────────────────────────────────────────────────

  function render(r) {
    els.results.hidden = false;
    document.body.classList.add("has-results");
    if (els.cantonese) { els.cantonese.hidden = true; els.cantonese.textContent = ""; }
    const actions = document.getElementById("result-actions");
    if (actions) for (const control of [els.btnSpeak, els.speakLang, els.btnRetake, els.cantonese]) {
      if (control) actions.appendChild(control);
    }

    els.docType.textContent = r.categoryLabel;
    els.summaryText.textContent = r.summary;

    const conf = Math.round(r.classificationConfidence * 100);
    const ocrConf = Math.round(r.ocrConfidence * 100);
    els.resultMeta.textContent = [
      "Type confidence " + conf + "% (" + strengthWord(r.classificationConfidence) + ")",
      "Text confidence " + ocrConf + "%",
      r.classificationSource === "minilm" ? "MiniLM model" : "keyword classifier",
      "reader: " + (r.engine.ocr || "crnn"),
      "restorer: " + r.engine.restorer,
      r.sourceName,
      r.pageNote,
      r.ms + " ms",
    ].filter(Boolean).join(" · ");

    // Fields
    if (!r.fields.length) {
      els.fieldsCard.hidden = true;
    } else {
      els.fieldsCard.hidden = false;
      els.fieldsList.textContent = "";
      const sections = new Map();
      for (const f of r.fields) {
        const label = String(f.label).toLowerCase();
        const title = /warning|maximum|limit/.test(label) ? "Warnings"
          : /dosage|frequency|direction/.test(label) ? "Dose and directions"
          : /medication|medicine/.test(label) ? "Medicine"
          : /patient|birth/.test(label) ? "Personal details"
          : /prescriber|pharmacist|doctor/.test(label) ? "Prescriber"
          : /balance|amount|total|tax|currency/.test(label) ? "Amounts"
          : "Other details";
        if (!sections.has(title)) {
          const section = document.createElement("section");
          const heading = document.createElement("h4");
          heading.textContent = title;
          const list = document.createElement("dl");
          section.append(heading, list); els.fieldsList.appendChild(section);
          sections.set(title, list);
        }
        const row = document.createElement("div");
        row.className = "field-row";
        const l = document.createElement("dt");
        l.className = "label";
        l.textContent = f.label;
        const v = document.createElement("dd");
        v.className = "val";
        v.textContent = f.value;
        row.appendChild(l);
        row.appendChild(v);
        sections.get(title).appendChild(row);
      }
    }

    // Entities
    if (!r.entities.length) {
      els.entitiesCard.hidden = true;
    } else {
      els.entitiesCard.hidden = false;
      els.entitiesList.textContent = "";
      for (const e of r.entities.slice(0, 40)) {
        const s = document.createElement("span");
        s.className = "tag";
        s.textContent = e.label + ": " + e.text;
        els.entitiesList.appendChild(s);
      }
    }

    // Warnings
    if (r.warnings && r.warnings.length) {
      els.warningBox.hidden = false;
      els.warningBox.textContent = r.warnings.join(" ");
    } else {
      els.warningBox.hidden = true;
    }

    els.ocrText.textContent = r.ocrText || "(no text recognised)";
    els.btnSpeak.hidden = false;
    els.btnRetake.hidden = false;
    if (els.speakLang) els.speakLang.hidden = false;
  }

  /**
   * After the FIRST successful scan, ask the service worker to put the ONNX
   * models + runtime into its own cache.
   *
   * This is the difference between the documented offline promise and a broken
   * one. A scan puts the models in the HTTP cache, but an HTTP cache is
   * evictable and is not the offline guarantee; sw.js exposes
   * `precache-optional` for exactly this and NOTHING called it, so an installed
   * app that had scanned once still could not scan from a cold cache. (Verified
   * by tools/ios_full_check.js: "cold install" before the call was added, and
   * "models are in the service-worker cache after one scan" after.)
   *
   * Once per session: the SW writes the same bytes on a repeat, and this is not
   * worth a progress UI.
   */
  let _optionalRequested = false;
  function precacheModelsOnce() {
    if (_optionalRequested) return;
    _optionalRequested = true;
    try {
      const sw = navigator.serviceWorker && navigator.serviceWorker.controller;
      if (sw) sw.postMessage({ type: "precache-optional" });
    } catch (e) { /* no service worker: http:// on a LAN IP, or a first load */ }
  }

  function strengthWord(c) {
    if (c >= 0.75) return "high";
    if (c >= 0.45) return "moderate";
    return "low";
  }

  // ── Speech ────────────────────────────────────────────────────────────

  /** The zh/en medicine-name lexicon (CC-BY-4.0, Taiwan FDA leaflet archive). 1.7 MB, so it is
   *  fetched on first use and cached by the service worker rather than parsed on every load. */
  let lexiconPromise = null;
  function loadLexicon() {
    if (!lexiconPromise) {
      lexiconPromise = fetch("data/drug_lexicon.json", { cache: "force-cache" })
        .then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
    }
    return lexiconPromise;
  }

  async function speakSummary() {
    if (!lastResult) return;
    if (TTS.speaking()) { TTS.stop(); els.btnSpeak.textContent = "Read Aloud"; return; }
    if (!TTS.available()) { showError("This browser has no speech synthesis available."); return; }
    const lang = (els.speakLang && els.speakLang.value) || "en-GB";
    els.btnSpeak.textContent = "Stop";
    announce("Reading summary.");
    try {
      if (lastResult.engine.ocr === "ppocrv5" && !lastResult.speechText) {
        const message = "Not clear enough to read aloud. Try another photo in better light, or ask someone to check the label with you.";
        showError(message);
        await TTS.speak(message, { rate: 0.95, lang: "en-GB" });
        return;
      }
      // ── Cantonese: a BUILT utterance, not the English summary in a Cantonese accent ──
      // The drug name and dose are translated from facts we hold (see js/say_cantonese.js);
      // anything the lexicon cannot translate is refused, never guessed, and never spoken.
      if (lang === "zh-HK") {
        const lexicon = await loadLexicon();
        const said = SayCantonese.build(lastResult.speechText !== undefined ? lastResult.speechText : (lastResult.ocrText || ""), lexicon);
        if (els.cantonese) {
          els.cantonese.hidden = false;
          els.cantonese.textContent = said.zh;
        }
        if (!said.ok) {
          showError("Read too poorly to say it in Cantonese — nothing was guessed. English summary below.");
        }
        if (!TTS.hasVoice("zh-HK")) {
          // Refuse to fake it: an English voice reading Cantonese is worse than silence.
          showError("This device has no Cantonese voice installed, so the app will not pretend. "
            + "The Cantonese text it would say is shown above.");
          els.btnSpeak.textContent = "Read Aloud";
          return;
        }
        await TTS.speak(said.zh, { rate: 0.95, lang: "zh-HK" });
        return;
      }
      // Warnings are spoken BEFORE the summary: the summary states fields as
      // fact, so a low-confidence read must be caveated in the same breath —
      // this is the honesty gate, in the medium this product actually serves.
      const warned = lastResult.warnings && lastResult.warnings.length
        ? "Caution. " + lastResult.warnings.join(" ") + " " : "";
      await TTS.speak(warned + lastResult.summary, { rate: 0.95, lang: "en-GB" });
    } catch (e) {
      showError("Speech failed: " + ((e && e.message) || "unknown"));
    } finally {
      els.btnSpeak.textContent = "Read Aloud";
    }
  }

  // ── Offline badge + service worker ────────────────────────────────────

  let offlineReady = false;
  function updateOnlineBadge() {
    if (offlineReady) setBadge("Ready offline", "ok");
    else setBadge(navigator.onLine ? "Preparing offline reading" : "Offline. Reading files may be missing.", navigator.onLine ? "pending" : "warn");
  }

  async function registerSW() {
    if (!("serviceWorker" in navigator)) {
      setBadge("Offline saving is unavailable", "warn");
      return;
    }
    try {
      const reg = await navigator.serviceWorker.register("sw.js", { scope: "./" });
      if (reg.active && navigator.serviceWorker.controller) {
        setBadge("Checking offline reading", "pending");
      } else {
        setBadge("Saving for offline reading", "pending");
        reg.addEventListener("updatefound", () => {
          const sw = reg.installing;
          if (sw) sw.addEventListener("statechange", () => {
            if (sw.state === "activated") setBadge("Checking offline reading", "pending");
          });
        });
      }
      navigator.serviceWorker.addEventListener("message", (e) => {
        if (e.data && e.data.type === "precache-report") {
          const d = e.data;
          offlineReady = d.failed === 0;
          if (offlineReady) updateOnlineBadge();
          else setBadge("Offline saving did not finish", "warn");
        }
      });
      const ready = await navigator.serviceWorker.ready;
      const worker = navigator.serviceWorker.controller || ready.active;
      if (worker) worker.postMessage({ type: "precache-status" });
    } catch (e) {
      setBadge("Offline saving did not finish", "warn");
    }
  }

  // ── Tabs ──────────────────────────────────────────────────────────────

  function setTab(which) {
    const isCam = which === "cam";
    els.tabCam.classList.toggle("active", isCam);
    els.tabCam.setAttribute("aria-selected", isCam ? "true" : "false");
    els.tabFile.classList.toggle("active", !isCam);
    els.tabFile.setAttribute("aria-selected", !isCam ? "true" : "false");
    els.tabs.setAttribute("role", "tablist");
    els.tabCam.tabIndex = isCam ? 0 : -1;
    els.tabFile.tabIndex = isCam ? -1 : 0;
    els.viewfinder.setAttribute("aria-labelledby", isCam ? "tab-cam" : "tab-file");
  }

  // ── Init ──────────────────────────────────────────────────────────────

  function cacheDom() {
    const ids = [
      "live", "live-assertive", "badge", "error-banner", "viewfinder", "video", "preview",
      "scan-line", "scan-flash", "quality-hint", "file-input", "btn-capture",
      "btn-pick", "btn-retake", "btn-speak", "speak-lang", "cantonese", "progress-wrap", "progress-label",
      "progress-track", "progress-fill", "results", "doc-type", "summary-text",
      "result-meta", "fields-card", "fields-list", "entities-card",
      "entities-list", "warning-box", "ocr-text", "tabs", "tab-cam", "tab-file",
      "skip-restore", "skip-classify", "ocr-model",
    ];
    for (const id of ids) els[id.replace(/-([a-z])/g, (m, c) => c.toUpperCase())] = $(id);
  }

  async function init() {
    cacheDom();
    if (els.ocrModel) {
      try { const saved = localStorage.getItem("sightline.reader"); if (["ppocrv5", "crnn"].includes(saved)) els.ocrModel.value = saved; } catch (e) {}
      els.ocrModel.addEventListener("change", () => {
        // Terminate the enhanced reader's heap when switching back, rather than
        // retaining both readers and optional model sessions at once.
        SightLine.dispose();
        try { localStorage.setItem("sightline.reader", els.ocrModel.value); } catch (e) {}
      });
    }

    // pdf.js worker, same-origin so it is precached and works offline.
    if (window.pdfjsLib && window.pdfjsLib.GlobalWorkerOptions) {
      const base = SightLine.vendorBase();
      window.pdfjsLib.GlobalWorkerOptions.workerSrc = base + "vendor/pdfjs/pdf.worker.min.js";
    }

    els.btnCapture.addEventListener("click", () => {
      if (els.viewfinder.dataset.mode === "none") return startCamera();
      const src = els.viewfinder.dataset.mode === "camera" ? captureFrame() : els.preview;
      if (!src) return;
      if (!src.width && !src.naturalWidth) {
        const c2 = document.createElement("canvas");
        c2.width = els.preview.naturalWidth; c2.height = els.preview.naturalHeight;
        c2.getContext("2d").drawImage(els.preview, 0, 0);
        return runScan(c2, "photo");
      }
      return runScan(src, els.viewfinder.dataset.mode === "camera" ? "camera" : "photo");
    });
    els.btnPick.addEventListener("click", pickFile);
    els.btnRetake.addEventListener("click", () => {
      lastResult = null;
      els.results.hidden = true;
      document.body.classList.remove("has-results");
      els.btnSpeak.hidden = true;
      els.btnRetake.hidden = true;
      startCamera();
    });
    els.btnSpeak.addEventListener("click", speakSummary);
    els.fileInput.addEventListener("change", onFile);
    const openCam = () => { setTab("cam"); startCamera(); };
    const openFile = () => { setTab("file"); stopCamera(); pickFile(); };
    els.tabCam.addEventListener("click", openCam);
    els.tabFile.addEventListener("click", openFile);
    // role="tab" divs have no native activation: Enter and Space must switch
    // tabs for keyboard and switch-access users (WCAG 2.1.1).
    for (const [el, open] of [[els.tabCam, openCam], [els.tabFile, openFile]]) {
      el.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); }
        else if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) {
          e.preventDefault();
          const target = e.key === "Home" ? els.tabCam : e.key === "End" ? els.tabFile : el === els.tabCam ? els.tabFile : els.tabCam;
          target.focus();
        }
      });
    }

    window.addEventListener("online", updateOnlineBadge);
    window.addEventListener("offline", updateOnlineBadge);
    window.addEventListener("beforeunload", () => { stopCamera(); TTS.stop(); });

    // iOS memory: release every ORT session when the page is going away.
    // `beforeunload` is unreliable on iOS; `pagehide` is the signal WebKit
    // actually fires on navigation, on app-switch and on tab close. Releasing
    // returns the recogniser's 4 MB graph and the classifier's 22.9 MB int8
    // encoder to ort's runtime instead of leaving them resident in a wasm heap
    // that Safari will jettison under memory pressure — the failure mode being
    // a dead tab the user has to force-quit, reported by them as "the app
    // stopped working".
    window.addEventListener("pagehide", () => {
      stopCamera();
      TTS.stop();
      try { SightLine.dispose(); } catch (e) { /* nothing was loaded */ }
    });

    // Live quality hint while the camera is up.
    if (els.video) {
      els.video.addEventListener("loadeddata", () => {
        els.qualityHint.hidden = false;
        els.qualityHint.textContent = "Fill the frame with the document";
        window.setTimeout(() => { els.qualityHint.hidden = true; }, 2500);
      });
    }

    updateOnlineBadge();
    registerSW();
    // Camera permission belongs to a real tap. Optional model sessions are
    // initialized by restore()/classify() only when a scan actually uses them;
    // startup no longer compiles an unused second WASM heap alongside PP-OCR.
    els.btnCapture.textContent = "Start camera";
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Exposed for the test harnesses, which drive the real controller rather than
  // a copy of it: selftest.html checks the DOM, warning_gate.js checks that a
  // low-confidence scan is actually flagged. runScan resolves with the result
  // so callers do not have to reach into a private variable.
  window.SightLineApp = {
    runScan: runScan,
    render: render,
    get lastResult() { return lastResult; },
  };
})();
