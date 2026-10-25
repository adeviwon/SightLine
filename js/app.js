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
  let historyRequest = 0,
    sourceImageBlob = null;
  let savedId = null,
    saving = false,
    saveEnabled = false,
    saveImages = false,
    deleteTarget = null;
  let speechRequest = 0,
    speechPreparing = false;
  function stopReading() {
    ++speechRequest;
    speechPreparing = false;
    TTS.stop();
  }
  let screen = "home",
    scanGeneration = 0,
    cameraGeneration = 0;
  let importing = false,
    activeImport = 0,
    lastAttempt = null,
    sourceURL = null,
    retrySource = "camera";
  let offlineStatus = { text: "Checking offline access", state: "pending" };
  let readerNeedsReload = false;

  function updateBadge() {
    const pending = offlineStatus.state === "pending";
    const text =
      readerNeedsReload
        ? "Reader needs reloading"
        : screen === "processing"
        ? "Reading on this device"
        : pending && screen === "results"
          ? "Private on this device"
          : offlineStatus.text;
    els.badge.textContent = text;
    const retry = $("btn-save-offline");
    if (retry)
      retry.hidden =
        offlineStatus.state !== "warn" || !("serviceWorker" in navigator);
    els.badge.dataset.state =
      readerNeedsReload ? "warn" : screen === "processing" || (pending && screen === "results")
        ? "neutral"
        : offlineStatus.state;
    if (els.offlineDetail)
      els.offlineDetail.textContent = offlineStatus.text + ".";
  }

  function setStage(next, focus = true) {
    screen = next;
    document.body.dataset.screen = next;
    for (const [name, id] of Object.entries({
      home: "home-view",
      capture: "capture-view",
      processing: "processing-view",
      results: "results",
      error: "error-view",
      history: "history-view",
    })) {
      $(id).hidden = name !== next;
    }
    $("processing-view").setAttribute(
      "aria-busy",
      String(next === "processing"),
    );
    updateBadge();
    if (focus) {
      const target = $(
        next === "results"
          ? "results-title"
          : next === "history"
            ? "history-title"
            : next === "home"
              ? "home-title"
              : next === "capture"
                ? "capture-title"
                : next === "error"
                  ? "error-title"
                  : "processing-title",
      );
      target?.focus({ preventScroll: true });
      window.scrollTo({ top: 0, behavior: "auto" });
    }
  }

  function goHome() {
    const wasWorking = busy || importing;
    ++scanGeneration;
    ++activeImport;
    importing = false;
    busy = false;
    els.fileInput.value = "";
    if (wasWorking) SightLine.dispose();
    stopCamera();
    stopReading();
    lastResult = null;
    savedId = null;
    sourceImageBlob = null;
    lastAttempt = null;
    if (sourceURL) {
      URL.revokeObjectURL(sourceURL);
      sourceURL = null;
    }
    els.preview.removeAttribute("src");
    els.sourceImage.removeAttribute("src");
    els.preview.hidden = true;
    els.video.hidden = true;
    els.viewfinder.dataset.mode = "none";
    showError("");
    showProgress(false);
    setScanEnabled(true);
    setStage("home");
  }

  function openDialog(id) {
    $(id).showModal();
  }
  function appearanceChanged() {
    const choice = $("appearance").value;
    document.documentElement.dataset.theme =
      choice === "system"
        ? window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light"
        : choice;
    try {
      localStorage.setItem("sightline.appearance", choice);
    } catch (e) {}
  }

  function languageChanged() {
    stopReading();
    const lang = els.speakLang.value;
    SightLineI18n.set(lang);
    $("language-label").textContent =
      lang === "zh-HK" ? "閱讀語言：廣東話" : "Reading language: English";
    try {
      localStorage.setItem("sightline.language", lang);
    } catch (e) {
      announce("Language could not be remembered on this device.");
    }
    TTS.loadVoices().then(updateSpeechPreview);
    updateSpeechPreview();
    if (screen === "results" && lastResult) render(lastResult);
    if (screen === "history") renderHistory();
    $("device-frame").contentWindow?.postMessage(
      { type: "device-language", language: lang },
      location.origin,
    );
  }
  function speechPlan() {
    return els.speakLang.value === "zh-HK"
      ? CantoneseSummary.build(lastResult)
      : SightLineSpeech.plan(lastResult);
  }
  function renderSpokenPreview(result, plan, language) {
    const container = $("spoken-preview");
    container.replaceChildren();
    if (!plan.eligible) {
      container.textContent = plan.text || plan.reason || "";
      return;
    }
    const groups = new Map();
    if (result?.category === "medical") {
      const layout = SightLineMedicineSections.build(result);
      for (const section of layout.sections || [])
        groups.set(section.title, section.fields);
    } else if (result?.category === "receipt") {
      const accepted = (result.fields || []).filter(
        (field) =>
          Number.isFinite(field.confidence) &&
          field.confidence >= 0.75 &&
          String(field.value || "").trim() &&
          String(field.sourceLine || "").trim() &&
          (language !== "zh-HK" ||
            [
              "Shop", "Date", "Time", "Total", "Subtotal", "Before tax",
              "Discount", "Tax", "Payment method", "Card ending",
              "Receipt number", "Transaction number", "Quantity",
            ].includes(field.label)),
      );
      const titleFor = (field) =>
        /shop|merchant/i.test(field.label)
          ? "Shop"
          : /date|time/i.test(field.label)
            ? "Purchase"
            : /total|subtotal|tax|discount|amount|currency/i.test(field.label)
              ? "Totals"
              : /payment|card/i.test(field.label)
                ? "Payment"
                : "Other details";
      for (const field of accepted) {
        const title = titleFor(field);
        if (!groups.has(title)) groups.set(title, []);
        groups.get(title).push(field);
      }
    }
    if (!groups.size) {
      container.textContent = plan.text || plan.reason || "";
      return;
    }
    for (const [title, fields] of groups) {
      if (!fields.length) continue;
      const card = document.createElement("section");
      card.className = "spoken-summary-section";
      const heading = document.createElement("h3");
      heading.textContent = SightLineI18n.t(title);
      heading.id = "spoken-group-" + container.childElementCount;
      card.setAttribute("aria-labelledby", heading.id);
      card.appendChild(heading);
      const list = document.createElement("table");
      list.className = "spoken-summary-table";
      list.setAttribute("aria-labelledby", heading.id);
      const body = document.createElement("tbody");
      list.appendChild(body);
      for (const field of fields) {
        const row = document.createElement("tr");
        const label = document.createElement("th");
        label.scope = "row";
        const value = document.createElement("td");
        if (result.category === "medical" && fields.length === 1 && title !== "Medicine")
          label.classList.add("sr-only");
        const localized = SightLineFieldLocalization.localize(
          result,
          field,
          language,
        );
        label.textContent = SightLineI18n.t(localized.label);
        value.textContent = localized.value;
        value.lang = localized.lang;
        row.className = "spoken-summary-row";
        row.append(label, value);
        body.appendChild(row);
      }
      card.appendChild(list);
      container.appendChild(card);
    }
    const note = document.createElement("p");
    note.className = "spoken-summary-note";
    const key =
      result.category === "medical" && language === "en-GB"
        ? "These cards show the key fields. The full verified label text will also be read aloud."
        : "These cards organize key details. The reading follows its supported summary; check the original for anything missing.";
    note.textContent = SightLineI18n.t(key);
    container.appendChild(note);
  }
  function updateSpeechPreview() {
    if (!lastResult) return;
    const plan = speechPlan();
    renderSpokenPreview(lastResult, plan, els.speakLang.value);
    $("spoken-preview").lang = els.speakLang.value;
    const cantonese = els.speakLang.value === "zh-HK";
    const voiceReady = TTS.hasVoice(els.speakLang.value);
    $("btn-use-english").hidden = !cantonese;
    $("speech-note").textContent = !TTS.available()
      ? SightLineI18n.t(
          "Read-aloud isn’t available in this browser. You can still read the text here.",
        )
      : !voiceReady && cantonese
        ? SightLineI18n.t(
            "No Cantonese voice is available. Install an offline Cantonese voice in your device speech settings, or choose English.",
          )
        : !voiceReady
          ? SightLineI18n.t(
              "No offline English voice is available. Install an English voice in your device speech settings, then try again.",
            )
          : !plan.eligible
            ? plan.text || plan.reason
            : cantonese
              ? "可以朗讀已核對嘅摘要，原文保持不變。"
              : "Listen at your own pace. You can stop the reading at any time.";
    // Unverified documents offer spoken recovery help, never unverified text.
    els.btnSpeak.hidden = false;
    els.btnSpeak.disabled = !TTS.available();
    if (!TTS.speaking() && !speechPreparing) speechLabel("Read Aloud");
    $("speech-note").hidden = TTS.available() && plan.eligible && voiceReady;
    SightLineI18n.apply();
  }
  function storePreferences() {
    try {
      localStorage.setItem(
        "sightline.history-consent",
        JSON.stringify({ enabled: saveEnabled, images: saveImages }),
      );
    } catch {
      announce("Saving preferences could not be remembered.");
    }
  }
  async function saveCurrentScan(ask = true) {
    if (!lastResult || saving) return;
    if (!saveEnabled) {
      if (ask) $("save-consent").showModal();
      return;
    }
    const target = lastResult,
      oldId = savedId;
    saving = true;
    $("btn-save-scan").disabled = true;
    $("save-status").textContent = SightLineI18n.t("Saving…");
    try {
      let image = null;
      if (saveImages && els.preview.getAttribute("src")) {
        if (sourceImageBlob) image = sourceImageBlob;
        else {
          const response = await fetch(els.preview.src);
          if (!response.ok)
            throw new Error(
              "The original image could not be saved. Try saving without images.",
            );
          image = await response.blob();
          sourceImageBlob = image;
        }
      }
      const id = await SightLineHistory.save(target, image, oldId);
      if (target === lastResult) {
        savedId = id;
        $("save-status").textContent = SightLineI18n.t("Saved on this device");
        $("btn-save-scan").textContent = SightLineI18n.t(
          "Saved on this device",
        );
      }
    } catch (e) {
      if (target === lastResult) {
        $("save-status").textContent = SightLineI18n.t(
          SightLineHistory.message(e),
        );
        $("btn-save-scan").textContent = SightLineI18n.t("Save scan");
      }
    } finally {
      saving = false;
      $("btn-save-scan").disabled = false;
    }
  }
  async function renderHistory() {
    const request = ++historyRequest;
    const container = $("history-list");
    container.replaceChildren();
    $("history-empty").hidden = true;
    $("history-status").textContent = SightLineI18n.t("Opening saved scans…");
    try {
      const records = await SightLineHistory.list();
      if (request !== historyRequest || screen !== "history") return;
      $("history-status").textContent = "";
      $("history-empty").hidden = records.length > 0;
      $("history-clear").disabled = !records.length;
      for (const record of records) {
        const r = record.result;
        if (!r || !Array.isArray(r.fields)) continue;
        const type =
          r.category === "receipt"
            ? "Receipt"
            : r.category === "medical"
              ? "Medicine label or prescription"
              : "Document";
        const knownNames = [
          ...new Set(
            (r.entities || [])
              .filter(
                (e) =>
                  e.label === "DRUG" &&
                  e.confidence >= 0.75 &&
                  r.speechText &&
                  r.speechText.toLowerCase().includes(e.text.toLowerCase()),
              )
              .map((e) => e.text),
          ),
        ];
        const title =
          r.fields.find((f) => f.label === "Shop" && f.confidence >= 0.75)
            ?.value ||
          (r.category === "medical" && knownNames.length === 1
            ? knownNames[0]
            : type);
        const total = r.fields.find(
          (f) => f.label === "Total" && f.confidence >= 0.75,
        )?.value;
        const article = document.createElement("article");
        article.className = "history-record";
        const heading = document.createElement("h2");
        heading.className = "history-title original-value";
        heading.textContent = title === type ? SightLineI18n.t(type) : title;
        const meta = document.createElement("p");
        meta.textContent =
          SightLineI18n.t(type) +
          " · " +
          new Intl.DateTimeFormat(els.speakLang.value, {
            dateStyle: "medium",
            timeStyle: "short",
          }).format(new Date(r.timestamp || record.savedAt));
        meta.className = "original-value";
        const amount = document.createElement("p");
        amount.className = "original-value";
        amount.textContent = total
          ? SightLineI18n.t("Total") + ": " + total
          : "";
        const actions = document.createElement("div");
        actions.className = "history-actions";
        const open = document.createElement("button");
        open.className = "button secondary";
        open.textContent = SightLineI18n.t("Open scan");
        open.setAttribute(
          "aria-label",
          SightLineI18n.t("Open scan") + ": " + heading.textContent,
        );
        open.addEventListener("click", () => openSavedScan(record.id));
        const del = document.createElement("button");
        del.className = "text-button";
        del.textContent = SightLineI18n.t("Delete");
        del.addEventListener("click", () => confirmDelete(record.id));
        actions.append(open, del);
        article.append(heading, meta, amount, actions);
        container.append(article);
      }
    } catch (e) {
      $("history-empty").hidden = true;
      $("history-status").textContent = SightLineI18n.t(
        SightLineHistory.message(e),
      );
    }
  }
  async function openSavedScan(id) {
    const opening = ++scanGeneration;
    stopReading();
    try {
      const record = await SightLineHistory.get(id);
      if (opening !== scanGeneration) return;
      if (!record) throw new Error("This saved scan is no longer available.");
      let imageBlob = null;
      if (record.image) {
        try {
          imageBlob = new Blob([await record.image.arrayBuffer()], {
            type: record.image.type,
          });
        } catch {}
      }
      if (opening !== scanGeneration) return;
      if (sourceURL) URL.revokeObjectURL(sourceURL);
      sourceImageBlob = imageBlob;
      sourceURL = imageBlob ? URL.createObjectURL(imageBlob) : null;
      if (sourceURL) els.preview.src = sourceURL;
      else els.preview.removeAttribute("src");
      lastResult = record.result;
      savedId = record.id;
      render(lastResult);
      $("save-status").textContent = SightLineI18n.t("Saved on this device");
      $("history-image-note").hidden = !!sourceImageBlob;
    } catch (e) {
      $("history-status").textContent = SightLineI18n.t(
        SightLineHistory.message(e),
      );
    }
  }
  function confirmDelete(id) {
    deleteTarget = id;
    $("delete-title").textContent = id
      ? "Delete saved scan?"
      : "Delete all saved scans?";
    $("delete-description").textContent = id
      ? "This saved copy will be deleted from this device."
      : "All saved scans will be deleted. Automatic saving will turn off.";
    SightLineI18n.apply();
    $("history-delete-dialog").showModal();
  }

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
    window.setTimeout(() => {
      el.textContent = msg;
    }, 30);
  }

  function setProgress(pct, label, measured) {
    const known = measured && measured.total > 0;
    const p = known
      ? Math.round((measured.done / measured.total) * 100)
      : pct === 100
        ? 100
        : null;
    if (els.progressFill)
      els.progressFill.style.width = p === null ? "100%" : p + "%";
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
    offlineStatus = { text, state: state || "ok" };
    updateBadge();
  }

  function showError(msg) {
    if (!els.errorBanner) return;
    if (!msg) {
      els.errorBanner.hidden = true;
      els.errorBanner.textContent = "";
      return;
    }
    els.errorBanner.hidden = false;
    els.errorBanner.textContent = msg;
    announce("Error: " + msg, true);
  }

  // ── Camera ────────────────────────────────────────────────────────────

  async function startCamera() {
    if (busy) return false;
    showError("");
    stopCamera();
    retrySource = "camera";
    $("capture-title").textContent = "Bring the label into view.";
    $("capture-help").textContent =
      "Keep your hands steady and use good light.";
    setStage("capture");
    const generation = cameraGeneration;
    els.video.hidden = true;
    els.preview.hidden = true;
    $("camera-placeholder").hidden = false;
    els.btnCapture.hidden = false;
    els.btnCapture.disabled = true;
    $("btn-read-photo").hidden = true;
    $("btn-camera-retake").hidden = true;
    if (!navigator.mediaDevices?.getUserMedia) {
      showError(
        "The camera is unavailable here. You can upload a photo or PDF instead.",
      );
      setStage("error");
      return false;
    }
    try {
      const opened = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 3840 },
          height: { ideal: 2160 },
        },
        audio: false,
      });
      if (generation !== cameraGeneration || screen !== "capture") {
        opened.getTracks().forEach((track) => track.stop());
        return false;
      }
      stream = opened;
      els.video.srcObject = stream;
      els.video.hidden = false;
      els.viewfinder.dataset.mode = "camera";
      $("camera-placeholder").hidden = true;
      els.btnCapture.disabled = false;
      announce(
        "Camera ready. Keep all the words in view, then tap Take photo.",
      );
      return true;
    } catch (error) {
      if (generation !== cameraGeneration) return false;
      showError(
        error?.name === "NotAllowedError"
          ? "Camera access wasn’t allowed. Allow camera access in your browser, or upload a photo or PDF."
          : "We couldn’t open the camera. Try again, or upload a photo or PDF.",
      );
      setStage("error");
      return false;
    }
  }

  function stopCamera() {
    ++cameraGeneration;
    if (stream) {
      for (const t of stream.getTracks()) {
        try {
          t.stop();
        } catch (e) {
          /* already stopped */
        }
      }
      stream = null;
    }
    if (els.video) els.video.srcObject = null;
  }

  function captureFrame() {
    sourceImageBlob = null;
    if (!stream || !els.video.videoWidth) {
      showError(
        "No camera frame available. Start the camera or choose a photo.",
      );
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
    const file = e.target.files?.[0];
    if (!file || busy || importing) return;
    sourceImageBlob = file.type.startsWith("image/") ? file : null;
    importing = true;
    const importId = ++activeImport;
    retrySource = "file";
    const generation = ++scanGeneration;
    showError("");
    lastResult = null;
    stopCamera();
    stopReading();
    setScanEnabled(false);
    setStage("processing");
    showProgress(true);
    setProgress(null, "Opening your document.");
    try {
      if (file.type === "application/pdf" || /\.pdf$/i.test(file.name)) {
        await scanPDF(file, generation);
      } else {
        if (sourceURL) URL.revokeObjectURL(sourceURL);
        sourceURL = URL.createObjectURL(file);
        els.preview.src = sourceURL;
        await els.preview.decode();
        if (generation !== scanGeneration) return;
        els.video.hidden = true;
        els.preview.hidden = false;
        els.viewfinder.dataset.mode = "preview";
        await runScan(els.preview, file.name);
      }
    } catch (error) {
      if (generation !== scanGeneration) return;
      showError(
        "We couldn’t open that file. Choose a photo or PDF that opens correctly on your device.",
      );
      setStage("error");
      showProgress(false);
      setScanEnabled(true);
    } finally {
      // Retain the object URL while the original-image comparison is available.
      // It is revoked when the user leaves the result or chooses another file.
      if (importId === activeImport) {
        importing = false;
        els.fileInput.value = "";
      }
    }
  }

  /**
   * scanPDF(): render page 1 at 3x scale to a canvas, then scan.
   * pdf.js is vendored; the worker is same-origin so it stays offline.
   */
  async function scanPDF(file, generation) {
    const pdfjs = window.pdfjsLib;
    if (!pdfjs) {
      showError("PDF reading is unavailable. Try choosing a photo instead.");
      setStage("error");
      setScanEnabled(true);
      return;
    }
    setProgress(null, "Opening your PDF.");
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
      await page.render({ canvasContext: c.getContext("2d"), viewport: vp })
        .promise;
      els.preview.src = c.toDataURL("image/jpeg", 0.92);
      els.preview.hidden = false;
      const placeholder = document.getElementById("camera-placeholder");
      if (placeholder) placeholder.hidden = true;
      els.video.hidden = true;
      els.viewfinder.dataset.mode = "preview";
      const total = doc.numPages;
      doc.destroy();
      if (generation !== scanGeneration) return;
      await runScan(c, file.name, total > 1 ? "Page 1 of " + total : null);
    } catch (e) {
      if (generation !== scanGeneration) return;
      showError("We couldn’t open that PDF. Try a photo of the page instead.");
      setStage("error");
      setScanEnabled(true);
      setProgress(0, "");
      showProgress(false);
    }
  }

  // ── Scan orchestration ────────────────────────────────────────────────

  async function runScan(source, name, note) {
    if (busy) return null;
    busy = true;
    const generation = ++scanGeneration;
    lastAttempt = { source, name, note };
    retrySource = "scan";
    stopReading();
    setStage("processing");
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
      const restoreMode =
        readingMethod === "ppocrv5"
          ? "none"
          : onnxRestorer
            ? "adaptive"
            : "classical";

      // Stage 1: preprocessing is unconditional. The restorer and the
      // classifier are accelerators layered on top, never prerequisites.
      const pp = SightLine.preprocess(source, 3200, { restoreMode });
      setProgress(15, "Making the picture clearer.");
      await frame();
      if (generation !== scanGeneration) return null;

      // Stage 2: optional neural restoration. Falls back silently.
      let ocrCandidates = pp.candidates;
      let engine = { restorer: restoreMode, classifier: "skipped" };
      if (onnxRestorer) {
        setProgress(20, "Restoring image…");
        const r = await SightLineRestorer.armRestorerClahe(pp.canvas, {
          onProgress: (m) => {
            if (m && m.total)
              setProgress(
                20 + (m.done / m.total) * 10,
                "Restoring " + m.done + "/" + m.total,
              );
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
        if (r.candidates)
          ocrCandidates = [pp.candidates[0]].concat(r.candidates);
      }
      await frame();
      if (generation !== scanGeneration) return null;

      // Stage 3: OCR.
      setProgress(32, "Getting ready to read.");
      const ocr = await SightLine.runOCR(
        ocrCandidates,
        (m) => {
          setProgress(
            null,
            readingMethod === "ppocrv5"
              ? m.status || "Reading the words."
              : "Reading the words.",
            m,
          );
        },
        { engine: readingMethod },
      );
      if (generation !== scanGeneration) return null;
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
      if (generation !== scanGeneration) return null;

      if (generation !== scanGeneration) return null;

      // Stage 5: understanding — identical code path in both cases.
      let u = SightLine.understand(
        ocr.text,
        cls
          ? {
              category: cls.category,
              confidence: cls.confidence,
              source: "minilm",
            }
          : undefined,
      );

      // Enhanced reading keeps complete lines only when every word clears
      // 0.75. Use that same conservative read for fields AND Cantonese speech;
      // never stitch a dose out of the remaining words of an uncertain line.
      if (readingMethod === "ppocrv5") {
        const trusted = SightLine.understand(ocr.speechText || "", {
          category: u.category,
          confidence: u.classificationConfidence,
          source: u.classificationSource,
        });
        u.fields = SightLine.retainVerbatim(
          trusted.fields,
          ocr.speechText || "",
        );
        if (u.category === "medical")
          u.fields = SightLine.medicalFieldContext(
            u.fields,
            ocr.speechText || "",
          );
        u.entities = SightLine.retainVerbatim(
          trusted.entities,
          ocr.speechText || "",
          "text",
        );
        u.summary = SightLine.buildSummary(u.category, u.fields);
      }

      // Aria adds the receipt category absent from MiniLM's fixed four classes.
      // Keep the medical whole-page speech policy; receipt cards require clear
      // source lines and independently gated amounts. Never use MiniLM indexes
      // as receipt labels or turn page confidence into classification confidence.
      const { receiptResult } = await import("./receipt.mjs");
      const receipt = receiptResult(ocr);
      if (receipt.result) u = receipt.result;
      u.documentDiagnostics = receipt.diagnostics;
      u.ocrWords = ocr.words || [];
      u.ocrLatencyMs = ocr.ms;

      // Honesty gate on the RECOGNISED text, which is where the safety risk
      // lives. preprocess() only knows about image statistics; a capture can
      // look fine (lapVar 133) and still OCR at 26%, and on a prescription that
      // is the difference between "take 500mg" and "take 300mg".
      //
      // The pipeline deliberately never guesses a missing field, so the risk is
      // not a fabricated dosage — it is a user believing an incomplete one.
      // Say so, rather than letting a confident-looking card imply certainty.
      const warnings = pp.warnings.slice();
      if (
        u.category === "receipt" &&
        ((u.receiptDiagnostics?.rejected || []).length ||
          (u.receiptDiagnostics?.missing || []).length)
      )
        warnings.push("Some details are unclear. They won’t be read aloud.");
      if (ocr.confidence < 0.4) {
        warnings.push(
          "This picture was hard to read. " +
            "Some details may be missing or wrong — check the original document, " +
            "and retake the photo in better light if a figure matters.",
        );
      } else if (ocr.confidence < 0.6) {
        warnings.push(
          "Some words are unclear. Check important figures against the original document.",
        );
      }
      if (u.fields.length === 0) {
        const clearRead =
          ocr.confidence >= 0.75 &&
          (readingMethod !== "ppocrv5" || ocr.speechText);
        warnings.push(
          clearRead
            ? "Some text could be read, but the document’s details may be incomplete. Check the original text below."
            : "No key details could be extracted. Try a closer, straighter photo with the whole page in frame.",
        );
      }
      if (
        readingMethod === "ppocrv5" &&
        !ocr.speechText &&
        u.category !== "receipt"
      ) {
        warnings.push(
          "Not clear enough to read aloud. Try another photo in better light.",
        );
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
        warnings.push(
          "The type is uncertain — this may not be a medicine label.",
        );
      }

      const ms = Math.round(performance.now() - t0);
      savedId = null;
      lastResult = Object.assign(u, {
        ocrConfidence: ocr.confidence,
        speechText:
          readingMethod === "ppocrv5" ? ocr.speechText || "" : ocr.text,
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

      lastResult.speechEligibility = SightLineSpeech.plan(lastResult);
      render(lastResult);
      if (saveEnabled) await saveCurrentScan(false);
      document.getElementById("results-title")?.focus({ preventScroll: true });
      els.results.scrollIntoView({ block: "start", behavior: "auto" });
      precacheModelsOnce();
      setProgress(100, "Done in " + (ms / 1000).toFixed(1) + "s");
      announce(
        "Scan complete. " +
          els.docType.textContent +
          ". " +
          els.fieldsList.querySelectorAll(".field-row").length +
          " key details found. " +
          (!SightLineSpeech.plan(lastResult).eligible
            ? "Some words need checking before they can be read aloud."
            : "Press Read Aloud to hear the text."),
      );
    } catch (e) {
      const msg = (e && e.message) || String(e);
      if (generation !== scanGeneration) return null;
      const loadingFailure = e?.code === "READER_LOAD_FAILED" || readerNeedsReload;
      readerNeedsReload = loadingFailure;
      $("error-recovery").textContent = loadingFailure
        ? "Reload the reader, then choose your photo again. Saved scans will stay on this device."
        : "A clearer photo or another file may help.";
      $("btn-retry").textContent = loadingFailure ? "Reload reader" : "Try again";
      $("reader-error-details").hidden = !loadingFailure;
      $("reader-error-detail").textContent = loadingFailure
        ? "Reader module: " + String(e.cause?.message || msg) : "";
      showError(loadingFailure ? "The reading software could not load." : "We couldn’t finish reading this document. " + msg);
      setStage("error");
      setProgress(0, "Failed");
      // The panel still shows the PREVIOUS scan. Never let Read Aloud speak
      // a stale document after a failed one — a blind user cannot see that
      // the card did not update.
      lastResult = null;
      els.results.hidden = true;
      document.body.classList.remove("has-results");
      els.btnSpeak.hidden = true;
    } finally {
      if (generation === scanGeneration) {
        els.scanLine.classList.remove("on");
        showProgress(false);
        busy = false;
        setScanEnabled(true);
      }
    }
    return lastResult;
  }

  /** Yield to the compositor so the progress bar actually paints. */
  function frame() {
    return new Promise((r) =>
      window.requestAnimationFrame(() => window.setTimeout(r, 0)),
    );
  }

  function setScanEnabled(on) {
    for (const b of [
      els.btnCapture,
      els.btnPick,
      els.btnCamera,
      els.ocrModel,
      els.skipClassify,
      els.skipRestore,
    ])
      if (b) b.disabled = !on;
  }

  // ── Render ────────────────────────────────────────────────────────────

  function render(r) {
    setStage("results");
    if (els.cantonese) {
      els.cantonese.hidden = true;
      els.cantonese.textContent = "";
    }
    els.docType.textContent =
      (r.documentEvidence || 0) >= 2 || r.classificationConfidence >= 0.75
        ? r.category === "medical"
          ? "Medicine label or prescription"
          : r.categoryLabel
        : "Document";
    if (r.pageNote) els.docType.textContent += " · " + r.pageNote;
    els.summaryText.textContent = r.summary;
    document.querySelector(".result-reminder").textContent =
      r.category === "receipt"
        ? "Check important amounts against the original receipt."
        : "Check the original. Ask a pharmacist if unsure.";
    els.resultMeta.textContent = [
      "Text confidence " + Math.round(r.ocrConfidence * 100) + "%",
      "Reading method: " + (r.engine?.ocr || "crnn"),
      r.sourceName,
      r.pageNote,
      r.ms + " ms",
    ]
      .filter(Boolean)
      .join(" · ");
    let fields = (r.fields || []).filter((f) => String(f.value || "").trim());
    els.fieldsList.textContent = "";
    els.fieldsCard.hidden = fields.length === 0;
    $("empty-fields").hidden = fields.length !== 0;
    const clearUnstructured = !!r.speechText && r.ocrConfidence >= 0.75;
    $("empty-fields").querySelector("h2").textContent = clearUnstructured
      ? "Here are the words we found."
      : "Some words need a closer look.";
    $("empty-fields").querySelector("p").textContent = clearUnstructured
      ? "We couldn’t organise this document into individual details. You can read the text below and compare it with the original."
      : "We haven’t filled in details we can’t read reliably. You can check the text below or try a clearer photo.";
    const sections = new Map();
    const medical = r.category === "medical";
    const medicineLayout = medical ? SightLineMedicineSections.build(r) : null;
    // `String(x ?? "")`, not `x.toLowerCase()`. A field or entity that arrives without its
    // text/value used to throw a TypeError HERE -- "Cannot read properties of undefined
    // (reading 'toLowerCase')" -- and the whole scan died with "we couldn't finish reading
    // this document", on every prescription, which is the document this product exists for.
    // A malformed row must cost its own row, never the user's read.
    const medicineNames = new Set(
      (r.entities || [])
        .filter((e) => e.label === "DRUG")
        .map((e) => String(e.text == null ? "" : e.text).toLowerCase())
        .filter(Boolean),
    );
    const doseValues = new Set(
      fields
        .filter((f) => f.label === "Dosage")
        .map((f) => String(f.value == null ? "" : f.value).toLowerCase().replace(/\s+/g, ""))
        .filter(Boolean),
    );
    const unassociated =
      medical && (medicineNames.size > 1 || medicineLayout?.ambiguous || (doseValues.size > 1 && !medicineLayout?.directionsResolved));
    $("association-notice").hidden = !unassociated;
    if (unassociated)
      fields = fields.filter(
        (field) => !/dosage|frequency|direction/i.test(field.label),
      );
    // Promote only an existing, literal, speech-eligible drug entity. No new
    // medicine recognition or dose-to-medicine association happens in the UI.
    if (medical && r.speechText) {
      const literal = (value) =>
        String(value).toLowerCase().replace(/\s+/g, "");
      const source = literal(r.speechText);
      for (const entity of r.entities || []) {
        if (
          entity.label === "DRUG" &&
          source.includes(literal(entity.text)) &&
          !fields.some(
            (field) =>
              /medicine|medication/i.test(field.label) &&
              literal(field.value) === literal(entity.text),
          )
        ) {
          fields.unshift({ label: "Medication", value: entity.text });
        }
      }
    }
    els.fieldsCard.hidden = fields.length === 0;
    $("empty-fields").hidden = fields.length !== 0;
    const order = [
      "Medicine",
      "Strength on the label",
      "Amount shown on the label",
      "How much at once",
      "When to take it",
      "How long",
      "Strength and directions",
      "Warnings",
      "Personal details",
      "Prescriber",
      "Purchase",
      "Totals",
      "Payment",
      "Reference",
      "Other details",
    ];
    const titleFor = (label) =>
      /warning|maximum|limit/.test(label)
        ? "Warnings"
        : /dosage|frequency|direction/.test(label)
          ? "Strength and directions"
          : /medication|medicine/.test(label)
            ? "Medicine"
            : /patient|birth/.test(label)
              ? "Personal details"
              : /prescriber|pharmacist|doctor/.test(label)
                ? "Prescriber"
                : /balance|amount|total|tax|currency/.test(label)
                  ? "Totals"
                  : /payment|card|account/.test(label)
                    ? "Payment"
                    : /shop|merchant|purchase|item/.test(label)
                      ? "Purchase"
                      : /date|reference|case/.test(label)
                        ? "Reference"
                        : "Other details";
    for (const field of fields) {
      const title = titleFor(String(field.label).toLowerCase());
      if (!sections.has(title)) sections.set(title, []);
      sections.get(title).push(field);
    }
    if (medicineLayout?.sections.length) {
      sections.clear();
      for (const group of medicineLayout.sections) sections.set(group.title, group.fields);
      fields = medicineLayout.sections.flatMap(group => group.fields);
      els.fieldsCard.hidden = false;
      $("empty-fields").hidden = true;
    }
    for (const title of order) {
      if (!sections.has(title)) continue;
      const section = document.createElement("section");
      section.className = "field-group";
      const heading = document.createElement("h3");
      heading.textContent = title;
      heading.id = "field-group-" + els.fieldsList.childElementCount;
      section.setAttribute("aria-labelledby", heading.id);
      section.appendChild(heading);
      if (medical && title === "Strength and directions") {
        section.dataset.hasDirections = "true";
      }
      const list = document.createElement("table");
      list.className = "field-table";
      list.setAttribute("aria-labelledby", heading.id);
      const body = document.createElement("tbody");
      list.appendChild(body);
      section.appendChild(list);
      for (const field of sections.get(title)) {
        const row = document.createElement("tr");
        row.className = "field-row";
        const label = document.createElement("th");
        label.scope = "row";
        label.className = "label";
        if (medicineLayout?.sections.length && sections.get(title).length === 1 && title !== "Medicine")
          label.classList.add("sr-only");
        const englishLabel =
          field.label === "Dosage"
            ? "Strength or dose text"
            : field.label === "Medication"
              ? "Medicine name"
              : field.label;
        const localized = SightLineFieldLocalization.localize(
          r,
          { ...field, label: englishLabel },
          els.speakLang.value,
        );
        label.textContent = localized.label;
        const value = document.createElement("td");
        value.className = "val";
        value.lang = localized.lang;
        const primaryValue = document.createElement("span");
        primaryValue.textContent = localized.value;
        value.appendChild(primaryValue);
        if (localized.translated) {
          const original = document.createElement("span");
          original.className = "field-original";
          original.lang = "zh-HK";
          original.append(document.createTextNode("英文原文："));
          const sourceValue = document.createElement("span");
          sourceValue.lang = "en";
          sourceValue.textContent = String(field.value ?? "");
          original.appendChild(sourceValue);
          value.appendChild(original);
        } else if (localized.needsVerification && els.speakLang.value === "zh-HK") {
          const note = document.createElement("span");
          note.className = "field-original";
          note.textContent = "英文原文，暫未有核實譯文";
          note.lang = "zh-HK";
          value.appendChild(note);
        }
        row.append(label, value);
        body.appendChild(row);
      }
      if (section.dataset.hasDirections) {
        const note = document.createElement("p");
        note.className = "group-note";
        note.textContent =
          "Directions are shown as found. Check which medicine they belong to.";
        section.appendChild(note);
      }
      els.fieldsList.appendChild(section);
    }
    els.entitiesList.textContent = "";
    els.entitiesCard.hidden = !(r.entities || []).length;
    for (const entity of (r.entities || []).slice(0, 40)) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = entity.label + ": " + entity.text;
      els.entitiesList.appendChild(tag);
    }
    els.warningBox.hidden = !r.warnings?.length;
    els.warningBox.textContent = (r.warnings || []).join(" ");
    els.ocrText.textContent =
      r.ocrText ||
      "No readable words were found. Try a closer photo in better light.";
    const missingMedicine = $("medicine-missing");
    missingMedicine.hidden = !medicineLayout?.missing?.length;
    missingMedicine.textContent = medicineLayout?.missing?.length
      ? SightLineI18n.t("Not clearly found:") + " " + medicineLayout.missing.map(value => SightLineI18n.t(value)).join(", ") + ". " + SightLineI18n.t("Check the original label for missing details.")
      : "";
    $("transcript-details").open = fields.length === 0;
    els.sourceImage.src = els.preview.getAttribute("src") || "";
    els.sourceDetails.hidden = !els.preview.getAttribute("src");
    els.sourceDetails.open = window.matchMedia("(min-width: 900px)").matches;
    $("playback-status").hidden = true;
    const plan = SightLineSpeech.plan(r);
    const speechAvailable = TTS.available();
    $("speech-note").textContent = !plan.eligible
      ? plan.reason
      : !speechAvailable
        ? "Read-aloud isn’t available in this browser. You can still read the text here."
        : r.category === "receipt"
          ? "Listen to the clear receipt details. Unclear amounts are left out."
          : "Listen at your own pace. You can stop the reading at any time.";
    $("save-status").textContent = savedId
      ? SightLineI18n.t("Saved on this device")
      : "";
    $("btn-save-scan").textContent = savedId
      ? "Saved on this device"
      : "Save scan";
    $("history-image-note").hidden = true;
    updateSpeechPreview();
    els.btnRetake.hidden = false;
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
    } catch (e) {
      /* no service worker: http:// on a LAN IP, or a first load */
    }
  }

  function strengthWord(c) {
    if (c >= 0.75) return "high";
    if (c >= 0.45) return "moderate";
    return "low";
  }

  // ── Speech ────────────────────────────────────────────────────────────

  /** The zh/en medicine-name lexicon (CC-BY-4.0, Taiwan FDA leaflet archive). 1.7 MB, so it is
   *  fetched on first use and cached by the service worker rather than parsed on every load. */
  let lexiconPromise = null,
    readyLexicon = null;
  function loadLexicon() {
    if (!lexiconPromise) {
      lexiconPromise = fetch("data/drug_lexicon.json", { cache: "force-cache" })
        .then((r) => {
          if (!r.ok) throw new Error("Reading language files are missing");
          return r.json();
        })
        .then((value) => {
          readyLexicon = value;
          return value;
        })
        .catch(() => {
          lexiconPromise = null;
          return null;
        });
    }
    return lexiconPromise;
  }

  function speechLabel(text) {
    if (text === "Read Aloud" && lastResult && !speechPlan().eligible)
      text = "Hear reading help";
    const label = els.btnSpeak.querySelector("span");
    if (label) label.textContent = SightLineI18n.t(text);
    else els.btnSpeak.textContent = SightLineI18n.t(text);
  }

  async function ensureSpeechVoice(language, request) {
    if (TTS.hasVoice(language)) return true;
    if (TTS.nativeBackend()) {
      speechPreparing = true;
      speechLabel("Stop reading");
      $("playback-status").hidden = false;
      $("playback-status").textContent = SightLineI18n.t(
        "Finding an offline voice…",
      );
      await TTS.loadVoices();
      if (request !== speechRequest) return false;
      speechPreparing = false;
      updateSpeechPreview();
      if (TTS.hasVoice(language)) return true;
      throw Object.assign(
        new Error(TTS.errorMessage({ code: "no-local-voice" })),
        { code: "no-local-voice" },
      );
    }
    // Web Speech may require the original tap to start. Refresh in the
    // background, explain the retry, and let the next tap start synchronously.
    TTS.loadVoices().then(updateSpeechPreview);
    throw Object.assign(
      new Error(TTS.errorMessage({ code: "voices-not-ready" })),
      { code: "voices-not-ready" },
    );
  }

  async function speakSummary() {
    if (!lastResult) return;
    if (TTS.speaking() || speechPreparing) {
      stopReading();
      speechLabel("Read Aloud");
      return;
    }
    if (!TTS.available()) {
      showError("This browser has no speech synthesis available.");
      return;
    }
    const request = ++speechRequest;
    showError("");
    const lang = (els.speakLang && els.speakLang.value) || "en-GB";
    speechLabel("Stop");

    try {
      if (!speechPlan().eligible) {
        const message =
          lang === "zh-HK"
            ? CantoneseSummary.build(lastResult).text
            : "Not clear enough to read aloud. Try another photo in better light, or ask someone to check the label with you.";
        showError(message);
        await ensureSpeechVoice(lang, request);
        if (request !== speechRequest) return;
        await TTS.speak(message, { rate: 0.95, lang });
        return;
      }
      if (lang === "zh-HK") {
        const plan = CantoneseSummary.build(lastResult);
        if (!plan.eligible) {
          showError(plan.text);
          return;
        }
        await ensureSpeechVoice("zh-HK", request);
        if (request !== speechRequest) return;
        await TTS.speak(plan.text, { rate: 0.95, lang: "zh-HK" });
        return;
      }
      if (lastResult.category === "receipt") {
        await ensureSpeechVoice(lang, request);
        if (request !== speechRequest) return;
        await TTS.speak(SightLineSpeech.plan(lastResult).text, {
          rate: 0.95,
          lang,
        });
        return;
      }
      // Warn before reading the original words. Do not assemble a spoken dose
      // list that could join directions belonging to different medicines.
      const warned =
        lastResult.warnings && lastResult.warnings.length
          ? "Caution. " + lastResult.warnings.join(" ") + " "
          : "";
      await ensureSpeechVoice("en-GB", request);
      if (request !== speechRequest) return;
      await TTS.speak(
        (lastResult.pageNote ? lastResult.pageNote + ". " : "") +
          warned +
          (lastResult.speechText || lastResult.ocrText || lastResult.summary),
        { rate: 0.95, lang: "en-GB" },
      );
    } catch (e) {
      if (request === speechRequest) showError(TTS.errorMessage(e));
    } finally {
      if (request === speechRequest) speechPreparing = false;
      if (!TTS.speaking() && !speechPreparing) speechLabel("Read Aloud");
    }
  }

  // ── Offline badge + service worker ────────────────────────────────────

  let offlineReady = false,
    offlineMessagesBound = false;
  function updateOnlineBadge() {
    if (offlineReady) setBadge("Ready offline", "ok");
    else
      setBadge(
        navigator.onLine
          ? "Preparing offline reading"
          : "Offline. Reading files may be missing.",
        navigator.onLine ? "pending" : "warn",
      );
  }

  async function registerSW() {
    // WKWebView need not expose serviceWorker. The native host verifies EVERY
    // required PWA asset in the installed bundle before injecting this marker.
    // Its loopback server reads those files directly, including model/WASM files.
    const installed = window.SightLineNativeOffline;
    if (
      location.hostname === "127.0.0.1" &&
      installed?.verified === true &&
      installed.assetCount > 0 &&
      installed.bytes > 0
    ) {
      offlineReady = true;
      updateOnlineBadge();
      return;
    }
    if (!("serviceWorker" in navigator)) {
      setBadge("Offline saving is unavailable", "warn");
      return;
    }
    try {
      const reg = await navigator.serviceWorker.register("sw.js", {
        scope: "./",
      });
      if (reg.active && navigator.serviceWorker.controller) {
        setBadge("Checking offline reading", "pending");
      } else {
        setBadge("Saving for offline reading", "pending");
        reg.addEventListener("updatefound", () => {
          const sw = reg.installing;
          if (sw)
            sw.addEventListener("statechange", () => {
              if (sw.state === "activated")
                setBadge("Checking offline reading", "pending");
            });
        });
      }
      if (!offlineMessagesBound) {
        offlineMessagesBound = true;
        navigator.serviceWorker.addEventListener("message", (e) => {
          if (e.data && e.data.type === "precache-report") {
            const d = e.data;
            offlineReady = d.failed === 0;
            if (offlineReady) updateOnlineBadge();
            else setBadge("Offline saving did not finish", "warn");
          }
        });
      }
      const ready = await navigator.serviceWorker.ready;
      const worker = navigator.serviceWorker.controller || ready.active;
      if (worker) worker.postMessage({ type: "precache-status" });
    } catch (e) {
      setBadge("Offline saving did not finish", "warn");
    }
  }

  // ── Init ──────────────────────────────────────────────────────────────

  function cacheDom() {
    const ids = [
      "live",
      "live-assertive",
      "badge",
      "error-banner",
      "viewfinder",
      "video",
      "preview",
      "scan-line",
      "scan-flash",
      "quality-hint",
      "file-input",
      "btn-capture",
      "btn-pick",
      "btn-retake",
      "btn-speak",
      "speak-lang",
      "cantonese",
      "progress-wrap",
      "progress-label",
      "progress-track",
      "progress-fill",
      "results",
      "doc-type",
      "summary-text",
      "result-meta",
      "fields-card",
      "fields-list",
      "entities-card",
      "btn-camera",
      "source-image",
      "source-details",
      "offline-detail",
      "entities-list",
      "warning-box",
      "ocr-text",
      "skip-restore",
      "skip-classify",
      "ocr-model",
    ];
    for (const id of ids)
      els[id.replace(/-([a-z])/g, (m, c) => c.toUpperCase())] = $(id);
  }

  async function init() {
    cacheDom();
    if (els.ocrModel) {
      try {
        const saved = localStorage.getItem("sightline.reader");
        if (["ppocrv5", "crnn"].includes(saved)) els.ocrModel.value = saved;
      } catch (e) {}
      els.ocrModel.addEventListener("change", () => {
        // Terminate the enhanced reader's heap when switching back, rather than
        // retaining both readers and optional model sessions at once.
        SightLine.dispose();
        try {
          localStorage.setItem("sightline.reader", els.ocrModel.value);
        } catch (e) {}
      });
    }

    // pdf.js worker, same-origin so it is precached and works offline.
    if (window.pdfjsLib && window.pdfjsLib.GlobalWorkerOptions) {
      const base = SightLine.vendorBase();
      window.pdfjsLib.GlobalWorkerOptions.workerSrc =
        base + "vendor/pdfjs/pdf.worker.min.js";
    }

    els.btnCamera.addEventListener("click", startCamera);
    els.btnCapture.addEventListener("click", () => {
      const captured = captureFrame();
      if (!captured) return;
      stopCamera();
      els.video.hidden = true;
      els.btnCapture.hidden = true;
      $("btn-read-photo").hidden = false;
      $("btn-camera-retake").hidden = false;
      $("capture-title").textContent = "Is every word in view?";
      $("capture-help").textContent =
        "Retake if the words look blurry or cut off.";
      $("btn-read-photo").focus();
    });
    $("btn-read-photo").addEventListener("click", () =>
      runScan(els.preview, "camera"),
    );
    $("btn-camera-retake").addEventListener("click", startCamera);
    for (const id of [
      "btn-cancel-camera",
      "btn-result-back",
      "btn-retake",
      "btn-error-home",
      "btn-cancel-reading",
    ])
      $(id).addEventListener("click", goHome);
    els.btnPick.addEventListener("click", pickFile);
    els.btnSpeak.addEventListener("click", speakSummary);
    SightLineI18n.observe();
    $("visible-language").addEventListener("change", (e) => {
      els.speakLang.value = e.target.value;
      languageChanged();
    });
    $("btn-use-english").addEventListener("click", () => {
      els.speakLang.value = "en-GB";
      languageChanged();
    });
    $("btn-history").addEventListener("click", () => {
      stopReading();
      stopCamera();
      if (busy || importing) goHome();
      setStage("history");
      renderHistory();
    });
    $("history-back").addEventListener("click", () => {
      ++scanGeneration;
      setStage(lastResult ? "results" : "home");
    });
    $("btn-save-scan").addEventListener("click", () => saveCurrentScan(true));
    $("save-scans").addEventListener("change", (e) => {
      saveEnabled = e.target.checked;
      storePreferences();
    });
    $("save-images").addEventListener("change", (e) => {
      saveImages = e.target.checked;
      storePreferences();
    });
    $("consent-save").addEventListener("click", () => {
      saveEnabled = true;
      $("save-scans").checked = true;
      storePreferences();
      $("save-consent").close();
      saveCurrentScan(false);
    });
    $("history-clear").addEventListener("click", () => confirmDelete(null));
    $("confirm-delete").addEventListener("click", async () => {
      const id = deleteTarget;
      $("confirm-delete").disabled = true;
      try {
        if (id) await SightLineHistory.remove(id);
        else {
          await SightLineHistory.clear();
          saveEnabled = false;
          $("save-scans").checked = false;
          storePreferences();
        }
        if (!id || id === savedId) {
          savedId = null;
          $("save-status").textContent = "";
          $("btn-save-scan").textContent = SightLineI18n.t("Save scan");
        }
        $("history-delete-dialog").close();
        await renderHistory();
      } catch (e) {
        $("delete-description").textContent = SightLineHistory.message(e);
      } finally {
        $("confirm-delete").disabled = false;
      }
    });
    try {
      const consent = JSON.parse(
        localStorage.getItem("sightline.history-consent") || "{}",
      );
      saveEnabled = consent.enabled === true;
      saveImages = consent.images === true;
    } catch {}
    $("save-scans").checked = saveEnabled;
    $("save-images").checked = saveImages;

    TTS.loadVoices().then(updateSpeechPreview);
    window.speechSynthesis?.addEventListener(
      "voiceschanged",
      updateSpeechPreview,
    );
    TTS.subscribe((state) => {
      const active = ["preparing", "starting", "speaking"].includes(
        state.status,
      );
      speechLabel(active ? "Stop reading" : "Read Aloud");
      els.btnSpeak.setAttribute("aria-pressed", String(active));
      const messages = {
        preparing: "Finding an offline voice…",
        starting: "Starting the voice…",
        speaking: "Reading aloud. Tap Stop reading to stop.",
        ended: "Reading finished. You can listen again.",
        stopped: "Reading stopped. Tap Read Aloud to listen again.",
      };
      let playbackMessage = messages[state.status] || "";
      if (state.status === "speaking" && state.outputVolume !== null) {
        if (state.outputVolume <= 0.01) {
          playbackMessage = SightLineI18n.t(
            "No sound: your iPhone media volume is at 0%. Press volume up and try again.",
          );
        } else if (state.outputKind === "none") {
          playbackMessage = SightLineI18n.t(
            "Reading could not find an audio output. Check your volume and connected headphones.",
          );
        } else {
          const volume = Math.round(state.outputVolume * 100);
          const volumeLabel = SightLineI18n.t("iPhone media volume:");
          const kind = /Bluetooth/i.test(state.outputKind || "")
            ? "Bluetooth"
            : /Headphones|Headset/i.test(state.outputKind || "")
              ? "Headphones"
              : /Speaker|Receiver/i.test(state.outputKind || "")
                ? "Speaker"
                : "";
          if (kind === "Bluetooth")
            playbackMessage = `${SightLineI18n.t("Sound is going to Bluetooth audio.")} ${volumeLabel} ${volume}%.`;
          else if (kind === "Headphones")
            playbackMessage = `${SightLineI18n.t("Sound is going to headphones.")} ${volumeLabel} ${volume}%.`;
          else if (kind === "Speaker")
            playbackMessage = `${SightLineI18n.t("Sound is going to the iPhone speaker.")} ${SightLineI18n.t("Media volume:")} ${volume}%.`;
          else
            playbackMessage = `${SightLineI18n.t("The iPhone voice started.")} ${state.voiceName || ""}. ${SightLineI18n.t("Media volume:")} ${volume}%.`;
        }
      }
      if (screen === "results") {
        $("playback-status").hidden = false;
        $("playback-status").textContent =
          state.status === "error"
            ? TTS.errorMessage({ code: state.error })
            : playbackMessage;
      }
    });
    els.fileInput.addEventListener("change", onFile);
    $("btn-retry").addEventListener("click", () => {
      // A failed module fetch can remain failed in WebKit's module map even
      // after the server recovers. A new document resets that map safely.
      if (readerNeedsReload) { window.location.reload(); return; }
      if (retrySource === "scan" && lastAttempt)
        runScan(lastAttempt.source, lastAttempt.name, lastAttempt.note);
      else if (retrySource === "camera") startCamera();
      else pickFile();
    });
    function closeDevice() {
      $("device-frame").contentWindow?.postMessage(
        { type: "stop-device-voice" },
        location.origin,
      );
      $("device-dialog").close();
      $("btn-help").focus();
    }
    document.querySelectorAll("[data-open-device]").forEach((button) =>
      button.addEventListener("click", () => {
        stopReading();
        button.closest("dialog")?.close();
        if (!$("device-frame").getAttribute("src"))
          $("device-frame").src = "selftest.html?embedded=1";
        $("device-frame").contentWindow?.postMessage(
          {
            type: "device-theme",
            theme: document.documentElement.dataset.theme,
          },
          location.origin,
        );
        $("device-dialog").showModal();
      }),
    );
    $("close-device").addEventListener("click", closeDevice);
    $("device-dialog").addEventListener("cancel", (e) => {
      e.preventDefault();
      closeDevice();
    });
    window.addEventListener("message", (e) => {
      if (
        e.origin === location.origin &&
        e.source === $("device-frame").contentWindow &&
        e.data?.type === "sightline-device-close"
      )
        closeDevice();
    });
    for (const id of ["btn-preferences", "btn-language"])
      $(id).addEventListener("click", () => openDialog("preferences-dialog"));
    for (const id of ["btn-help", "btn-footer-help"])
      $(id).addEventListener("click", () => openDialog("help-dialog"));
    document
      .querySelectorAll("[data-close-dialog]")
      .forEach((button) =>
        button.addEventListener("click", () =>
          button.closest("dialog").close(),
        ),
      );
    try {
      const saved = localStorage.getItem("sightline.language");
      if (["zh-HK", "en-GB"].includes(saved)) els.speakLang.value = saved;
    } catch (e) {}
    els.speakLang.addEventListener("change", languageChanged);
    languageChanged();
    try {
      const saved = localStorage.getItem("sightline.appearance");
      if (["light", "dark", "system"].includes(saved))
        $("appearance").value = saved;
    } catch (e) {}
    $("appearance").addEventListener("change", appearanceChanged);
    appearanceChanged();
    window
      .matchMedia("(prefers-color-scheme: dark)")
      .addEventListener("change", () => {
        if ($("appearance").value === "system") appearanceChanged();
      });
    window.addEventListener("online", updateOnlineBadge);
    window.addEventListener("offline", updateOnlineBadge);
    window.addEventListener("beforeunload", () => {
      stopCamera();
      stopReading();
    });

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
      stopReading();
      try {
        SightLine.dispose();
      } catch (e) {
        /* nothing was loaded */
      }
    });

    // Live quality hint while the camera is up.
    if (els.video) {
      els.video.addEventListener("loadeddata", () => {
        els.qualityHint.hidden = false;
        els.qualityHint.textContent = "Fill the frame with the document";
        window.setTimeout(() => {
          els.qualityHint.hidden = true;
        }, 2500);
      });
    }

    $("btn-save-offline").addEventListener("click", () => {
      setBadge("Saving for offline reading", "pending");
      registerSW();
    });
    updateOnlineBadge();
    registerSW();
    // Camera permission belongs to a real tap. Optional model sessions are
    // initialized by restore()/classify() only when a scan actually uses them;
    // startup no longer compiles an unused second WASM heap alongside PP-OCR.
    setStage("home", false);
    // Warm JavaScript only, while native loopback serving is active. No camera,
    // model session, WASM heap or inference is started here.
    window.__SightLineReaderWarmup = SightLine.prepareReader().catch((error) => {
      readerNeedsReload = true;
      updateBadge();
      return { error: error.code || "READER_LOAD_FAILED" };
    });
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
    get lastResult() {
      return lastResult;
    },
  };
})();
