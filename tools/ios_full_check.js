/**
 * Extensive browser test of the SHIPPED app.
 *
 * tools/browser_smoke.js checks the shell; tools/ios_webkit_check.js runs the
 * selftest battery in WebKit; this drives the REAL controller the user touches
 * (file input -> runScan -> results) and then attacks the paths none of the
 * others reach: session release and re-scan, the no-WASM-SIMD device, the
 * refusal cases, the PDF path, the offline reload, and the iOS pagehide
 * release. Every suite runs in whichever engine you ask for — WebKit is iOS
 * Safari's engine, Chromium is the sanity check that a failure is not
 * engine-specific.
 *
 * Fixtures (regenerate before running):
 *   /tmp/sightline_samples/*       bash run.sh py app/tools/render_samples.py
 *   blank/noise/letter.pdf         bash run.sh py app/tools/make_fixtures.py
 *
 * Usage: node tools/ios_full_check.js [--engine=webkit|chromium|both] [--suites=a,b,c]
 *                                   [--json=artifacts/pipeline_raw.json]
 * Exit code 0 only when every check in every suite passed.
 */
"use strict";

const http = require("http");
const fs = require("fs");
const os = require("os");
const path = require("path");

const APP = path.resolve(__dirname, "..");
const SAMPLES = "/tmp/sightline_samples";
const TIMEOUT_DONE = 240000;

// ── playwright-core / browsers ───────────────────────────────────────────────
const pw = require("./pw_core.js");
const chromeBin = require("./chrome_path.js");


// ── static server over app/ ──────────────────────────────────────────────────
const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm", ".png": "image/png", ".ico": "image/x-icon",
  ".md": "text/markdown; charset=utf-8",
};

function serve() {
  const state = { requested: [], notFound: [] };
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, "http://localhost");
    let p = decodeURIComponent(url.pathname);
    if (p === "/") p = "/index.html";
    // Fixtures are served under /samples/ so the page can fetch them by URL and
    // push them through the SAME pipeline functions the app uses (file:// is
    // blocked in both engines, which is why this is done over HTTP).
    const file = p.startsWith("/samples/")
      ? path.join(SAMPLES, p.slice("/samples/".length))
      : path.join(APP, p);
    if (!file.startsWith(p.startsWith("/samples/") ? SAMPLES : APP)
        || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      state.notFound.push(p);
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("not found: " + p);
      return;
    }
    state.requested.push(p);
    res.writeHead(200, {
      "Content-Type": MIME[path.extname(file)] || "application/octet-stream",
      "Cache-Control": "no-store",
      "Service-Worker-Allowed": "/",
    });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve({ server, state })));
}

const ALL_RECORDS = [];   // bench records from every engine that ran

// ── tiny assertion harness ───────────────────────────────────────────────────
class Report {
  constructor(engine) {
    this.engine = engine;
    this.suites = [];
    this.current = null;
    this.records = [];
  }
  suite(name) {
    this.current = { name, checks: [] };
    this.suites.push(this.current);
    process.stdout.write("\n[" + this.engine + "] " + name + "\n");
  }
  check(name, pass, detail) {
    this.current.checks.push({ name, pass, detail });
    console.log("  " + (pass ? "ok  " : "FAIL") + "  " + name
      + (pass || detail === undefined ? "" : "  -> " + String(detail).slice(0, 300)));
    return pass;
  }
  /** Raw per-sample record: data, no verdicts. The metric definitions live in
   *  ml/src/bench_hackathon.py so the two cannot disagree. */
  record(rec) {
    this.records.push(rec);
  }
  get count() {
    return this.records.length;
  }
  get failures() {
    return this.suites.flatMap((s) => s.checks.filter((c) => !c.pass).map((c) => s.name + " / " + c.name));
  }
  get total() {
    return this.suites.reduce((n, s) => n + s.checks.length, 0);
  }
}

// ── app helpers (page-side) ──────────────────────────────────────────────────
const DONE_SIGNAL = `(() => {
  const r = document.getElementById("results");
  const b = document.getElementById("error-banner");
  return (r && !r.hidden) || (b && !b.hidden && (b.textContent||"").trim().length > 0);
})()`;

async function scanFile(page, file) {
  await page.evaluate(() => {
    const r = document.getElementById("results");
    const b = document.getElementById("error-banner");
    if (r) r.hidden = true;
    if (b) { b.hidden = true; b.textContent = ""; }
    window.__lastScan = null;
  });
  await page.setInputFiles("#file-input", file);
  let timedOut = false;
  try {
    await page.waitForFunction(DONE_SIGNAL, null, { timeout: TIMEOUT_DONE });
  } catch (e) { timedOut = true; }
  const state = await page.evaluate(() => {
    const r = document.getElementById("results");
    const b = document.getElementById("error-banner");
    const lr = window.SightLineApp && window.SightLineApp.lastResult;
    const ocrEl = document.getElementById("ocr-text");
    return {
      resultsShown: !!(r && !r.hidden),
      error: b && !b.hidden ? (b.textContent || "").trim() : "",
      // understand() deliberately does not return the raw text; the OCR text
      // the user can inspect lives in the #ocr-text element.
      // The app writes the literal placeholder into #ocr-text when nothing was
      // recognised, so a length check ALONE passes on an empty read.
      ocrText: ocrEl ? (ocrEl.textContent || "").trim() : "",
      ocrRead: !!ocrEl && !/^\(no text recognised\)$/.test((ocrEl.textContent || "").trim()),
      lastResult: lr ? {
        category: lr.category, categoryLabel: lr.categoryLabel,
        // understand() DOES return ocrText (it is rawText in pipeline.js); the
        // old note here claiming otherwise was stale. Carry it, so a refusal
        // failure can show WHAT was read as well as what was invented.
        ocrText: (lr.ocrText || "").slice(0, 200),
        fields: (lr.fields || []).map((f) => (f && (f.value || f.text || f.raw)) || String(f)),
        ocrConfidence: lr.ocrConfidence, classificationConfidence: lr.classificationConfidence,
        classificationSource: lr.classificationSource,
        warnings: (lr.warnings || []).length, engine: lr.engine, ms: lr.ms,
        sourceName: lr.sourceName,
      } : null,
    };
  });
  return { timedOut, ...state };
}

async function recogniserReady(page, opts) {
  const doInit = !!(opts && opts.init);
  return page.evaluate(async (init) => {
    const m = await import("./js/recognizer.js");
    if (init) await m.init();
    return { ready: m.isReady(), error: m.getError(), simd: m.simdSupported() };
  }, doInit);
}

async function newPage(ctx) {
  const page = await ctx.context.newPage();
  page.on("pageerror", (e) => ctx.pageErrors.push(String((e && e.message) || e)));
  page.on("console", (m) => { if (m.type() === "error") ctx.consoleErrors.push(m.text()); });
  return page;
}

// ── suites ───────────────────────────────────────────────────────────────────
const SUITES = [];
const suite = (name, fn) => SUITES.push({ name, fn });

suite("boot — shell, service worker, layout, a11y", async (ctx, R) => {
  const page = ctx.page;
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  await page.waitForTimeout(2500);

  const g = await page.evaluate(`(() => ({
    pipeline: typeof SightLine, restorer: typeof SightLineRestorer,
    classifier: typeof SightLineClassifier, tts: typeof TTS,
    app: typeof SightLineApp, tesseract: typeof Tesseract,
    speech: typeof window.speechSynthesis,
    sw: !!navigator.serviceWorker,
    hOverflow: document.documentElement.scrollWidth - window.innerWidth,
    live: (document.getElementById("live")||{}).getAttribute
      ? document.getElementById("live").getAttribute("aria-live") : null,
    smallButtons: Array.from(document.querySelectorAll("button"))
      .filter(b => b.offsetParent !== null && b.getBoundingClientRect().height < 44)
      .map(b => (b.textContent||"").trim().slice(0,24)),
  }))()`);

  R.check("pipeline global present", g.pipeline === "object", g.pipeline);
  R.check("classifier + restorer + tts + app globals present",
    g.classifier === "object" && g.restorer === "object" && g.tts === "object" && g.app === "object",
    JSON.stringify(g));
  R.check("tesseract is not loaded (own CRNN instead)", g.tesseract === "undefined", g.tesseract);
  R.check("no horizontal overflow at 390px", g.hOverflow <= 1, g.hOverflow + "px");
  R.check("aria-live region is polite", g.live === "polite", g.live);
  R.check("every visible button is >= 44px tall", g.smallButtons.length === 0, JSON.stringify(g.smallButtons));

  const sw = await page.evaluate(async () => {
    const reg = await navigator.serviceWorker.getRegistration();
    const keys = await caches.keys();
    return { scope: reg && reg.scope, active: !!(reg && reg.active), keys };
  });
  R.check("service worker registered and active", !!sw.active, JSON.stringify(sw));
  // DERIVE the expected cache name from the shipped source instead of hardcoding
  // it. This said "sightline-v1.1.5" and went red the moment the model changed and
  // the cache name had to be bumped -- a check failing for a reason unrelated to
  // what it tests, which is how checks get ignored. Reading sw.js tests the real
  // property: the RUNTIME cache name matches the SHIPPED source.
  const swSrc = fs.readFileSync(path.join(APP, "sw.js"), "utf8");
  const shipped = (swSrc.match(/const VERSION = "([^"]+)"/) || [])[1];
  R.check("cache is the shipped VERSION",
    !!shipped && sw.keys.includes("sightline-" + shipped),
    "shipped=" + shipped + " runtime=" + JSON.stringify(sw.keys));
  R.check("no page errors on boot", ctx.pageErrors.length === 0, ctx.pageErrors.join(" | "));
  R.check("no console errors on boot", ctx.consoleErrors.length === 0, ctx.consoleErrors.join(" | "));
  R.check("server served every requested asset", ctx.serverState.notFound.length === 0,
    ctx.serverState.notFound.join(", "));
});

suite("wasm — SIMD probe agrees with the engine", async (ctx, R) => {
  const page = ctx.page;
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  const rec = await recogniserReady(page, { init: true });
  R.check("simdSupported() is true on this engine", rec.simd === true, JSON.stringify(rec));
  R.check("recogniser session created", rec.ready === true, JSON.stringify(rec));
});

suite("scan — studio_clean documents (3 types)", async (ctx, R) => {
  const page = ctx.page;
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  await page.waitForTimeout(1500);
  for (const doc of ["prescription", "banking", "legal"]) {
    const file = SAMPLES + "/" + doc + "__studio_clean.png";
    const r = await scanFile(page, file);
    const exp = ctx.manifest;
    R.check(doc + ": scan completed without error", !r.timedOut && !r.error,
      r.timedOut ? "timed out" : r.error);
    R.check(doc + ": results shown and result object present", r.resultsShown && !!r.lastResult,
      JSON.stringify(r).slice(0, 200));
    if (!r.lastResult) continue;
    R.check(doc + ": classified as " + exp.category[doc], r.lastResult.category === exp.category[doc],
      r.lastResult.category + " (" + r.lastResult.categoryLabel + ")");
    // COMPARE THE WAY THE PRODUCT COMPARES. This used a raw lowercase substring,
    // so the extracted value "400 MG" did not match the ground truth "400mg" and
    // a document the app had plainly recovered was scored a miss -- while the
    // read text actually contained "400mg". The app's own metric (and Python's
    // evaluate.fields_found, which this mirrors) normalises by lowercasing and
    // stripping everything that is not [a-z0-9], where the two are identical.
    // Neither a space nor a case difference is a recovery failure; a missing
    // digit is, and that is what still fails below.
    const norm = (s) => String(s == null ? "" : s).toLowerCase().replace(/[^a-z0-9]/g, "");
    const want = exp.fields[doc].map(norm);
    const got = norm((r.lastResult.fields || []).join(" ") + " " + (r.lastResult.ocrText || ""));
    R.check(doc + ": recovers at least one ground-truth field (" + want.join("/") + ")",
      want.some((w) => got.includes(w.replace(/[,.]/g, "")) || got.includes(w)), got.slice(0, 120));
    R.check(doc + ": OCR text produced", r.ocrRead && r.ocrText.length > 10,
      "ocr chars=" + r.ocrText.length + " read=" + r.ocrRead);
    R.check(doc + ": OCR confidence sane (0..1)", r.lastResult.ocrConfidence > 0 && r.lastResult.ocrConfidence <= 1,
      r.lastResult.ocrConfidence);
    R.check(doc + ": no page errors during scan", ctx.pageErrors.length === 0, ctx.pageErrors.join(" | "));
  }
});

suite("scan — handheld_light captures (honest degradation)", async (ctx, R) => {
  const page = ctx.page;
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  await page.waitForTimeout(1500);
  for (const doc of ["prescription", "banking", "legal"]) {
    const r = await scanFile(page, SAMP_DOC(doc, "handheld_light"));
    R.check(doc + " handheld: no crash and a result or an honest error",
      !r.timedOut && (!!r.lastResult || !!r.error), r.timedOut ? "timed out" : r.error);
    if (!r.lastResult) continue;
    // The product rule: a low-confidence read must be WARNED about, and a read
    // with no fields must say so. Either is honest; silence is the bug.
    const warned = r.lastResult.warnings > 0;
    const weak = r.lastResult.ocrConfidence < 0.6;
    const empty = r.lastResult.fields.length === 0;
    R.check(doc + " handheld: low confidence or no fields implies a warning",
      (!weak && !empty) || warned,
      "conf=" + r.lastResult.ocrConfidence + " fields=" + r.lastResult.fields.length + " warnings=" + r.lastResult.warnings);
  }
});

function SAMP_DOC(doc, profile) { return SAMPLES + "/" + doc + "__" + profile + ".png"; }

suite("release — dispose() then scan again (iOS memory path)", async (ctx, R) => {
  const page = ctx.page;
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  await page.waitForTimeout(1500);

  for (const round of [1, 2, 3]) {
    const r = await scanFile(page, SAMP_DOC("banking", "studio_clean"));
    R.check("round " + round + ": scan produced a result", !!r.lastResult && !r.error,
      r.error || JSON.stringify(r.lastResult && r.lastResult.fields));
    const before = await recogniserReady(page);
    R.check("round " + round + ": recogniser is ready mid-flight", before.ready === true, JSON.stringify(before));
    await page.evaluate(async () => { await window.SightLine.dispose(); });
    await page.waitForTimeout(600);
    const after = await recogniserReady(page);
    R.check("round " + round + ": dispose() released the recogniser session", after.ready === false,
      JSON.stringify(after));
  }
  R.check("re-scan after release leaves no page errors", ctx.pageErrors.length === 0, ctx.pageErrors.join(" | "));
});

suite("no-simd device — the iOS < 16.4 path explains itself", async (ctx, R) => {
  // A fresh context with WebAssembly.validate spoofed to reject the recogniser's
  // SIMD probe — the device class iOS 15/16.0-16.3 actually is.
  const page = await ctx.rawContext.newPage();
  const errs = [];
  page.on("pageerror", (e) => errs.push(String((e && e.message) || e)));
  await page.addInitScript(() => {
    const orig = WebAssembly.validate.bind(WebAssembly);
    WebAssembly.validate = function (bytes) {
      let b;
      try { b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes); } catch (e) { b = null; }
      // the recogniser's probe: a 31-byte module declaring f(..)->v128, i.e.
      // the result type byte 0x7b at index 14 of the type section
      if (b && b.length === 31 && b[14] === 123) return false;
      return orig(bytes);
    };
  });
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  await page.waitForTimeout(1200);

  const probe = await page.evaluate(async () => {
    const m = await import("./js/recognizer.js");
    const s = await m.init();
    return { simd: m.simdSupported(), session: s === null ? null : "created", error: m.getError() };
  });
  R.check("simdSupported() is false under the spoof", probe.simd === false, JSON.stringify(probe));
  R.check("init() resolves to null instead of throwing", probe.session === null, JSON.stringify(probe));
  R.check("the error names the real cause (WebAssembly SIMD / iOS version)",
    /WebAssembly SIMD/i.test(probe.error || "") && /16\.4/.test(probe.error || ""), probe.error);

  const t0 = Date.now();
  const r = await scanFile(page, SAMP_DOC("banking", "studio_clean"));
  const took = Date.now() - t0;
  R.check("a scan on a no-SIMD device fails fast, not by hanging", !r.timedOut && took < 60000,
    took + "ms timeout=" + r.timedOut);
  R.check("the user sees the SIMD reason, not a generic failure",
    /WebAssembly SIMD/i.test(r.error || ""), r.error);
  R.check("no unhandled page errors on the no-SIMD path", errs.length === 0, errs.join(" | "));
  await page.close();
});

suite("refusal — blank and speckle pages are not guessed at", async (ctx, R) => {
  const page = ctx.page;
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  await page.waitForTimeout(1500);

  for (const [label, file] of [["blank", "/tmp/blank.png"], ["speckle", "/tmp/noise.png"]]) {
    const r = await scanFile(page, file);
    R.check(label + ": no crash", !r.timedOut && (!!r.lastResult || !!r.error),
      r.timedOut ? "timed out" : r.error);
    if (!r.lastResult) continue;
    // Report the read's own confidence alongside the fields: whether a field was
    // invented is a question about the READ, and without this number a failure
    // here says "bad" but not "how bad" or "how close to right it was".
    R.check(label + ": invents no fields", r.lastResult.fields.length === 0,
      "fields=" + JSON.stringify(r.lastResult.fields).slice(0, 120) +
      " ocrConfidence=" + (r.lastResult.ocrConfidence == null
        ? "?" : r.lastResult.ocrConfidence.toFixed(3)) +
      " text=" + JSON.stringify(r.lastResult.ocrText || "").slice(0, 60));
    R.check(label + ": says it could not read the document", r.lastResult.warnings > 0,
      "warnings=" + r.lastResult.warnings);
  }
  R.check("refusal path leaves no page errors", ctx.pageErrors.length === 0, ctx.pageErrors.join(" | "));
});

suite("pdf — a letter that arrives as a PDF", async (ctx, R) => {
  const page = ctx.page;
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  await page.waitForTimeout(1500);
  const r = await scanFile(page, "/tmp/letter.pdf");
  R.check("pdf scan completed without error", !r.timedOut && !r.error,
    r.timedOut ? "timed out" : r.error);
  if (r.lastResult) {
    R.check("pdf produced OCR text", r.ocrRead && r.ocrText.length > 10,
      "ocr chars=" + r.ocrText.length + " read=" + r.ocrRead);
    R.check("pdf scan is labelled as a file, not a camera frame",
      /pdf|file/i.test(r.lastResult.sourceName + " " + (r.lastResult.pageNote || "")),
      r.lastResult.sourceName + " / " + r.lastResult.pageNote);
  }
});

suite("offline — reload with no network, then scan", async (ctx, R) => {
  const page = ctx.page;
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  await page.waitForTimeout(1500);
  const warm = await scanFile(page, SAMP_DOC("banking", "studio_clean"));
  R.check("online scan warms the cache", !!warm.lastResult, warm.error);

  // The app asks the SW to precache the models after the first successful scan
  // (precacheModelsOnce). Without that call the models live only in the
  // evictable HTTP cache, which is not the offline guarantee.
  let cachedModels = [];
  for (let i = 0; i < 30; i++) {
    cachedModels = await page.evaluate(async () => {
      const keys = (await caches.keys()).filter((k) => k.startsWith("sightline-"));
      const c = await caches.open(keys[keys.length - 1]);
      return (await c.keys()).map((r) => new URL(r.url).pathname);
    });
    if (cachedModels.some((p) => /crnn\.onnx$/.test(p))) break;
    await page.waitForTimeout(1000);
  }
  R.check("after one scan the recogniser model is in the SERVICE-WORKER cache",
    cachedModels.some((p) => /crnn\.onnx$/.test(p)),
    cachedModels.filter((p) => /models|ort/.test(p)).join(", ") || "(no model entries)");
  R.check("after one scan the ORT wasm runtime is in the SERVICE-WORKER cache",
    cachedModels.some((p) => /ort-wasm.*\.wasm$/.test(p)),
    cachedModels.filter((p) => /ort/.test(p)).join(", ") || "(no ort entries)");

  await ctx.context.setOffline(true);
  await page.reload({ waitUntil: "load" });
  await page.waitForTimeout(2500);
  const boot = await page.evaluate(() => ({
    pipeline: typeof SightLine,
    badge: (document.getElementById("badge") || {}).textContent || "",
  }));
  R.check("app boots while OFFLINE", boot.pipeline === "object", JSON.stringify(boot));
  R.check("offline badge is shown", /offline/i.test(boot.badge), boot.badge);
  const off = await scanFile(page, SAMP_DOC("banking", "studio_clean"));
  R.check("scan still works OFFLINE (models came from cache)", !!off.lastResult && !off.error,
    off.error || "no result");
  await ctx.context.setOffline(false);
});

suite("tts — chunking and the no-voice path", async (ctx, R) => {
  const page = ctx.page;
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  const t = await page.evaluate(() => {
    const blurb = "Your prescription is for Amoxicillin 500mg three times daily. "
      + "Take it with food. Do not skip doses. If you feel drowsy, do not drive. "
      + "Finish the whole course even if you feel better. Store below 25 degrees.";
    const chunks = TTS.chunkText(blurb, 180);
    return {
      n: chunks.length,
      all: chunks.every((c) => c.length <= 180),
      words: chunks.join(" ").split(/\s+/).length === blurb.trim().split(/\s+/).length,
      empty: TTS.chunkText("", 180).length,
      available: TTS.available(),
      voices: (window.speechSynthesis && window.speechSynthesis.getVoices().length) || 0,
    };
  });
  R.check("long text splits into multiple chunks", t.n > 1, t.n);
  R.check("every chunk is within the 180-char iOS limit", t.all === true, JSON.stringify(t));
  R.check("chunking loses no words", t.words === true, JSON.stringify(t));
  R.check("empty text yields no chunks", t.empty === 0, t.empty);
  R.check("TTS reports availability honestly", typeof t.available === "boolean", JSON.stringify(t));

  // speak() must not reject unhandled when the engine has no voices (headless).
  const spoke = await page.evaluate(async () => {
    try { await TTS.speak("test", { rate: 1 }); return "resolved"; }
    catch (e) { return "rejected: " + e.message; }
  });
  R.check("speak() settles cleanly with no voices", /resolved|rejected:/.test(spoke), spoke);
});

suite("pagehide — the iOS memory release actually fires", async (ctx, R) => {
  const page = ctx.page;
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  await page.waitForTimeout(1500);
  await scanFile(page, SAMP_DOC("legal", "studio_clean"));
  const before = await recogniserReady(page);
  R.check("recogniser ready before pagehide", before.ready === true, JSON.stringify(before));

  await page.evaluate(() => window.dispatchEvent(new Event("pagehide")));
  await page.waitForTimeout(800);
  const after = await recogniserReady(page);
  R.check("pagehide releases the recogniser session", after.ready === false, JSON.stringify(after));

  const res = await page.evaluate(async () => {
    const c = window.SightLineClassifier.status();
    return { classifierReady: c.ready, error: c.error };
  });
  R.check("pagehide also releases the classifier (22.9 MB encoder)", res.classifierReady === false,
    JSON.stringify(res));

  const r = await scanFile(page, SAMP_DOC("legal", "studio_clean"));
  R.check("a scan after pagehide re-initialises and succeeds", !!r.lastResult && !r.error,
    r.error || "no result");
});

suite("geometry — what the recogniser is actually fed", async (ctx, R) => {
  const page = ctx.page;
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  await page.waitForTimeout(1000);

  const geom = await page.evaluate(async () => {
    const layout = await import("./js/layout.js");
    const img = new Image();
    img.src = "/samples/banking__studio_clean.png";
    await img.decode();
    const c = document.createElement("canvas");
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    c.getContext("2d", { willReadFrequently: true }).drawImage(img, 0, 0);
    const pp = window.SightLine.preprocess(c, 3200, {});
    const cv = pp.canvas;
    const d = cv.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, cv.width, cv.height).data;
    const gray = new Uint8Array(cv.width * cv.height);
    for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
      gray[i] = (d[p] * 77 + d[p + 1] * 150 + d[p + 2] * 29) >> 8;
    }
    const boxes = layout.detectLines(gray, cv.width, cv.height);
    // EXPORT_W = 256 in recognizer.js, and ocr_engine passes it as maxW.
    const crops = boxes.map((b) => {
      const cr = layout.cropForRecognizer(gray, cv.width, b, 32, 256);
      const rows = cr.data.length / cr.width;              // always 32
      let filled = 0;
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cr.width; x++) if (cr.data[y * cr.width + x] !== 0) { filled++; break; }
      }
      return { bw: b.x1 - b.x0, bh: b.y1 - b.y0, nw: cr.width, filledRows: filled };
    });
    return { page: cv.width + "x" + cv.height, lines: boxes.length, crops };
  });

  R.check("the detector finds text lines on the page", geom.lines > 0,
    geom.lines + " lines on " + geom.page);
  const withCrop = (geom.crops || []).filter((c) => c.bw > 0);
  const tall = withCrop.filter((c) => c.filledRows >= 32).length;
  // CROP_H = 32 is the height the recogniser was trained at (realdata.CROP_H);
  // a crop shorter than that is resampled text with a zero-filled band, and
  // zero maps to -1 = ink in buildInput, not to paper.
  R.check("every crop fills the model's full 32px training height",
    withCrop.length > 0 && tall === withCrop.length,
    tall + "/" + withCrop.length + " crops are 32px tall; e.g. box "
      + JSON.stringify(withCrop.slice(0, 4).map((c) => c.bw + "x" + c.bh + " -> " + c.nw + "x" + c.filledRows)));
  R.check("median crop keeps at least half the training height",
    withCrop.length > 0
      && withCrop.map((c) => c.filledRows).sort((a, b) => a - b)[Math.floor(withCrop.length / 2)] >= 16,
    JSON.stringify(withCrop.map((c) => c.filledRows).slice(0, 8)));
});

suite("cold install — install online, airplane mode before the FIRST scan", async (ctx, R) => {
  // The documented offline rehearsal (docs/04 §4) is: load, SCAN ONCE, then
  // airplane mode. This suite is the case that comes *before* that: a fresh
  // install that has never scanned, taken offline. The engine modules are
  // precached, so the app boots and the failure must be honest and actionable
  // rather than a hang or a generic "scan failed".
  const context = ctx.rawContext;
  const page = await context.newPage();
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  await page.waitForTimeout(4000);                       // SW install + precache
  const keys = await page.evaluate(() => caches.keys());
  R.check("service worker installed a cache", keys.length > 0, JSON.stringify(keys));
  const pre = await page.evaluate(async () => {
    const c = await caches.open((await caches.keys()).find((k) => k.startsWith("sightline-")));
    return (await c.keys()).map((r) => new URL(r.url).pathname);
  });
  R.check("the OCR engine modules are precached at install (no scan needed)",
    pre.some((p) => /ocr_engine\.js$/.test(p)) && pre.some((p) => /layout\.js$/.test(p)),
    pre.filter((p) => /engine|layout/.test(p)).join(", ") || "(missing)");

  await context.setOffline(true);
  await page.reload({ waitUntil: "load" });
  await page.waitForTimeout(2500);
  const boot = await page.evaluate(() => typeof SightLine);
  R.check("app boots offline straight after install", boot === "object", boot);

  const r = await scanFile(page, SAMP_DOC("banking", "studio_clean"));
  const msg = r.error || "";
  R.check("a never-scanned install fails offline with a message that names the cause",
    !r.timedOut && !!msg && /fetch|load|module|model|network|offline/i.test(msg),
    r.timedOut ? "timed out instead of reporting" : msg);
  await context.setOffline(false);
  await page.close();
});

// ── the benchmark the Imperial brief asks to be measured ─────────────────────
// Drives the SHIPPED app over every rendered sample and records what it did.
// Deliberately no metrics here: ml/src/bench_hackathon.py owns the definitions
// (word accuracy, CER, field recovery, classification), so the browser and the
// report cannot drift apart. Writes with --json=<path>.
suite("bench — the shipped pipeline, one record per sample", async (ctx, R) => {
  const page = ctx.page;
  await page.goto(ctx.base + "/index.html", { waitUntil: "load" });
  await page.waitForTimeout(1500);
  const m = ctx.manifest;
  const docs = Object.keys(m.text || {});
  const profiles = m.profiles || [];
  for (const doc of docs) {
    for (const profile of profiles) {
      const r = await scanFile(page, SAMP_DOC(doc, profile));
      R.record({
        doc, profile,
        expected_category: (m.category || {})[doc] || null,
        expected_fields: (m.fields || {})[doc] || [],
        expected_text: (m.text || {})[doc] || "",
        // WHICH CANDIDATE WON. preprocess() now offers the engine three images
        // (raw deskewed grayscale, CLAHE, CLAHE+Otsu) and runOCR ranks them by
        // structure_score; without this the bench cannot say whether a candidate
        // change did anything, only that the aggregate moved or did not.
        psm_rank: r.lastResult ? r.lastResult.psmRank : null,
        structure: r.lastResult ? r.lastResult.structure : null,
        category: r.lastResult ? r.lastResult.category : null,
        fields: r.lastResult ? r.lastResult.fields : [],
        ocr_text: r.ocrRead ? r.ocrText : "",
        ocr_confidence: r.lastResult ? r.lastResult.ocrConfidence : null,
        // a sample that produced nothing is a ZERO in the report, not a missing
        // row, so the failure has to be recorded as a reason here.
        error: r.error || (r.timedOut ? "scan timed out"
          : (r.lastResult || r.error ? null : "no result produced")),
        ms: r.lastResult ? r.lastResult.ms : null,
      });
      process.stdout.write(".");
    }
  }
  console.log("");
  const want = (m.samples || []).length;
  R.check("recorded one row per sample (no silent skips)", R.count === want,
    R.count + " of " + want);
  R.check("every row carries the read text or a reason it is empty",
    R.records.every((x) => x.ocr_text || x.error), "some rows are silently empty");
});

// ── runner ───────────────────────────────────────────────────────────────────
async function runEngine(engineName, jobs) {
  const R = new Report(engineName);
  const { server, state } = await serve();
  const base = "http://127.0.0.1:" + server.address().port;
  const pwc = pw();
  let browser;
  if (engineName === "webkit") {
    browser = await pwc.webkit.launch();
  } else {
    browser = await pwc.chromium.launch({
      executablePath: chromeBin(), args: ["--no-sandbox", "--disable-dev-shm-usage"],
    });
  }
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const manifest = JSON.parse(fs.readFileSync(SAMPLES + "/manifest.json", "utf8"));

  for (const s of jobs) {
    const ctx = {
      base, context, manifest, serverState: state,
      pageErrors: [], consoleErrors: [],
    };
    ctx.page = await context.newPage();
    ctx.page.on("pageerror", (e) => ctx.pageErrors.push(String((e && e.message) || e)));
    ctx.page.on("console", (m) => { if (m.type() === "error") ctx.consoleErrors.push(m.text()); });
    // suites that need their own context (addInitScript) get this raw one
    ctx.rawContext = await browser.newContext({ viewport: { width: 390, height: 844 } });
    R.suite(s.name);
    try {
      await s.fn(ctx, R);
    } catch (e) {
      R.check(s.name + " suite ran to completion", false, (e && e.stack) || String(e));
    }
    await ctx.page.close().catch(() => {});
    await ctx.rawContext.close().catch(() => {});
  }

  await browser.close();
  server.close();

  ALL_RECORDS.push(...R.records);

  const fails = R.failures;
  console.log("\n[" + engineName + "] " + (R.total - fails.length) + "/" + R.total + " checks passed");
  for (const f of fails) console.log("   FAILED: " + f);
  console.log("[" + engineName + "] " + (fails.length ? "FAIL" : "PASS"));
  return fails.length;
}

(async () => {
  const engArg = (process.argv.find((a) => a.startsWith("--engine=")) || "--engine=both").split("=")[1];
  const only = (process.argv.find((a) => a.startsWith("--suites=")) || "").split("=")[1];
  const jobs = only ? SUITES.filter((s) => only.split(",").some((k) => s.name.includes(k))) : SUITES;

  const engines = engArg === "both" ? ["webkit", "chromium"] : [engArg];
  let bad = 0;
  for (const e of engines) bad += await runEngine(e, jobs);

  const jsonArg = process.argv.find((a) => a.startsWith("--json="));
  if (jsonArg) {
    const out = jsonArg.split("=").slice(1).join("=");
    fs.mkdirSync(path.dirname(out), { recursive: true });
    fs.writeFileSync(out, JSON.stringify(ALL_RECORDS, null, 2) + "\n");
    console.log("\n[json] " + ALL_RECORDS.length + " record(s) -> " + out);
  }
  process.exit(bad ? 1 : 0);
})().catch((e) => { console.error("runner failed:", e); process.exit(2); });
