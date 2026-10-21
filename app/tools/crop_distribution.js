/**
 * crop_distribution.js — WHAT THE APP ACTUALLY FEEDS THE RECOGNISER.
 *
 * THE QUESTION
 * ------------
 * The same checkpoint scores 57.0% exact line match on held-out CORPUS crops
 * (ml/src/build_dataset.crop_to_array) and ~17-22% word accuracy on the app's 12
 * bench documents. This project has been bitten three times by a train/serve
 * geometry mismatch (the served width, the padding, the over-harsh capture
 * profiles), each of which moved more than any model work. So: measure the
 * app's crop distribution against the training one.
 *
 * This half drives the APP'S OWN code — app/js/pipeline.js preprocess() (loaded
 * as the app loads it) and app/js/layout.js detectLines() + cropForRecognizer()
 * — over every sample in the bench manifest, and dumps per crop:
 *
 *   BEFORE  the detector's box (bw x bh, the page pixels the line occupies)
 *   AFTER   what cropForRecognizer returns = what recogniseAll() is handed
 *           (nw x CROP_H), i.e. the app's crop_to_array
 *   STATS   ink density / ink-paper contrast / Laplacian variance, computed the
 *           same way real_degradation_cache.crop_stats does (Otsu split on the
 *           crop itself, ink = a <= thr, Laplacian variance)
 *
 * It prints the app-side width/height/statistics table, and writes the same
 * numbers as JSON (--json=<path>) for ml/src/dist_compare.py, which loads the
 * training corpus, prints the side-by-side table, the fraction of app crops
 * inside the training width range, and the verdict.
 *
 * WHICH CANDIDATE. preprocess() offers three arms (raw deskewed grayscale, CLAHE
 * = pp.canvas, CLAHE+Otsu) and runOCR ranks them by structure_score, which needs
 * the model. Both the raw and the enhanced (pp.canvas) arms are therefore
 * measured here and dumped; the report uses pp.canvas, matching the app's own
 * geometry suite in ios_full_check.js.
 *
 * Run:  node app/tools/crop_distribution.js --json=/tmp/cropdist_app.json [--engine=webkit|chromium]
 * Exit 0 when every sample produced at least one crop.
 */
"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");

const APP = path.resolve(__dirname, "..");
const ROOT = path.resolve(APP, "..");
const SAMPLES = process.env.CROP_SAMPLES || "/tmp/sightline_samples";
const MANIFEST = path.join(SAMPLES, "manifest.json");
const TIMEOUT_MS = 180000;

// The app's serving constants. maxDim 3200 is app.js/pipeline's scan() default
// (the 1600px long edge in docs/01_ARCHITECTURE.md is the phone-photo downscale
// doc, not what this code calls); the samples are 900px wide, so preprocess()
// leaves them at full size and the number actually served is reported per page.
const MAX_DIM = 3200;
const MAX_W = 320;          // recognizer.EXPORT_W === realdata.MAX_W, passed by ocr_engine
const CROP_H = 32;

const pw = require("./pw_core.js")();
const chromeBin = require("./chrome_path.js")();

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm", ".onnx": "application/octet-stream",
  ".png": "image/png", ".css": "text/css",
};

// pipeline.js is a CLASSIC script (it publishes globalThis.SightLine), so it is
// loaded with a <script> tag the way index.html loads it. Nothing else boots:
// no service worker, no model, no scan path.
const BLANK = "<!doctype html><meta charset=utf-8><title>crop distribution</title>"
  + "<script src=\"/js/pipeline.js\"></script>";

/** Static server: app/ at the root, the bench samples under /samples/. */
function serve() {
  const server = http.createServer((req, res) => {
    const p = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (p === "/__cropdist.html") {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      return res.end(BLANK);
    }
    let file;
    if (p.startsWith("/samples/")) {
      file = path.join(SAMPLES, p.slice("/samples/".length));
      if (!file.startsWith(SAMPLES)) {
        res.writeHead(403, { "Content-Type": "text/plain" });
        return res.end("no");
      }
    } else {
      file = path.join(APP, p);
      if (!file.startsWith(APP)) {
        res.writeHead(403, { "Content-Type": "text/plain" });
        return res.end("no");
      }
    }
    if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      return res.end("not found: " + p);
    }
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream",
                         "Cache-Control": "no-store" });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((r) => server.listen(0, "127.0.0.1", () => r(server)));
}

/**
 * Everything that runs INSIDE the page. Self-contained by construction:
 * Playwright serialises the source, so no outer-scope reference is allowed.
 */
async function inPage(cfg) {
  // ── crop_stats, ported from real_degradation_cache.crop_stats ────────────
  function otsuThresh(a) {
    const hist = new Uint32Array(256);
    for (let i = 0; i < a.length; i++) hist[a[i]]++;
    const total = a.length;
    let sum = 0;
    for (let i = 0; i < 256; i++) sum += i * hist[i];
    let sumB = 0, wB = 0, best = 0, thr = 0;
    for (let i = 0; i < 256; i++) {
      wB += hist[i];
      if (!wB) continue;
      const wF = total - wB;
      if (!wF) break;
      sumB += i * hist[i];
      const mB = sumB / wB, mF = (sum - sumB) / wF;
      const between = wB * wF * (mB - mF) * (mB - mF);
      if (between > best) { best = between; thr = i; }
    }
    return thr;
  }

  // cv2.Laplacian(a, CV_64F) with the default 3x3 kernel and BORDER_REFLECT_101.
  const refl = (i, n) => (i < 0 ? -i : (i >= n ? 2 * n - i - 2 : i));
  function laplacianVar(a, w, h) {
    let s = 0, s2 = 0;
    const n = w * h;
    for (let y = 0; y < h; y++) {
      const ym = refl(y - 1, h), yp = refl(y + 1, h);
      for (let x = 0; x < w; x++) {
        const xm = refl(x - 1, w), xp = refl(x + 1, w);
        const v = a[ym * w + x] + a[yp * w + x] + a[y * w + xm] + a[y * w + xp]
                  - 4 * a[y * w + x];
        s += v; s2 += v * v;
      }
    }
    const m = s / n;
    return s2 / n - m * m;
  }

  function cropStats(a, w, h) {
    const thr = otsuThresh(a);                  // ink = a <= thr (Otsu lands on the ink mode)
    let nInk = 0, sumInk = 0, nPaper = 0, sumPaper = 0;
    for (let i = 0; i < a.length; i++) {
      if (a[i] <= thr) { nInk++; sumInk += a[i]; } else { nPaper++; sumPaper += a[i]; }
    }
    const density = nInk / a.length;
    const contrast = (nInk && nPaper) ? (sumPaper / nPaper) - (sumInk / nInk) : 0;
    return { ink: density, contrast, sharp: laplacianVar(a, w, h) };
  }

  // ocr_engine.toGray's Rec.601 integer luma — the app's own conversion, so the
  // crop pixels here are the pixels the recogniser is handed.
  function luma(cv) {
    const w = cv.width, h = cv.height;
    const d = cv.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h).data;
    const g = new Uint8Array(w * h);
    for (let i = 0, p = 0; i < g.length; i++, p += 4) {
      g[i] = (d[p] * 77 + d[p + 1] * 150 + d[p + 2] * 29) >> 8;
    }
    return { gray: g, w, h };
  }

  try {
    const L = await import(cfg.layoutUrl);
    const pp0 = globalThis.SightLine;
    if (!pp0) throw new Error("pipeline.js did not publish SightLine");
    const out = [];
    for (const s of cfg.samples) {
      const img = new Image();
      img.src = "/samples/" + s.file.split("/").pop();
      await img.decode();
      const c = document.createElement("canvas");
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext("2d", { willReadFrequently: true }).drawImage(img, 0, 0);
      const pp = pp0.preprocess(c, cfg.maxDim, { restoreMode: "adaptive" });
      const rec = { file: s.file, doc: s.doc, profile: s.profile,
                    src: img.naturalWidth + "x" + img.naturalHeight,
                    page: pp.width + "x" + pp.height, arms: pp.arms, cand: {} };
      const arms = { raw: pp.candidates[0], enhanced: pp.candidates[1] };
      for (const name of ["raw", "enhanced"]) {
        const cv = arms[name];
        const { gray, w, h } = luma(cv);
        const boxes = L.detectLines(gray, w, h);
        const crops = [];
        for (const b of boxes) {
          const bw = b.x1 - b.x0, bh = b.y1 - b.y0;
          const cr = L.cropForRecognizer(gray, w, b, cfg.cropH, cfg.maxW);
          const nw = cr.width, nh = cr.data.length / nw;
          let filled = 0;
          for (let y = 0; y < nh; y++) {
            for (let x = 0; x < nw; x++) if (cr.data[y * nw + x] !== 0) { filled++; break; }
          }
          const st = cropStats(cr.data, nw, nh);
          crops.push({ bw, bh, nw, nh, filled, ink: st.ink,
                       contrast: st.contrast, sharp: st.sharp });
        }
        rec.cand[name] = { size: cv.width + "x" + cv.height, lines: boxes.length, crops };
      }
      out.push(rec);
    }
    return { ok: true, maxW: cfg.maxW, cropH: cfg.cropH, maxDim: cfg.maxDim, samples: out };
  } catch (e) {
    const msg = (e && e.name ? e.name + ": " : "") + (e && e.message ? e.message : String(e));
    const st = e && e.stack ? String(e.stack).split(String.fromCharCode(10)).slice(0, 4).join(" | ") : null;
    return { ok: false, error: msg, stack: st };
  }
}

async function runEngine(engine, url, cfg) {
  const report = { engine, refused: false, error: null };
  let browser = null;
  try {
    browser = engine === "webkit" ? await pw.webkit.launch()
            : await pw.chromium.launch({ executablePath: chromeBin });
    const page = await browser.newPage();
    page.setDefaultTimeout(TIMEOUT_MS);
    await page.goto(url, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => !!globalThis.SightLine, null, { timeout: 30000 });
    const res = await page.evaluate(inPage, cfg);
    if (!res) { report.refused = true; report.error = "page.evaluate returned undefined"; }
    else if (!res.ok) { report.refused = true; report.error = res.error; report.stack = res.stack; }
    else { report.data = res; }
  } catch (e) {
    report.refused = true;
    report.error = (e && e.message ? e.message : String(e));
  } finally {
    if (browser) { try { await browser.close(); } catch (e) { /* gone */ } }
  }
  return report;
}

// ── reporting ───────────────────────────────────────────────────────────────

const q = (xs, p) => {
  if (!xs.length) return NaN;
  const s = xs.slice().sort((a, b) => a - b);
  const i = (s.length - 1) * p;
  const lo = Math.floor(i), hi = Math.ceil(i);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (i - lo);
};
const med = (xs) => q(xs, 0.5);
const mean = (xs) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
const f2 = (v) => (Number.isFinite(v) ? v.toFixed(2) : "-");
const f3 = (v) => (Number.isFinite(v) ? v.toFixed(3) : "-");

/** Widths of one sample's crops for one candidate arm. */
function widths(rec, arm) {
  return (rec.cand[arm] || { crops: [] }).crops.map((c) => c.nw);
}

function printSide(rec, arm) {
  const crops = rec.cand[arm].crops;
  const ws = crops.map((c) => c.nw), hs = crops.map((c) => c.bh);
  const at = ws.filter((w) => w >= MAX_W).length;
  console.log("  " + rec.doc + "/" + rec.profile + "  page " + rec.page
    + "  (" + rec.src + " source)  lines " + crops.length);
  console.log("    width nw   min " + (Math.min.apply(null, ws) || 0) + "  p25 " + f2(q(ws, 0.25))
    + "  med " + f2(med(ws)) + "  p75 " + f2(q(ws, 0.75)) + "  max " + Math.max.apply(null, ws)
    + "   at " + MAX_W + ": " + at + "/" + ws.length
    + "   narrow (<16): " + ws.filter((w) => w < 16).length);
  console.log("    box bh     min " + (Math.min.apply(null, hs) || 0) + "  med " + f2(med(hs))
    + "  max " + Math.max.apply(null, hs) + "     crops 32 tall: "
    + crops.filter((c) => c.filled === CROP_H && c.nh === CROP_H).length + "/" + crops.length);
  console.log("    ink " + f3(med(crops.map((c) => c.ink)))
    + "   contrast " + f2(med(crops.map((c) => c.contrast)))
    + "   sharp(Laplacian var) " + f2(med(crops.map((c) => c.sharp))));
}

function summarise(label, recs, arm) {
  const crops = [];
  for (const r of recs) for (const c of r.cand[arm].crops) crops.push(c);
  const ws = crops.map((c) => c.nw), hs = crops.map((c) => c.bh);
  const at = ws.filter((w) => w >= MAX_W).length;
  console.log("  " + label.padEnd(22) + " crops " + String(crops.length).padStart(4)
    + "  nw min " + Math.min.apply(null, ws) + " p25 " + f2(q(ws, 0.25)) + " med " + f2(med(ws))
    + " p75 " + f2(q(ws, 0.75)) + " max " + Math.max.apply(null, ws)
    + "  ==320 " + (100 * at / crops.length).toFixed(1) + "%"
    + "  <320 " + (100 * (crops.length - at) / crops.length).toFixed(1) + "%"
    + "  narrow(<16) " + ws.filter((w) => w < 16).length);
  console.log("  " + " ".repeat(22) + " bh min " + Math.min.apply(null, hs) + " med " + f2(med(hs))
    + " max " + Math.max.apply(null, hs)
    + "  | ink med " + f3(med(crops.map((c) => c.ink))) + " mean " + f3(mean(crops.map((c) => c.ink)))
    + " | contrast med " + f2(med(crops.map((c) => c.contrast)))
    + " | sharp med " + f2(med(crops.map((c) => c.sharp))));
  return { crops, ws, hs };
}

(async () => {
  if (!fs.existsSync(MANIFEST)) {
    console.error("missing " + MANIFEST + " — the bench samples are not where this expects them");
    process.exit(2);
  }
  const manifest = JSON.parse(fs.readFileSync(MANIFEST, "utf8"));
  const samples = manifest.samples || [];
  const arg = (process.argv.find((a) => a.startsWith("--engine=")) || "").split("=")[1];
  const engines = arg === "webkit" || arg === "chromium" ? [arg] : ["webkit"];   // webkit = iOS Safari
  const jsonPath = (process.argv.find((a) => a.startsWith("--json=")) || "").split("=")[1];

  const server = await serve();
  const base = "http://127.0.0.1:" + server.address().port + "/";
  const cfg = { layoutUrl: base + "js/layout.js", maxDim: MAX_DIM, maxW: MAX_W,
                cropH: CROP_H, samples };
  let bad = 0;
  const dumped = {};
  for (const e of engines) {
    const r = await runEngine(e, base + "__cropdist.html", cfg);
    dumped[e] = r;
    console.log("\n── " + e + " ─────────────────────────────────────────────────────");
    if (r.refused) {
      console.log("  CROP DISTRIBUTION DUMP REFUSED: " + r.error);
      if (r.stack) console.log("  at: " + r.stack);
      bad++;
      continue;
    }
    const recs = r.data.samples;
    console.log("  constants: maxDim " + r.data.maxDim + " MAX_W/EXPORT_W " + r.data.maxW
      + " CROP_H " + r.data.cropH + "  samples " + recs.length);
    console.log("\n  WHAT THE APP FEEDS THE RECOGNISER (pp.canvas = the CLAHE arm, per document):");
    for (const rec of recs) printSide(rec, "enhanced");
    console.log("\n  APP TOTALS");
    const en = summarise("enhanced (served)", recs, "enhanced");
    summarise("raw (arm 0)", recs, "raw");
    const noCrop = recs.filter((x) => x.cand.enhanced.crops.length === 0);
    if (noCrop.length) {
      console.log("  NO CROPS DETECTED on: " + noCrop.map((x) => x.doc + "/" + x.profile).join(", "));
      bad++;
    }
    if (jsonPath) {
      fs.writeFileSync(jsonPath, JSON.stringify({
        engine: e, maxW: r.data.maxW, cropH: r.data.cropH, maxDim: r.data.maxDim,
        samples: recs.map((x) => ({ doc: x.doc, profile: x.profile, src: x.src, page: x.page,
                                    lines: x.cand.enhanced.lines, crops: x.cand.enhanced.crops,
                                    raw: x.cand.raw.crops })),
      }, null, 1) + "\n");
      console.log("\n  wrote " + jsonPath);
    }
    // A dump with no crops anywhere measured nothing and must not read as a pass.
    const total = recs.reduce((s, x) => s + x.cand.enhanced.crops.length, 0);
    if (!total) { console.log("  NO CROPS AT ALL — nothing was measured"); bad++; }
    else {
      console.log("  " + total + " crops measured on " + recs.length + " documents"
        + "  (enhanced median nw " + f2(med(en.ws)) + ", ==320 "
        + (100 * en.ws.filter((w) => w >= MAX_W).length / en.ws.length).toFixed(1) + "%)");
    }
  }
  server.close();
  console.log("\n" + (bad ? "CROP DISTRIBUTION DUMP INCOMPLETE (" + bad + " problem(s))"
                         : "CROP DISTRIBUTION DUMP OK"));
  process.exit(bad ? 1 : 0);
})();
