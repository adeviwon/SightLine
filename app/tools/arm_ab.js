/**
 * arm_ab.js — WHICH PREPROCESSING ARM DOES THE RECOGNISER ACTUALLY READ BEST,
 * ON THE APP'S OWN DOCUMENTS, WITH THE CROP SET HELD FIXED?
 *
 * THE GAP THIS MEASURES
 * ---------------------
 * Paired on 300 held-out crops from the CORPUS the model was trained on, the
 * recogniser scores 57.0% exact line match / CER 0.156. On the app's own 12
 * rendered documents it reads at roughly 17-22% word accuracy. The two
 * populations differ in exactly one known way: the app PREPROCESSES. The corpus
 * is RAW rendered crops; the app restores / CLAHEs / binarises the page before
 * the detector and recogniser see it (js/pipeline.js preprocess -> candidates;
 * js/restorer.js armRestorerClahe -> the shipped candidate set).
 *
 * THE DESIGN: CROP SET FIXED, ONLY THE ARM VARIES
 * ----------------------------------------------
 * Every earlier arm comparison (tools/restore_ab.js, tools/e2e_ocr.js) ran the
 * app's OWN pipeline end to end per arm, and the pipeline re-DETECTS lines on
 * whichever image it is given. So a per-arm win could come from a better
 * DETECTION or from a better READ, and neither number can be attributed.
 *
 * Here:
 *   1. detect lines ONCE per document, on the RAW deskewed grayscale, with the
 *      app's own detector (js/layout.js detectLines) -> a frozen box list.
 *   2. for EVERY arm, crop the SAME boxes from THAT arm's full-page image with
 *      the app's own js/layout.js cropForRecognizer, and recognise with the
 *      app's own js/recognizer.js recogniseAll.
 * Same boxes, same model (models/crnn.onnx), same geometry — only the page
 * preprocessing that produced the pixels under each box differs. Paired.
 *
 * ARMS (each is a full-page transform of the SAME deskewed image)
 *   raw            js/pipeline.js preprocess candidate 0 — deskewed grayscale
 *   tile_clahe     candidate 1 — clahe(deskewed, 3.0, 8x8)
 *   clahe_otsu     candidate 2 — binarize(tile_clahe)
 *   restored       the SHIPPED app arm (js/restorer.js armRestorerClahe): the
 *                  restorer's output — ONNX, gate-skipped, or classical
 *   restored_clahe the shipped arm candidate 1 — clahe(restored, 2.0, 8)
 *   restored_otsu  the shipped arm candidate 2 — binarize(unsharp(clahe))
 * The shipped ranking (js/ocr_engine.js runOCR: structure_score desc, then
 * confidence, first wins ties) chooses among the last three — that is the
 * candidate list app.js hands it. Its pick is marked in the table.
 *
 * GROUND TRUTH comes from ml/src/capture.py DOCS (the lines the renderer DREW),
 * not from a retyped copy — render_samples.py warns that a hand-copied ground
 * truth is one typo away from scoring the wrong thing. Lines are paired to
 * detected bands BY ORDER (the renderer draws them top to bottom); bands/line is
 * printed so the reader can see how clean the pairing is.
 *
 * Run:  node app/tools/arm_ab.js --engine=webkit
 *       node app/tools/arm_ab.js --engine=chromium
 */
"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const pw = require("./pw_core.js")();
const chromeBin = require("./chrome_path.js")();

const APP = path.resolve(__dirname, "..");
const ROOT = path.resolve(APP, "..");
const SAMPLES = "/tmp/sightline_samples";
const MAX_DIM = 3200;                   // app.js's runScan() default: no downscale here

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm", ".onnx": "application/octet-stream",
  ".png": "image/png", ".css": "text/css",
};

// ── ground truth: the lines the renderer actually drew ───────────────────────
const GT_CACHE = path.join(SAMPLES, "gt_lines.json");

function groundTruthLines() {
  // Cache first: the drawn lines never change, and spawning python on every run
  // is wasteful (and, under CPU contention, fragile).
  try {
    const c = JSON.parse(fs.readFileSync(GT_CACHE, "utf8"));
    if (c && c.prescription) return { lines: c, source: "ml/src/capture.py DOCS (cached)" };
  } catch (e) { /* not cached yet */ }

  const code = "import sys,json;sys.path.insert(0,'ml/src');import capture;"
    + "print(json.dumps({d:list(lines) for lines,d in capture.DOCS}))";
  let why = null;
  for (const bin of ["python3", "python"]) {
    const dumped = spawnSync(bin, ["-c", code], { cwd: ROOT, encoding: "utf8", maxBuffer: 8 << 20 });
    if (dumped.status === 0 && dumped.stdout) {
      try {
        const lines = JSON.parse(dumped.stdout);
        try { fs.writeFileSync(GT_CACHE, JSON.stringify(lines)); } catch (e) { /* read-only fs */ }
        return { lines, source: "ml/src/capture.py DOCS" };
      } catch (e) { why = bin + ": JSON.parse: " + e.message; }
    } else {
      why = bin + ": status " + dumped.status + " err " + ((dumped.error && dumped.error.message)
        || (dumped.stderr || "").slice(0, 200));
    }
  }
  // Degraded: the manifest's joined text is one line, not the drawn lines.
  const m = JSON.parse(fs.readFileSync(path.join(SAMPLES, "manifest.json"), "utf8"));
  const lines = {};
  for (const d of Object.keys(m.text || {})) lines[d] = [m.text[d]];
  return { lines, source: "manifest.text (DEGRADED: " + why + ")", warn: true };
}

function serve() {
  const srv = http.createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split("?")[0]);
    let p;
    if (rel.startsWith("/sample/")) p = path.join(SAMPLES, rel.slice("/sample/".length));
    else p = path.join(APP, rel);
    if (!p.startsWith(APP) && !p.startsWith(SAMPLES)) { res.writeHead(403); return res.end(); }
    fs.readFile(p, (e, d) => {
      if (e) { res.writeHead(404); return res.end("not found: " + rel); }
      res.writeHead(200, { "Content-Type": MIME[path.extname(p)] || "application/octet-stream",
                           "Cache-Control": "no-store" });
      res.end(d);
    });
  });
  return new Promise((r) => srv.listen(0, "127.0.0.1", () => r(srv)));
}

// ── everything that runs INSIDE the page ─────────────────────────────────────
// A real function (Playwright serialises the source, so it closes over nothing;
// all config arrives via cfg).
async function inPage(cfg) {
  try {
    const L = await import(cfg.base + "js/layout.js");
    const R = await import(cfg.base + "js/recognizer.js");
    if (typeof SightLine === "undefined") return { ok: false, error: "SightLine not on the page" };
    await R.init();

    // Rec. 601 integer luma — the same weighting ocr_engine.js toGray uses, so
    // the detector/recogniser see the pixels the app would feed them.
    const toGray = (c) => {
      const w = c.width, h = c.height;
      const d = c.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h).data;
      const g = new Uint8Array(w * h);
      for (let i = 0, p = 0; i < g.length; i++, p += 4) {
        g[i] = (d[p] * 77 + d[p + 1] * 150 + d[p + 2] * 29) >> 8;
      }
      return { gray: g, w, h };
    };

    const docs = [];
    for (const s of cfg.samples) {
      const img = new Image();
      img.src = "/sample/" + s.name;
      await img.decode();

      // restoreMode "none": raw is the pure deskewed grayscale the corpus
      // resembles; every other arm is an explicit transform of it.
      const pp = SightLine.preprocess(img, cfg.maxDim, { restoreMode: "none" });
      const raw = pp.candidates[0], tileClahe = pp.candidates[1], claheOtsu = pp.candidates[2];

      // The SHIPPED app candidate set: armRestorerClahe(pp.canvas) —
      // exactly what app.js hands to runOCR.
      let rr = null;
      try { rr = await SightLineRestorer.armRestorerClahe(pp.canvas, {}); } catch (e) { rr = null; }
      const rc = (rr && rr.candidates) || [];

      const arms = [
        ["raw", raw],
        ["tile_clahe", tileClahe],
        ["clahe_otsu", claheOtsu],
        ["restored", rc[0]],
        ["restored_clahe", rc[1]],
        ["restored_otsu", rc[2]],
      ].filter((a) => a[1]);

      // ── detect ONCE, on the raw deskewed grayscale: the frozen crop set ──
      const ref = toGray(raw);
      const boxes = L.detectLines(ref.gray, ref.w, ref.h)
        .slice().sort((a, b) => a.y0 - b.y0);

      const textsByArm = {};
      for (const [name, canvas] of arms) {
        const g = toGray(canvas);
        if (g.w !== ref.w || g.h !== ref.h) {
          textsByArm[name] = { error: "size " + g.w + "x" + g.h + " != " + ref.w + "x" + ref.h };
          continue;
        }
        const crops = boxes.map((box) => {
          const c = L.cropForRecognizer(g.gray, g.w, box, R.CROP_H, R.EXPORT_W);
          return { gray: c.data, width: c.width };   // cropForRecognizer -> recogniseAll
        });
        const out = await R.recogniseAll(crops);
        const joined = out.map((r) => r.text).join(" ");
        textsByArm[name] = {
          texts: out.map((r) => r.text), conf: out.map((r) => r.confidence),
          // The ranking's OWN objective and the product's OWN metric, on this
          // arm's full document text — so the table shows whether the shipped
          // pick was chosen for structure while losing fields, or neither.
          structure: SightLine.structure_score(joined),
          fields: SightLine.fields_found(SightLine.normalizeOCRText(joined), s.doc),
        };
      }

      // ── the SHIPPED ranking's pick, from the real code ──────────────────
      let pick = null;
      try {
        const o = await SightLine.runOCR(rc.length ? rc : [raw, tileClahe, claheOtsu], () => {});
        pick = { psmRank: o.psmRank, structure: o.structure, confidence: o.confidence,
                 chars: (o.text || "").length };
      } catch (e) { pick = { error: String((e && e.message) || e) }; }

      docs.push({
        name: s.name, doc: s.doc, profile: s.profile,
        dims: pp.width + "x" + pp.height,
        bands: boxes.length,
        fieldSlots: ((SightLine.FIELDS && SightLine.FIELDS[s.doc]) || []).length,
        restorer: rr ? { engine: rr.engine, weight: rr.weight, error: rr.error || null } : null,
        armOrder: arms.map((a) => a[0]),
        textsByArm, pick,
      });
    }
    return { ok: true, docs };
  } catch (e) {
    const msg = (e && e.name ? e.name + ": " : "") + (e && e.message ? e.message : String(e));
    return { ok: false, error: msg, stack: e && e.stack ? String(e.stack).split("\n").slice(0, 4).join(" | ") : null };
  }
}

// ── metrics (Node side, on the returned crop texts) ──────────────────────────
// norm() mirrors ml/src/evaluate.py:norm — strip non-alphanumeric, lowercase.
const norm = (s) => String(s == null ? "" : s).toLowerCase().replace(/[^a-z0-9]/g, "");

function lev(a, b) {                             // Levenshtein / CER edit distance
  const n = a.length, m = b.length;
  if (!n) return m;
  if (!m) return n;
  let prev = new Array(m + 1), cur = new Array(m + 1);
  for (let j = 0; j <= m; j++) prev[j] = j;
  for (let i = 1; i <= n; i++) {
    cur[0] = i;
    const ai = a.charCodeAt(i - 1);
    for (let j = 1; j <= m; j++) {
      const cost = ai === b.charCodeAt(j - 1) ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    const t = prev; prev = cur; cur = t;
  }
  return prev[m];
}

const words = (s) => String(s || "").split(/\s+/).filter(Boolean).map(norm).filter(Boolean);
function wordAcc(gt, hyp) {
  const g = words(gt);
  if (!g.length) return 0;
  const h = new Set(words(hyp));
  return g.filter((w) => h.has(w)).length / g.length;
}

const ARMS = ["raw", "tile_clahe", "clahe_otsu", "restored", "restored_clahe", "restored_otsu"];
const SHIPPED = { 0: "restored", 1: "restored_clahe", 2: "restored_otsu" };

function main(docs) {
  const acc = {};
  for (const a of ARMS) acc[a] = { crops: 0, exact: 0, gtl: 0, ed: 0, gtWords: 0, wordsHit: 0,
                                   refChars: 0, docEd: 0, struct: 0, fields: 0, slots: 0 };
  const picks = [];
  const lines = [];

  for (const d of docs) {
    const gt = (d.gtLines || []);
    const n = Math.min(d.bands, gt.length);
    const perArm = [];
    for (const a of ARMS) {
      const ta = d.textsByArm[a];
      if (!ta || ta.error || !ta.texts) { perArm.push({ a, skip: ta && ta.error }); continue; }
      let exact = 0, gtl = 0, ed = 0;
      for (let i = 0; i < n; i++) {
        const hyp = norm(ta.texts[i]), ref = norm(gt[i]);
        gtl += ref.length;
        ed += lev(hyp, ref);
        if (hyp === ref && ref.length) exact++;
      }
      const hypDoc = ta.texts.join(" ");
      const refDoc = gt.join(" ");
      const refN = norm(refDoc);
      const docEd = lev(norm(hypDoc), refN);
      perArm.push({ a, exact, n, gtl, ed, docEd, refChars: refN.length,
                    wa: wordAcc(refDoc, hypDoc), gtWords: words(refDoc).length });
      const A = acc[a];
      A.crops += n; A.exact += exact; A.gtl += gtl; A.ed += ed;
      A.refChars += refN.length; A.docEd += docEd;
      A.gtWords += words(refDoc).length;
      A.wordsHit += wordAcc(refDoc, hypDoc) * words(refDoc).length;
      A.struct += ta.structure || 0;
      A.fields += (ta.fields || []).length;
      A.slots += (d.fieldSlots || 0);
    }
    const pickArm = d.pick && typeof d.pick.psmRank === "number" ? SHIPPED[d.pick.psmRank] : null;
    picks.push({ doc: d.name, pickArm, pick: d.pick });
    lines.push({ d, gt, n, perArm, pickArm });
  }

  // ── the arm table ─────────────────────────────────────────────────────────
  const pct = (x) => (100 * x).toFixed(1).padStart(5) + "%";
  console.log("\n=== ARM TABLE  (same detected crops for every arm; paired) ===");
  console.log("arm".padEnd(16) + "exact".padStart(7) + "  lineCER".padStart(8)
    + "  wordAcc".padStart(8) + "  fields".padStart(8) + "  struct".padStart(7) + "  crops");
  console.log("-".repeat(68));
  const pickCount = {};
  const score = {};
  for (const a of ARMS) {
    const A = acc[a];
    if (!A.crops) { console.log(a.padEnd(16) + "  (not present on this page set)"); continue; }
    const exactRate = A.exact / A.crops;
    const lineCer = A.ed / Math.max(1, A.gtl);
    const wa = A.wordsHit / Math.max(1, A.gtWords);
    const docCer = A.docEd / Math.max(1, A.refChars);
    score[a] = { exactRate, lineCer, wa, docCer, crops: A.crops,
                 fields: A.fields, slots: A.slots, struct: A.struct };
    const isShipped = Object.values(SHIPPED).includes(a);
    console.log(a.padEnd(16) + pct(exactRate) + "  " + lineCer.toFixed(3).padStart(6)
      + "  " + pct(wa) + "  " + (A.fields + "/" + A.slots).padStart(8)
      + "  " + String(A.struct).padStart(5)
      + "  " + String(A.crops).padStart(4) + (isShipped ? "   [shipped candidate]" : ""));
  }
  for (const p of picks) if (p.pickArm) pickCount[p.pickArm] = (pickCount[p.pickArm] || 0) + 1;

  // ── the two direct answers ────────────────────────────────────────────────
  const scored = Object.keys(score);
  const bestExact = scored.slice().sort((x, y) => score[y].exactRate - score[x].exactRate)[0];
  const bestCer = scored.slice().sort((x, y) => score[x].lineCer - score[y].lineCer)[0];
  const best = score[bestExact].exactRate >= score[bestCer].exactRate ? bestExact : bestCer;
  const mostPicked = Object.keys(pickCount).sort((x, y) => pickCount[y] - pickCount[x])[0] || null;

  console.log("\n=== DIRECT ANSWERS ===");
  console.log("(a) best arm on the app's OWN documents:");
  console.log("    highest exact  : " + bestExact + "  " + pct(score[bestExact].exactRate)
    + "  lineCER " + score[bestExact].lineCer.toFixed(3));
  console.log("    lowest  lineCER: " + bestCer + "  " + pct(score[bestCer].exactRate)
    + "  lineCER " + score[bestCer].lineCer.toFixed(3));
  console.log("(b) the shipped ranking picks: " + (mostPicked || "nothing")
    + "  (" + (pickCount[mostPicked] || 0) + "/" + docs.length + " documents)"
    + (Object.keys(pickCount).length > 1 ? "  distribution " + JSON.stringify(pickCount) : ""));
  if (mostPicked && score[mostPicked] && mostPicked !== best) {
    const dEx = 100 * (score[best].exactRate - score[mostPicked].exactRate);
    const dCer = score[mostPicked].lineCer - score[best].lineCer;
    console.log("    -> MISMATCH: the ranking is costing " + dEx.toFixed(1)
      + " pts exact and " + dCer.toFixed(3) + " lineCER versus " + best);
    console.log("       on the product metrics too: fields "
      + score[best].fields + "/" + score[best].slots + " (" + best + ") vs "
      + score[mostPicked].fields + "/" + score[mostPicked].slots + " (" + mostPicked + ");"
      + " structure tokens " + score[best].struct + " vs " + score[mostPicked].struct
      + " (the ranking's own objective)");
  } else if (mostPicked) {
    console.log("    -> the ranking picks the best-measured arm; preprocessing choice is not the gap.");
  }

  console.log("\n=== PER-DOCUMENT PAIRING (bands vs ground-truth lines) ===");
  for (const { d, gt, n, perArm, pickArm } of lines) {
    const ex = {};
    for (const p of perArm) if (!p.skip) ex[p.a] = (100 * p.exact / Math.max(1, p.n)).toFixed(0) + "%";
    console.log(d.name.replace(/\.png$/, "").padEnd(30)
      + " bands " + String(d.bands).padStart(2) + " / gt " + String(gt.length).padStart(2)
      + "  restored=" + String((d.restorer && d.restorer.engine) || "n/a").padEnd(13)
      + " pick=" + String(pickArm).padEnd(15)
      + " exact " + ARMS.filter((a) => ex[a]).map((a) => a.slice(0, 4) + ":" + ex[a]).join(" "));
  }
  console.log("\nground truth source: " + GT_source
    + (GT_warn ? "   *** DEGRADED — line pairing is unreliable ***" : ""));
  return { score, best, bestExact, bestCer, mostPicked, pickCount };
}

let GT_source = "", GT_warn = false;

(async () => {
  const arg = (process.argv.find((a) => a.startsWith("--engine=")) || "").split("=")[1];
  const engine = arg === "chromium" ? "chromium" : "webkit";

  const manifestPath = path.join(SAMPLES, "manifest.json");
  if (!fs.existsSync(manifestPath)) {
    console.log("SKIP: no sample manifest at " + manifestPath + " (run: bash run.sh samples)");
    process.exit(0);
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const gt = groundTruthLines();
  GT_source = gt.source; GT_warn = !!gt.warn;
  const samples = (manifest.samples || []).filter((s) => fs.existsSync(s.file));
  console.log("engine=" + engine + "  fixtures=" + samples.length
    + "  model=" + path.join(APP, "models", "crnn.onnx")
    + (fs.existsSync(path.join(APP, "models", "restorer.onnx")) ? "  restorer.onnx=present" : "  restorer.onnx=ABSENT"));

  const server = await serve();
  const base = "http://127.0.0.1:" + server.address().port + "/";
  let browser = null;
  try {
    browser = engine === "chromium"
      ? await pw.chromium.launch({ executablePath: chromeBin })
      : await pw.webkit.launch();
    const page = await browser.newPage();
    page.setDefaultTimeout(600000);
    await page.goto(base + "index.html", { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(5000);

    const cfg = {
      base, maxDim: MAX_DIM,
      samples: samples.map((s) => ({ name: path.basename(s.file), doc: s.doc, profile: s.profile })),
    };
    const res = await page.evaluate(inPage, cfg);
    if (!res || !res.ok) {
      console.log("PAGE ERROR: " + ((res && (res.stack || res.error)) || "evaluate returned undefined"));
      process.exitCode = 1;
      return;
    }
    for (const d of res.docs) d.gtLines = (gt.lines[d.doc] || []);
    const out = main(res.docs);

    console.log("\n=== INTEGRATION (prose; no source file is modified by this tool) ===");
    if (out.mostPicked && out.best && out.mostPicked !== out.best) {
      console.log("A non-shipped arm wins by a meaningful margin. The change is in the");
      console.log("candidate set, not the ranking rule:");
      console.log("  app/js/pipeline.js preprocess(): `const candidates = [raw, enhanced, binary];`");
      console.log("  app/js/app.js runScan(): the shipped candidate set is");
      console.log("  `SightLineRestorer.armRestorerClahe(pp.canvas).candidates`.");
      console.log("  Reorder/trim so the winning arm (" + out.best + ") is the one the");
      console.log("  recogniser is served — or drop the losing transform from the list.");
      console.log("  Structure_score ranking is UNCHANGED; only which images reach it changes.");
    } else {
      console.log("The shipped ranking already serves a competitive arm; no candidate-set");
      console.log("change is warranted by this measurement.");
    }
  } catch (e) {
    console.log("error: " + ((e && e.message) || e));
    process.exitCode = 1;
  } finally {
    if (browser) { try { await browser.close(); } catch (e) { /* gone */ } }
    server.close();
  }
})();
