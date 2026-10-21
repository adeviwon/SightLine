/**
 * deskew_verify.js — DOES THE APP'S OWN DESKEW (pipeline.js estimateSkew +
 * rotate) HELP OR HURT LINE DETECTION ON THE off_axis PAGES?
 *
 * The question, compressed: `off_axis` is dict(perspective=0.045, rot=4.0).
 * The shipped detector loses ~85% of the lines there (legal/off_axis: 2-3 bands
 * against 15 drawn lines), and preprocess() ALREADY deskews every page before
 * any detector sees it (`const deskewed = angle !== 0 ? rotate(gray, angle) :
 * gray;`). So either estimateSkew() returns a sane angle and the bander still
 * fails on the straightened page, or the estimator itself is wrong and the
 * "deskew" is rotating a tilted page into a worse tilt.
 *
 * MEASURE NOTHING BEFORE REPRODUCING SOMETHING KNOWN. This tool refuses to
 * report off_axis numbers until the same detectLines() call has REPRODUCED the
 * drawn line counts on the app's own control fixtures. If the controls do not
 * come back, the harness is broken and the off_axis numbers are noise.
 *
 * WHAT IT PRINTS
 *   1. HARNESS VALIDATION — control profiles (studio_clean, handheld_light,
 *      jpeg_social), detected bands on the RAW fixture vs drawn lines from
 *      ml/src/capture.py DOCS (gt_lines.json — the arm_ab.js convention).
 *   2. OFF_AXIS TABLE — per document: estimateSkew()'s angle, the generator's
 *      own rot (4.0), bands on the raw page / after the app's own deskew / after
 *      a true-angle de-rotation, and the ink-centroid slope against x before and
 *      after the deskew (the ml/src/line_detect.py _slope_of measure).
 *   3. CAUSE — (a) estimator wrong / (b) estimator right, bander fails on the
 *      corrected page / (c) needs a rectification the app does not have, with
 *      the rule that decides among them printed alongside the numbers.
 *
 * Run:  node app/tools/deskew_verify.js [--engine=webkit]
 * No source file is modified. No recogniser is run — this is a DETECTION question.
 */
"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");

const pw = require("./pw_core.js")();
const chromeBin = require("./chrome_path.js")();

const APP = path.resolve(__dirname, "..");
const SAMPLES = "/tmp/sightline_samples";
const GT_CACHE = path.join(SAMPLES, "gt_lines.json");
const MAX_DIM = 3200;
const TRUE_ROT = 4.0;                     // capture.CAPTURE_PROFILES["off_axis"]["rot"]
const CONTROL = ["studio_clean", "handheld_light", "jpeg_social"];
// A page's count is "reproduced" when it lands within this of the drawn count.
// The detector splits one drawn line into a fragment or merges a tight pair, so
// +-1 is normal; anything past this is the detector losing the page, not noise.
const TOL = 2;

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8", ".wasm": "application/wasm",
  ".onnx": "application/octet-stream", ".png": "image/png", ".css": "text/css",
};

/** drawn lines, from the renderer that DREW them (arm_ab.js convention). */
function groundTruthLines() {
  const c = JSON.parse(fs.readFileSync(GT_CACHE, "utf8"));
  return c;                               // { prescription: [...], banking, legal }
}

function serve() {
  const srv = http.createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split("?")[0]);
    const p = rel.startsWith("/sample/") ? path.join(SAMPLES, rel.slice(8))
                                         : path.join(APP, rel);
    if (!p.startsWith(APP) && !p.startsWith(SAMPLES)) { res.writeHead(403); return res.end("no"); }
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
async function inPage(cfg) {
  try {
    const L = await import(cfg.base + "js/layout.js");
    if (typeof SightLine === "undefined") return { ok: false, error: "SightLine not on the page" };

    const toGray = (c) => {
      const w = c.width, h = c.height;
      const d = c.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h).data;
      const g = new Uint8Array(w * h);
      for (let i = 0, p = 0; i < g.length; i++, p += 4) g[i] = (d[p] * 77 + d[p + 1] * 150 + d[p + 2] * 29) >> 8;
      return { gray: g, w, h };
    };
    const mask = (g, w, h) => { const S = L.scaleFor(h); return L.inkMask(g, w, h, S.median, S.delta); };

    // the JS mirror of ml/src/line_detect.py _slope_of: least-squares dy/dx over
    // per-column ink centroids, one centroid per occupied column.
    function slopeOf(ink, w, h) {
      let nInk = 0; const cols = [], cents = [];
      for (let x = 0; x < w; x++) {
        let s = 0, n = 0;
        for (let y = 0; y < h; y++) if (ink[y * w + x]) { s += y; n++; }
        if (n) { cols.push(x); cents.push(s / n); nInk += n; }
      }
      if (nInk < 200 || cols.length < 20) return 0;
      const m = cols.length;
      let mx = 0, my = 0;
      for (let i = 0; i < m; i++) { mx += cols[i]; my += cents[i]; }
      mx /= m; my /= m;
      let cov = 0, var_ = 0;
      for (let i = 0; i < m; i++) { cov += (cols[i] - mx) * (cents[i] - my); var_ += (cols[i] - mx) ** 2; }
      return var_ ? cov / var_ : 0;
    }

    // matches pipeline.js rotate(): content rotated by -deg in canvas coords,
    // white outside. Nearest neighbour (this is a probe, not the shipped arm).
    function rotateGray(g, w, h, deg) {
      const a = (-deg * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
      const cx = w / 2, cy = h / 2, out = new Uint8Array(w * h).fill(255);
      for (let Y = 0; Y < h; Y++) {
        const v = Y - cy;
        for (let X = 0; X < w; X++) {
          const u = X - cx;
          const sx = Math.round(cx + u * ca - v * sa);
          const sy = Math.round(cy + u * sa + v * ca);
          if (sx >= 0 && sx < w && sy >= 0 && sy < h) out[Y * w + X] = g[sy * w + sx];
        }
      }
      return out;
    }

    const bandsOf = (g, w, h) => L.detectLines(g, w, h).slice().sort((a, b) => a.y0 - b.y0);

    const docs = [];
    for (const s of cfg.samples) {
      const img = new Image();
      img.src = s.src;
      await img.decode();
      const cv = document.createElement("canvas");
      cv.width = img.naturalWidth; cv.height = img.naturalHeight;
      cv.getContext("2d", { willReadFrequently: true }).drawImage(img, 0, 0);

      const src = toGray(cv);
      const inkSrc = mask(src.gray, src.w, src.h);
      const bandsSrc = bandsOf(src.gray, src.w, src.h);
      const slopeSrc = slopeOf(inkSrc, src.w, src.h);

      // what the APP already does: preprocess() deskews, every candidate derives
      // from the deskewed canvas. restoreMode "none" => candidates[0] IS the
      // deskewed grayscale, rotation isolated from any restoration.
      const pp = SightLine.preprocess(img, cfg.maxDim, { restoreMode: "none" });
      const dsk = toGray(pp.candidates[0]);
      const sameDims = dsk.w === src.w && dsk.h === src.h;
      const inkDsk = sameDims ? mask(dsk.gray, dsk.w, dsk.h) : null;
      const dskBands = sameDims ? bandsOf(dsk.gray, dsk.w, dsk.h) : null;
      const slopeDsk = sameDims ? slopeOf(inkDsk, dsk.w, dsk.h) : null;

      const rec = {
        name: s.name, doc: s.doc, profile: s.profile, offAxis: !!s.offAxis,
        w: src.w, h: src.h, appSkew: +(pp.skew || 0),
        bandsSrc: bandsSrc.length, slopeSrc: +slopeSrc.toFixed(4),
        bandsDsk: dskBands ? dskBands.length : null,
        slopeDsk: slopeDsk == null ? null : +slopeDsk.toFixed(4),
      };

      if (s.offAxis) {
        // de-rotate by the generator's OWN rot, both signs; keep the one that
        // recovers more bands. A pure rotation is the only hypothesised fix.
        const arms = [];
        for (const r of [cfg.trueRot, -cfg.trueRot]) {
          const g = rotateGray(src.gray, src.w, src.h, r);
          const ink = mask(g, src.w, src.h);
          arms.push({ rot: r, bands: bandsOf(g, src.w, src.h).length,
                      slope: +slopeOf(ink, src.w, src.h).toFixed(4) });
        }
        arms.sort((a, b) => b.bands - a.bands);
        rec.derot = arms;                       // [best, other]
        rec.bandsTrue = arms[0].bands;
        rec.trueRot = arms[0].rot;
        rec.slopeTrue = arms[0].slope;

        // ── DIAGNOSTIC: is the app's deskew the SAME transform as this harness
        //    rotating the source by the angle the app itself reports? If dPlus
        //    (nearest-neighbour rotate by +appSkew) is small, the app applied
        //    +appSkew as this harness would and any band difference is image
        //    content; if dMinus is the small one, the app applied the OPPOSITE
        //    direction. dPlus==dMinus==large => the app did something else.
        const grayToCanvas = (g, w, h) => {
          const c = SightLine.makeCanvas(w, h);
          const ctx = SightLine._ctx2d(c);
          const im = ctx.createImageData(w, h);
          for (let i = 0, p = 0; i < g.length; i++, p += 4) { im.data[p] = im.data[p + 1] = im.data[p + 2] = g[i]; im.data[p + 3] = 255; }
          ctx.putImageData(im, 0, 0);
          return c;
        };
        const diff = (a, b) => { let t = 0; for (let i = 0; i < a.length; i++) t += Math.abs(a[i] - b[i]); return +(t / a.length).toFixed(2); };
        const gPlus = rotateGray(src.gray, src.w, src.h, +rec.appSkew);
        const gMinus = rotateGray(src.gray, src.w, src.h, -rec.appSkew);
        rec.diag = {
          appSkew: rec.appSkew,
          dZero: diff(dsk.gray, src.gray), dPlus: diff(dsk.gray, gPlus), dMinus: diff(dsk.gray, gMinus),
          bandsPlusAppSkew: bandsOf(gPlus, src.w, src.h).length,
          bandsMinusAppSkew: bandsOf(gMinus, src.w, src.h).length,
          estSkewOnSource: +SightLine.estimateSkew(cv),
          estSkewOnDsk: +SightLine.estimateSkew(pp.candidates[0]),
          estSkewOnPlus4: +SightLine.estimateSkew(grayToCanvas(gPlus, src.w, src.h)),
          estSkewOnMinus4: +SightLine.estimateSkew(grayToCanvas(gMinus, src.w, src.h)),
        };
      }
      docs.push(rec);
    }
    return { ok: true, docs };
  } catch (e) {
    const msg = (e && e.name ? e.name + ": " : "") + (e && e.message ? e.message : String(e));
    return { ok: false, error: msg,
             stack: e && e.stack ? String(e.stack).split("\n").slice(0, 4).join(" | ") : null };
  }
}

const med = (xs) => { if (!xs.length) return 0; const s = xs.slice().sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };

function main(docs, gt) {
  const truth = (d) => (gt[d.doc] || []).length;
  const ctl = docs.filter((d) => !d.offAxis);
  const off = docs.filter((d) => d.offAxis);

  // ── 1. HARNESS VALIDATION ────────────────────────────────────────────────
  console.log("\n=== 1. HARNESS VALIDATION — control profiles, bands vs DRAWN lines ===");
  console.log("   harness      = app/js/layout.js detectLines() on the RAW fixture gray");
  console.log("   truth        = lines ml/src/capture.py DREW (gt_lines.json)");
  console.log("   after_deskew = the same detector on SightLine.preprocess(restoreMode:none) candidate 0");
  console.log("   appSkew      = estimateSkew()'s angle on that page (deg; |a|<0.5 -> 0 by the estimator's own clamp)");
  console.log("   tolerance    = |bands-truth| <= " + TOL + "  (fragment/merge noise)\n");
  console.log("     " + "page".padEnd(30) + "truth" + "bands".padStart(7) + "after".padStart(7)
    + "appSkew".padStart(9) + "  pass");
  let pass = 0;
  for (const d of ctl) {
    const ok = Math.abs(d.bandsSrc - truth(d)) <= TOL;
    if (ok) pass++;
    console.log("     " + (d.doc + "/" + d.profile).padEnd(30) + String(truth(d)).padStart(5)
      + String(d.bandsSrc).padStart(7) + String(d.bandsDsk == null ? "-" : d.bandsDsk).padStart(7)
      + (d.appSkew + "deg").padStart(9) + "  " + (ok ? "ok" : "FAIL"));
  }
  const harnessOk = ctl.length > 0 && pass === ctl.length;
  console.log("\n   RESULT: " + pass + "/" + ctl.length + " control pages reproduced"
    + (harnessOk ? "  -> HARNESS VALID" : "  -> HARNESS INVALID, off_axis below is untrustworthy"));

  // ── 2. OFF_AXIS TABLE ────────────────────────────────────────────────────
  console.log("\n=== 2. OFF_AXIS TABLE — the app's deskew, measured ===");
  console.log("   trueRot   = capture.CAPTURE_PROFILES['off_axis']['rot'] = " + TRUE_ROT);
  console.log("     " + "page".padEnd(24) + "truth" + "estSkew".padStart(9) + "trueRot".padStart(9)
    + "bandsSrc".padStart(10) + "afterDsk".padStart(10) + "afterTrue".padStart(11)
    + "slopeSrc".padStart(11) + "slopeDsk".padStart(11) + "slopeTrue".padStart(11));
  for (const d of off) {
    console.log("     " + (d.doc + "/" + d.profile).padEnd(24) + String(truth(d)).padStart(5)
      + (d.appSkew + "d").padStart(9) + (d.trueRot + "d").padStart(9)
      + String(d.bandsSrc).padStart(10) + String(d.bandsDsk).padStart(10)
      + String(d.bandsTrue).padStart(11)
      + String(d.slopeSrc).padStart(11) + String(d.slopeDsk).padStart(11)
      + String(d.slopeTrue).padStart(11));
  }
  for (const d of off) {
    console.log("   " + (d.doc + "/" + d.profile) + "  derot arms: "
      + d.derot.map((a) => a.rot + "deg -> " + a.bands + " bands, slope " + a.slope).join("  |  "));
  }

  console.log("\n   DIRECTION DIAGNOSTIC — is the app's deskew the same transform?");
  console.log("     page                 appSkew  dZero   dPlus  dMinus  band+sk  band-sk  estSkewOnSrc  onDsk  on+4  on-4");
  for (const d of off) {
    const q = d.diag;
    console.log("     " + (d.doc + "/" + d.profile).padEnd(20)
      + (q.appSkew + "d").padStart(8) + String(q.dZero).padStart(7) + String(q.dPlus).padStart(7)
      + String(q.dMinus).padStart(7) + String(q.bandsPlusAppSkew).padStart(8)
      + String(q.bandsMinusAppSkew).padStart(9) + (q.estSkewOnSource + "d").padStart(14)
      + (q.estSkewOnDsk + "d").padStart(7) + (q.estSkewOnPlus4 + "d").padStart(6)
      + (q.estSkewOnMinus4 + "d").padStart(6));
  }

  // ── 3. CAUSE (the rule decides, printed with its inputs) ─────────────────
  // (a) estimator wrong  : afterTrue recovers the count but afterDsk does not AND
  //                        estSkew != -trueRot -> the estimator's angle is the bug.
  // (b) bander fails     : afterTrue ALSO loses the count -> the rotation is not
  //                        the problem, the bander is (a layout.js fix).
  // (c) needs rectification: afterTrue recovers the count ONLY partially and the
  //                        residual slope stays high -> a pure rotation cannot
  //                        undo the perspective component.
  const rows = off.map((d) => ({
    t: truth(d), dsk: d.bandsDsk, tru: d.bandsTrue, skew: d.appSkew, rot: d.trueRot,
  }));
  const trueRecovers = rows.every((r) => Math.abs(r.tru - r.t) <= TOL);
  const dskRecovers = rows.every((r) => Math.abs(r.dsk - r.t) <= TOL);
  const skewMatches = rows.every((r) => Math.abs(Math.abs(r.skew) - TRUE_ROT) <= 0.75);
  // right magnitude but the applied rotation does not straighten => sign inversion
  const inverted = off.every((d) => d.diag
    && Math.abs(d.diag.estSkewOnPlus4) < 0.5 && Math.abs(d.diag.estSkewOnDsk) > 0.5);
  let cause;
  if (dskRecovers) cause = "(none) the app's own deskew already recovers the lines";
  else if (trueRecovers && inverted) cause = "(a) estimator magnitude is right, the rotation SIGN is inverted "
    + "(estimateSkew and rotate() disagree on direction; applied rotation equals rotate(source, -angle), which DOUBLES the tilt)";
  else if (trueRecovers && !skewMatches) cause = "(a) estimator returns a wrong angle (magnitude too)";
  else if (trueRecovers && skewMatches) cause = "(d) estimator angle is right but something else keeps afterDsk low";
  else if (!trueRecovers) cause = "(b)/(c) a pure de-rotation is not enough";
  else cause = "(d) other";

  console.log("\n=== 3. CAUSE ===");
  console.log("   off_axis bands after a TRUE-angle de-rotation: " + rows.map((r) => r.tru + "/" + r.t).join(", ")
    + "  -> " + (trueRecovers ? "recovered (within +-" + TOL + ")" : "NOT recovered"));
  console.log("   off_axis bands after the APP's own deskew:     " + rows.map((r) => r.dsk + "/" + r.t).join(", ")
    + "  -> " + (dskRecovers ? "recovered" : "LOST"));
  console.log("   estimateSkew() magnitude vs generator " + TRUE_ROT + "deg:   " + rows.map((r) => Math.abs(r.skew) + "d").join(", ")
    + "  -> " + (skewMatches ? "right magnitude" : "WRONG magnitude"));
  console.log("   app's deskew output vs rotate(source,+appSkew)/-appSkew (mean|diff|; SMALLER = the sense actually applied): " + off.map((d) => d.diag ? d.diag.dMinus : "-").join(", ")
    + " (mean |diff|) vs +appSkew: " + off.map((d) => d.diag ? d.diag.dPlus : "-").join(", "));
  console.log("   estimateSkew(rotate(source, +|skew|)) = " + off.map((d) => d.diag ? d.diag.estSkewOnPlus4 + "d" : "-").join(", ")
    + "  (the estimator's OWN target reads as straight there)");
  console.log("   CAUSE: " + cause);

  // ── 4. PROSE CHANGE (no source file is modified by this tool) ────────────
  console.log("\n=== 4. THE CHANGE (prose — this tool edits nothing) ===");
  if (trueRecovers && !dskRecovers) {
    console.log("   app/js/pipeline.js preprocess():");
    console.log("     -  const deskewed = angle !== 0 ? rotate(gray, angle) : gray;");
    console.log("     +  const deskewed = angle !== 0 ? rotate(gray, -angle) : gray;");
    console.log("   rotate() has exactly one caller (preprocess:759), so the negation is contained.");
    console.log("   Guard: none needed — on every control page estimateSkew() returns 0, so rotate()");
    console.log("   is not called and the change is unreachable there (measured: after-deskew bands");
    console.log("   on the " + ctl.length + " control pages = " + ctl.map((d) => d.bandsDsk).join("/") + ", unchanged).");
    console.log("   No rectification is required: a pure +/-" + TRUE_ROT + "deg rotation already recovers the lines.");
  } else {
    console.log("   No single-expression deskew fix is warranted by these numbers; see the cause above.");
  }
  // THE ROUND TRIP — the gate that would have caught the sign inversion.
  // Apply the estimator's own correction, RE-ESTIMATE the result, require ~0. That is a
  // different property from "estimateSkew returns a plausible angle", and it was true
  // throughout the entire period the applied rotation was DOUBLING the tilt: the
  // estimator returned 4.5/3.25/4.0 against a true 4.0 the whole time, while the app
  // re-estimated its own deskewed output at 6.5-9.0 degrees.
  // Also asserts the estimator does not FIRE on straight pages: with |skew| < 0.5 the
  // rotation is never applied, which is why the working profiles cannot be affected.
  const roundTripOk = ctl.every((d) => Math.abs(d.appSkew) < 0.5)
    && off.every((d) => !(d.diag && Math.abs(d.appSkew) >= 0.5)
      || Math.abs(d.diag.estSkewOnDsk) < 0.5);
  console.log("\n=== 5. ROUND TRIP (the gate) ===");
  console.log("   apply estimateSkew's correction, re-estimate, require ~0");
  console.log("   controls with a fired deskew (must be 0): "
    + ctl.filter((d) => Math.abs(d.appSkew) >= 0.5).length
    + " / " + ctl.length + ";  off_axis residual skew: "
    + off.map((d) => (d.diag ? d.diag.estSkewOnDsk + "d" : "n/a")).join(" ")
    + "  ->  " + (roundTripOk ? "PASS" : "FAIL (the deskew does not straighten the page)"));
  return { ctl, off, harnessOk, roundTripOk, pass, cause, rows };
}

(async () => {
  const engArg = (process.argv.find((a) => a.startsWith("--engine=")) || "").split("=")[1];
  const engine = engArg === "chromium" ? "chromium" : "webkit";

  const manifest = JSON.parse(fs.readFileSync(path.join(SAMPLES, "manifest.json"), "utf8"));
  const gt = groundTruthLines();
  const samples = (manifest.samples || []).filter((s) => fs.existsSync(s.file)).map((s) => ({
    name: path.basename(s.file), doc: s.doc, profile: s.profile,
    src: "/sample/" + path.basename(s.file), offAxis: s.profile === "off_axis",
  }));
  const nCtlProfiles = new Set(samples.filter((s) => !s.offAxis).map((s) => s.profile)).size;

  console.log("engine=" + engine + "  fixtures=" + samples.length
    + "  (off_axis " + samples.filter((s) => s.offAxis).length + ", control "
    + samples.filter((s) => !s.offAxis).length + " across " + nCtlProfiles + " profiles)");

  const server = await serve();
  const base = "http://127.0.0.1:" + server.address().port + "/";
  let browser = null;
  try {
    browser = engine === "chromium" ? await pw.chromium.launch({ executablePath: chromeBin })
                                    : await pw.webkit.launch();
    const page = await browser.newPage();
    page.setDefaultTimeout(600000);
    await page.goto(base + "index.html", { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(4000);
    const res = await page.evaluate(inPage, { base, maxDim: MAX_DIM, samples, trueRot: TRUE_ROT });
    if (!res || !res.ok) {
      console.log("PAGE ERROR: " + ((res && (res.stack || res.error)) || "evaluate returned undefined"));
      process.exitCode = 1; return;
    }
    const out = main(res.docs, gt);
    if (!out.harnessOk) { console.log("\n   (off_axis rows above are NOT evidence until the harness passes.)"); process.exitCode = 1; }
    if (!out.roundTripOk) { console.log("\n   ROUND-TRIP GATE FAILED: the deskew does not straighten the page."); process.exitCode = 1; }
  } catch (e) {
    console.log("error: " + ((e && e.message) || e));
    process.exitCode = 1;
  } finally {
    if (browser) { try { await browser.close(); } catch (e) { /* gone */ } }
    server.close();
  }
})();
