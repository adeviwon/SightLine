/**
 * Node harness for the selftest.html IMAGE battery.
 *
 * Installs tools/canvas_shim.js so the real js/pipeline.js image code runs
 * headlessly, then executes the same assertions as testImagePipeline() in
 * selftest.html. Run: node tools/selftest_image.js
 */
"use strict";

const shim = require("./canvas_shim.js");
const teardown = shim.install();

const SightLine = require("../js/pipeline.js");
const SightLineRestorer = require("../js/restorer.js");

let total = 0, failed = 0;
const groups = [];
const group = (n) => { const g = { name: n, rows: [] }; groups.push(g); return g; };
function check(g, desc, actual, expected) {
  let pass;
  if (typeof expected === "function") pass = !!expected(actual);
  else pass = JSON.stringify(actual) === JSON.stringify(expected);
  total++;
  if (!pass) failed++;
  g.rows.push({ desc, pass, actual, expected });
  return pass;
}
const eq = (g, d, a, e) => check(g, d, a, e);

function makeDoc(w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#f8f8f8"; ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#1c1c1c";
  ctx.font = "14px sans-serif";
  for (let i = 0; i < 8; i++) ctx.fillText("Account Number 40218877 line " + i, 10, 14 + i * 18);
  return c;
}

// ── 1:1 with selftest.html testImagePipeline() ───────────────────────
const g = group("Image preprocessing - the canvas path");
const W = 320, H = 200;
const c = makeDoc(W, H);

const gray = SightLine.toGrayCanvas(c, W, H);
eq(g, "grayscale canvas keeps dimensions", gray.width + "x" + gray.height, W + "x" + H);

const sharp = SightLine.assessQuality(gray);
check(g, "text image is not classified as blurry", sharp.blurry, (v) => v === false);
check(g, "text image is not classified as dark", sharp.dark, (v) => v === false);
check(g, "quality score is within [0,1]", sharp.quality, (v) => v >= 0 && v <= 1);
check(g, "Laplacian variance is positive for text", sharp.lapVar, (v) => v > 0);

const blank = document.createElement("canvas");
blank.width = 64; blank.height = 64;
const bctx = blank.getContext("2d", { willReadFrequently: true });
bctx.fillStyle = "#ffffff"; bctx.fillRect(0, 0, 64, 64);
eq(g, "blank image is detected as blurry", SightLine.assessQuality(blank).blurry, true);

const dark = document.createElement("canvas");
dark.width = 64; dark.height = 64;
const dctx = dark.getContext("2d", { willReadFrequently: true });
dctx.fillStyle = "#000000"; dctx.fillRect(0, 0, 64, 64);
eq(g, "black image is detected as dark", SightLine.assessQuality(dark).dark, true);

const med = SightLine.median3(gray);
eq(g, "median preserves dimensions", med.width + "x" + med.height, W + "x" + H);
const un = SightLine.unsharp(gray, 1.8, 2);
eq(g, "unsharp preserves dimensions", un.width + "x" + un.height, W + "x" + H);
const cl = SightLine.clahe(gray, 2.0, 8);
eq(g, "CLAHE preserves dimensions", cl.width + "x" + cl.height, W + "x" + H);
const bin = SightLine.binarize(gray);
eq(g, "binarize preserves dimensions", bin.width + "x" + bin.height, W + "x" + H);

const bd = bin.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, W, H).data;
let binary = true;
for (let i = 0; i < bd.length; i += 4) if (bd[i] !== 0 && bd[i] !== 255) { binary = false; break; }
eq(g, "binarize output is strictly 0 or 255", binary, true);

const hist = new Uint32Array(256);
for (let i = 0; i < 8000; i++) hist[30]++;
for (let i = 0; i < 2000; i++) hist[220]++;
const t = SightLine.otsu(hist, 10000);
check(g, "Otsu threshold falls between the two modes", t, (v) => v >= 30 && v <= 220);

const pp = SightLine.preprocess(c, 3200);
check(g, "preprocess returns OCR candidates", pp.candidates.length, (v) => v >= 2);
check(g, "preprocess reports a skew estimate", typeof pp.skew, (v) => v === "number");
eq(g, "preprocess caps the long edge at maxDim",
  Math.max(pp.width, pp.height), Math.min(3200, Math.max(W, H)));
check(g, "preprocess returns warnings as an array", Array.isArray(pp.warnings), (v) => v === true);

// Downscale path.
const big = document.createElement("canvas");
big.width = 8000; big.height = 4000;
const gctx = big.getContext("2d", { willReadFrequently: true });
gctx.fillStyle = "#fff"; gctx.fillRect(0, 0, 8000, 4000);
gctx.fillStyle = "#000"; gctx.font = "70px sans-serif";
gctx.fillText("Sort Code 40-11-04", 100, 400);
const pp2 = SightLine.preprocess(big, 3200);
eq(g, "an 8000px image is downscaled to 3200px", pp2.width, 3200);
check(g, "aspect ratio is preserved by the downscale", Math.abs(pp2.height - 1600) <= 2, (v) => v === true);

// restoreMode. The modes are not always different, and the test has to know
// which is which or it asserts a coincidence:
//   adaptive  -> noisy: median+unsharp · blurry: unsharp only · neither: NO-OP
//   classical -> always median+unsharp
// So on a NOISY input they coincide exactly; they diverge on a merely BLURRY
// input (adaptive skips the median) and on a clean one (adaptive is a no-op).
// Blurring is also the case that matters, so it is what we test here.
// The blurred fixture is a real box blur applied with putImageData. A
// downscale/upscale round trip is not enough: the 5x7 glyphs are so
// high-frequency that even four passes leave lapVar ~4000, still "noisy".
// Six 7x7 box passes reach lapVar ~4, which is genuinely blurry.
function boxBlur(src, w, h, radius) {
  const out = src.getContext("2d", { willReadFrequently: true }).createImageData(w, h);
  const s = src.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h).data;
  const o = out.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, n = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
          sum += s[(yy * w + xx) * 4]; n++;
        }
      }
      const i = (y * w + x) * 4;
      const v = (sum / n) | 0;
      o[i] = o[i + 1] = o[i + 2] = v; o[i + 3] = 255;
    }
  }
  return out;
}
const blurred = document.createElement("canvas");
blurred.width = 320; blurred.height = 200;
const blctx = blurred.getContext("2d", { willReadFrequently: true });
blctx.fillStyle = "#f8f8f8"; blctx.fillRect(0, 0, 320, 200);
blctx.drawImage(c, 0, 0, 320, 200);
for (let pass = 0; pass < 6; pass++) {
  blctx.putImageData(boxBlur(blurred, 320, 200, 3), 0, 0);
}
const blq = SightLine.assessQuality(SightLine.toGrayCanvas(blurred, 320, 200));
check(g, "the blurred fixture is actually detected as blurry", blq.blurry, (v) => v === true);
check(g, "the blurred fixture is not also flagged as noisy", blq.noisy, (v) => v === false);

const px = (cv) => {
  const d = cv.getContext("2d", { willReadFrequently: true })
    .getImageData(0, 0, cv.width, cv.height).data;
  let h = 0;
  for (let i = 0; i < d.length; i += 4) h = (h * 31 + d[i]) | 0;
  return h;
};
const bA = SightLine.preprocess(blurred, 3200, { restoreMode: "adaptive" });
const bC = SightLine.preprocess(blurred, 3200, { restoreMode: "classical" });
const bN = SightLine.preprocess(blurred, 3200, { restoreMode: "none" });
eq(g, "preprocess reports the restore mode it used", bC.restoreMode, "classical");
check(g, "on blurred input, classical and adaptive differ (median vs no median)",
  px(bA.canvas), (v) => v !== px(bC.canvas));
check(g, "on blurred input, classical and none differ",
  px(bN.canvas), (v) => v !== px(bC.canvas));
// The clean fixture is noisy (the 5x7 glyphs are high-frequency), so adaptive
// and classical must coincide there — that is the documented behaviour, and
// asserting the coincidence is what stops someone "fixing" it later.
const cA = SightLine.preprocess(c, 3200, { restoreMode: "adaptive" });
const cC = SightLine.preprocess(c, 3200, { restoreMode: "classical" });
check(g, "on noisy input, adaptive and classical coincide (both median+unsharp)",
  px(cA.canvas), (v) => v === px(cC.canvas));
eq(g, "every mode still returns OCR candidates",
  [bA, bC, bN, cA, cC].every((m) => m.candidates.length >= 2), true);
eq(g, "the default mode is adaptive", SightLine.preprocess(c, 3200).restoreMode, "adaptive");

// ── Extra: verify the restorer's classical fallback runs and preserves pixels ──
const g2 = group("Restorer classical fallback - evaluate.arm_classical");
(async () => {
  const r = await SightLineRestorer.restore(gray, { forceClassical: true });
  eq(g2, "classical restore does not throw", r.engine, "classical");
  eq(g2, "classical restore preserves dimensions",
    r.canvas.width + "x" + r.canvas.height, W + "x" + H);
  check(g2, "classical restore produces a canvas", r.canvas, (v) => !!v && typeof v.getContext === "function");

  const rc = await SightLineRestorer.armRestorerClahe(gray, { forceClassical: true });
  eq(g2, "arm_restorer_clahe returns the restorer+clahe stage", rc.stage, "restorer_clahe");
  check(g2, "arm_restorer_clahe yields OCR candidates", rc.candidates.length, (v) => v >= 2);

  // The ONNX path must be a graceful no-op when the model is absent, not a throw.
  let threw = false;
  try { await SightLineRestorer.restore(gray, {}); } catch (e) { threw = true; }
  eq(g2, "restore() never throws even with no .onnx and no ORT runtime", threw, false);

  let threw2 = false;
  try { await SightLineRestorer.armRestorerClahe(gray, {}); } catch (e) { threw2 = true; }
  eq(g2, "armRestorerClahe() never throws either", threw2, false);

  for (const g3 of groups) {
    const bad = g3.rows.filter((r) => !r.pass);
    console.log((bad.length ? "FAIL" : "ok  ") + "  " + g3.name +
      "  (" + (g3.rows.length - bad.length) + "/" + g3.rows.length + ")");
    for (const row of bad) {
      console.log("        FAIL: " + row.desc);
      console.log("              got      " + JSON.stringify(row.actual));
      console.log("              expected " + JSON.stringify(row.expected));
    }
  }
  console.log("\n" + (failed === 0 ? "ALL " + total + " ASSERTIONS PASSED" : failed + " of " + total + " FAILED"));
  teardown();
  process.exit(failed === 0 ? 0 : 1);
})();
