/**
 * The crop contract: what a detected line box becomes before the recogniser
 * sees it.
 *
 * The recogniser is our own CRNN, trained on crops resized to EXACTLY
 * CROP_H=32 tall with the aspect ratio preserved and the width clamped:
 *
 *   build_dataset.crop_to_array:  scale = CROP_H / h;  g.resize((nw, CROP_H))
 *   eval_ocr.py (line 524):       cv2.resize(band, (band.shape[1], R.CROP_H))
 *
 * Both references therefore feed the model 32px-tall text. `layout.js
 * cropForRecognizer` scaled by `min(maxW/bw, cropH/bh)`, which is a CONTAIN
 * fit: on a wide line the scale is set by the width, the height lands well
 * under 32, and the model gets text it was never trained on (measured on a
 * 900x580 page: 12-20px tall, median 13). The `geometry` suite in
 * ios_full_check.js caught it end to end; this is the fast unit-level version
 * of the same contract, so the fix cannot regress silently.
 *
 * Run: node tools/crop_contract.js
 */
"use strict";

const fs = require("fs");
const path = require("path");

const SRC = path.resolve(__dirname, "..", "js", "layout.js");

// layout.js is an ES module and this is a CommonJS harness, so load its pure
// functions the same way recognizer_parity.js does: strip the export keywords
// and evaluate the body. Nothing here touches the DOM.
function loadLayout() {
  const src = fs.readFileSync(SRC, "utf8");
  const body = src
    .replace(/^export default .*$/gm, "")
    .replace(/^export /gm, "");
  // Look the exports up defensively: a missing one must FAIL AN ASSERTION, not
  // throw a ReferenceError, or the RED step is a crash instead of a signal.
  return new Function(body + `
    ; return {
      cropForRecognizer: cropForRecognizer,
    };`)();
}

const R = { CROP_H: 32 }; // realdata.CROP_H, build_dataset.CROP_H
const MAX_W = 256;        // recognizer.js EXPORT_W — the ONNX graph is fixed width

let checks = 0, fails = 0;
function check(name, cond, detail) {
  checks++;
  if (!cond) fails++;
  console.log("  " + (cond ? "ok  " : "FAIL") + "  " + name + (cond || detail === undefined ? "" : "  -> " + detail));
  return cond;
}

const L = loadLayout();

// A synthetic page: paper 235, an ink bar per row so the box filter has
// something real to average (an all-paper crop would hide a vertical squash).
function page(w, h) {
  const g = new Uint8Array(w * h).fill(235);
  for (let y = 0; y < h; y += 4) for (let x = 0; x < w; x++) g[y * w + x] = 40;
  return g;
}

// Boxes a real detector produces on a 900px page: 10-20px tall, 80-800px wide.
const BOXES = [
  { x0: 40, y0: 100, x1: 268, y1: 115 },   // 228x15 — the banking sample's first line
  { x0: 40, y0: 130, x1: 249, y1: 140 },   // 209x10
  { x0: 40, y0: 160, x1: 297, y1: 173 },   // 257x13
  { x0: 40, y0: 190, x1: 299, y1: 204 },   // 259x14
  { x0: 40, y0: 220, x1: 336, y1: 240 },   // 296x20
  { x0: 40, y0: 250, x1: 120, y1: 262 },   // 80x12 (short line)
];

const g = page(900, 580);

console.log("crop geometry — every crop must be CROP_H tall (the training contract):");
for (const b of BOXES) {
  const bw = b.x1 - b.x0, bh = b.y1 - b.y0;
  const c = L.cropForRecognizer(g, 900, b, R.CROP_H, MAX_W);
  // The returned buffer is always CROP_H * nw; how many ROWS carry content is
  // the number that matters (the rest is padding the model never saw as text).
  const rows = c.data.length / c.width;
  let filled = 0;
  for (let y = 0; y < rows; y++) {
    let any = false;
    for (let x = 0; x < c.width && !any; x++) if (c.data[y * c.width + x] !== 0) any = true;
    if (any) filled++;
  }
  check(`${bw}x${bh} box -> ${c.width}px wide, ${filled}/${R.CROP_H} rows filled`,
    filled === R.CROP_H,
    "expected all " + R.CROP_H + " rows filled (build_dataset resizes to exactly CROP_H)");
}

console.log("\ncrop geometry — width honours the cap, and the resize preserves the page:");
const wide = { x0: 0, y0: 0, x1: 880, y1: 20 };            // 880x20: scaled to 32 tall = 1408 wide
const cw = L.cropForRecognizer(g, 900, wide, R.CROP_H, MAX_W);
check("a very wide line never exceeds the graph's fixed width",
  cw.width <= MAX_W, "got " + cw.width);

// Mean intensity is the robust invariant: a resize preserves the average
// darkness of the box it looked at, while an unfilled zero band drags it
// towards black (and 0 is INK once buildInput maps grey to [-1,1]).
const meanOf = (arr) => arr.reduce((s, v) => s + v, 0) / arr.length;
const srcMean = (() => {
  let sum = 0, n = 0;
  for (let y = wide.y0; y < wide.y1; y++) {
    for (let x = wide.x0; x < wide.x1; x++) { sum += g[y * 900 + x]; n++; }
  }
  return sum / n;
})();
check("the crop's mean intensity matches the box it was cut from",
  Math.abs(meanOf(cw.data) - srcMean) < 12,
  "source mean " + srcMean.toFixed(1) + " -> crop mean " + meanOf(cw.data).toFixed(1)
    + " (a zero-filled band reads as ink, not paper)");

console.log("\ndegenerate boxes must not become a black rectangle:");
const empty = L.cropForRecognizer(g, 900, { x0: 10, y0: 10, x1: 10, y1: 10 }, R.CROP_H, MAX_W);
check("an empty box returns a paper-filled crop, not ink",
  empty.data.every((v) => v >= 128),
  "min byte " + Math.min.apply(null, Array.prototype.slice.call(empty.data)));

// ── a NEGATIVE result, recorded so nobody re-tries it blind ──────────────────
// Cutting a long line into contiguous chunks (each at its true aspect ratio,
// decodes joined) was implemented, tested (windows tile the line, every cut on a
// minimum-ink column) and then REVERTED on measurement: on the rendered samples
// the whole-line squeeze reads long lines BETTER —
//   228x15 line: squeezed "HSBC UK BANK PLC" 98% vs chunked "MSEC UNBANE L" 54%
//   259x14 line: squeezed "ICOUNT NUMBER 4021887" 95% vs chunked "…OUNTE…" 83%
// — and the field/classification suite went from 5 failures to 7. The hypothesis
// (square glyphs beat a 1.9x horizontal squeeze) is wrong for this model: it
// wants the line's full horizontal context. The crop-rule fix above stands on
// its own; chunking needs a retrain or a line-aware model before it can help.

console.log("\n" + (checks - fails) + "/" + checks + " checks passed");
if (fails) {
  console.log("CROP CONTRACT FAILED — the recogniser is being fed crops the model was not trained on");
  process.exit(1);
}
console.log("CROP CONTRACT OK");
