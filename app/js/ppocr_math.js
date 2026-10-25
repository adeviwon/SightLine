// js/ocr_ppocr.js — the PP-OCRv5 ALGORITHMS, as pure functions.
//
// WHY THIS IS A SEPARATE FILE FROM js/ocr.js
//   js/ocr.js holds the engine objects and talks to the browser (ORT, canvases,
//   timeouts). Everything in here is arithmetic on typed arrays with no imports
//   and no globals, so gate G2 can unit-test the two hard parts — DB box
//   post-processing and CTC decoding — in plain Node, on synthetic inputs, with
//   no WASM and no browser. Those are the parts where a silent mistake produces
//   plausible-looking but wrong text rather than an exception, so they are the
//   parts that most need a test.
//
// WHAT THE TWO MODELS ARE
//   DET  (ppocrv5-mobile-det) — a DB (Differentiable Binarization) segmenter:
//        in [1,3,H,W] float32, out [1,1,H,W] probability map, 1 = text. Turning
//        that map into boxes is `dbPostProcess()`.
//   REC  (ppocrv5-mobile-rec) — a CTC sequence model: in [1,3,48,W] float32,
//        out [1,T,C] logits, decoded by `ctcDecode()`.
//
// PORTED, NOT INVENTED
//   `dbPostProcess` and `ctcDecode` reproduce PaddleOCR's own
//   `DBPostProcess.__call__` / `CTCLabelDecode.__call__` behaviour. The two
//   deliberate deviations are recorded at their definitions: the boxes are
//   AXIS-ALIGNED rather than rotated polygons (no pyclipper here), and the
//   component labelling is a flood fill over the thresholded map rather than
//   OpenCV's `findContours`.

// ---------------------------------------------------------------------------
// Image plumbing
// ---------------------------------------------------------------------------

/**
 * Bilinear resample of an 8-bit single-channel image.
 * @returns {Uint8Array} dw*dh bytes
 */
export function resizeLuma(src, sw, sh, dw, dh) {
  const out = new Uint8Array(dw * dh);
  if (sw === dw && sh === dh) { out.set(src.subarray(0, dw * dh)); return out; }
  const xr = sw / dw;
  const yr = sh / dh;
  for (let y = 0; y < dh; y++) {
    // Sample at pixel centres, and clamp so the last row/column is not read
    // one pixel past the edge (which would darken every right/bottom edge).
    const sy = Math.min(sh - 1, Math.max(0, (y + 0.5) * yr - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(sh - 1, y0 + 1);
    const fy = sy - y0;
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, Math.max(0, (x + 0.5) * xr - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(sw - 1, x0 + 1);
      const fxx = sx - x0;
      const a = src[y0 * sw + x0];
      const b = src[y0 * sw + x1];
      const c = src[y1 * sw + x0];
      const d = src[y1 * sw + x1];
      const top = a + (b - a) * fxx;
      const bot = c + (d - c) * fxx;
      out[y * dw + x] = Math.round(top + (bot - top) * fy);
    }
  }
  return out;
}

/** Clamp a box to the image, returning integer bounds. Null if it is empty. */
export function clampBox(box, w, h) {
  const x0 = Math.max(0, Math.min(w - 1, Math.floor(box.x0)));
  const y0 = Math.max(0, Math.min(h - 1, Math.floor(box.y0)));
  const x1 = Math.max(x0 + 1, Math.min(w, Math.ceil(box.x1)));
  const y1 = Math.max(y0 + 1, Math.min(h, Math.ceil(box.y1)));
  if (x1 - x0 < 1 || y1 - y0 < 1) return null;
  return { x0, y0, x1, y1 };
}

/** Crop a box out of a luma image and resample it to dw x dh. */
export function cropResizeLuma(src, sw, sh, box, dw, dh) {
  const b = clampBox(box, sw, sh);
  if (!b) return null;
  const cw = b.x1 - b.x0;
  const ch = b.y1 - b.y0;
  const out = new Uint8Array(dw * dh);
  const xr = cw / dw;
  const yr = ch / dh;
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(ch - 1, Math.max(0, (y + 0.5) * yr - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(ch - 1, y0 + 1);
    const fy = sy - y0;
    const row0 = b.y0 + y0;
    const row1 = b.y0 + y1;
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(cw - 1, Math.max(0, (x + 0.5) * xr - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(cw - 1, x0 + 1);
      const fxx = sx - x0;
      const p0 = b.x0 + x0;
      const p1 = b.x0 + x1;
      const a = src[row0 * sw + p0];
      const bb = src[row0 * sw + p1];
      const c = src[row1 * sw + p0];
      const d = src[row1 * sw + p1];
      const top = a + (bb - a) * fxx;
      const bot = c + (d - c) * fxx;
      out[y * dw + x] = Math.round(top + (bot - top) * fy);
    }
  }
  return out;
}

/**
 * Pack a single-channel luma image into the CHW float32 tensor both models take.
 *
 * The normalisation is PaddleOCR's `NormalizeImage`. The two models disagree on
 * it, and the disagreement is not cosmetic: det uses ImageNet mean/std, rec uses
 * the symmetric 0.5/0.5 that the CRNN family has always used. Feeding det the
 * ImageNet statistics against a 0.5/0.5 normalised input still produces a
 * bimodal map (measured: 3.0% of pixels hot versus 2.7%) so a wrong choice here
 * degrades quietly rather than failing — which is why both are named constants
 * and the det one is asserted by a test.
 *
 * @param {Uint8Array} luma
 * @param {number[]} mean  3 numbers
 * @param {number[]} std   3 numbers
 * @returns {Float32Array} length 3*w*h, channel-major
 */
export function toChw(luma, w, h, mean, std) {
  const out = new Float32Array(3 * w * h);
  const plane = w * h;
  for (let i = 0; i < plane; i++) {
    const v = luma[i] / 255;
    // Grayscale in, three identical channels out: the models expect RGB and the
    // document carries no colour the models were trained to use.
    out[i] = (v - mean[0]) / std[0];
    out[plane + i] = (v - mean[1]) / std[1];
    out[2 * plane + i] = (v - mean[2]) / std[2];
  }
  return out;
}

// PaddleOCR's NormalizeImage defaults, per model.
export const DET_MEAN = [0.485, 0.456, 0.406];
export const DET_STD = [0.229, 0.224, 0.225];
export const REC_MEAN = [0.5, 0.5, 0.5];
export const REC_STD = [0.5, 0.5, 0.5];

/**
 * The size the detector is fed. PaddleOCR's det resize makes the long side no
 * bigger than `limit` and rounds BOTH sides up to a multiple of 32, because the
 * DB backbone downsamples by 32 and then upsamples back to the input size.
 * @returns {{width:number, height:number}}
 */
export function detInputSize(w, h, limit) {
  const scale = Math.min(1, limit / Math.max(w, h));
  return {
    width: Math.max(32, Math.round((w * scale) / 32) * 32),
    height: Math.max(32, Math.round((h * scale) / 32) * 32),
  };
}

// ---------------------------------------------------------------------------
// DB post-processing: probability map -> text boxes
// ---------------------------------------------------------------------------

/**
 * Connected components of a thresholded binary map, 8-connected.
 *
 * DEVIATION FROM PaddleOCR: that implementation calls OpenCV's `findContours`
 * and then a rotated-rect fit. There is no OpenCV here (§6.2 forbids it) and a
 * fitted rect would need a min-area-rect routine we do not have. The document is
 * already deskewed by js/preprocess.js before it reaches this module, so the
 * axis-aligned bounding box of each component is the right shape for this
 * pipeline rather than a compromise. Each box still carries the mean detector
 * probability inside it, which is the number PaddleOCR's box filter uses.
 *
 * @param {Uint8Array} bin  0/1 per pixel
 * @param {Float32Array} prob  the detector's probability map, same shape
 * @returns {Array<{x0:number,y0:number,x1:number,y1:number,area:number,score:number}>}
 */
export function connectedComponents(bin, prob, w, h) {
  const seen = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  const out = [];
  for (let start = 0; start < bin.length; start++) {
    if (!bin[start] || seen[start]) continue;
    let sp = 0;
    stack[sp++] = start;
    seen[start] = 1;
    let x0 = w;
    let y0 = h;
    let x1 = -1;
    let y1 = -1;
    let area = 0;
    let sum = 0;
    while (sp > 0) {
      const p = stack[--sp];
      const x = p % w;
      const y = (p - x) / w;
      area++;
      sum += prob[p];
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          const q = ny * w + nx;
          if (bin[q] && !seen[q]) { seen[q] = 1; stack[sp++] = q; }
        }
      }
    }
    out.push({ x0, y0, x1: x1 + 1, y1: y1 + 1, area, score: sum / area });
  }
  return out;
}

/**
 * Expand a box outwards, PaddleOCR's `unclip`.
 *
 * DEVIATION FROM PaddleOCR: that implementation offsets a polygon with
 * pyclipper. The closed form it approximates — `distance = area * ratio /
 * perimeter` — is reproduced exactly here and applied to the four sides of the
 * axis-aligned box, which produces the same box for a rectangle and needs no
 * clipper library.
 */
export function unclipBox(box, ratio) {
  const w = box.x1 - box.x0;
  const h = box.y1 - box.y0;
  const perimeter = 2 * (w + h);
  const d = perimeter > 0 ? (w * h * ratio) / perimeter : 0;
  return { x0: box.x0 - d, y0: box.y0 - d, x1: box.x1 + d, y1: box.y1 + d };
}

/**
 * The detector's probability map to a list of text boxes, in DET-INPUT pixels.
 *
 * THE FILTERS MATTER AS MUCH AS THE GEOMETRY. A DB probability map on a
 * high-contrast page is not only clean text regions: it also carries thin
 * slivers along the paper edge and around a rule line, a few pixels on a side.
 * Measured on the receipt fixture, the unfiltered map produced 70 components for
 * 16 text lines — and because those slivers were real components with real
 * scores, they reached the recogniser and came back as lines of em-dashes that
 * wrecked the character error rate. PaddleOCR's own `min_size`, `box_thresh` and
 * `drop_score` are what remove them, so all three are here rather than only the
 * geometry they feed.
 *
 * @param {Float32Array} prob
 * @param {{thresh?:number, boxThresh?:number, unclipRatio?:number,
 *          minArea?:number, minSide?:number}} [opts]
 */
export function dbPostProcess(prob, w, h, opts) {
  const o = opts || {};
  const thresh = o.thresh !== undefined ? o.thresh : 0.3;
  const boxThresh = o.boxThresh !== undefined ? o.boxThresh : 0.6;
  const unclipRatio = o.unclipRatio !== undefined ? o.unclipRatio : 1.5;
  // Sizes are in DET-INPUT pixels; the detector runs at ~0.6x the OCR image, so
  // 6 here is ~10 pixels in the page. A text line is ~26 det pixels tall.
  const minArea = o.minArea !== undefined ? o.minArea : 30;
  const minSide = o.minSide !== undefined ? o.minSide : 6;

  const bin = new Uint8Array(w * h);
  for (let i = 0; i < bin.length; i++) bin[i] = prob[i] > thresh ? 1 : 0;

  const raw = connectedComponents(bin, prob, w, h);
  const kept = [];
  for (const c of raw) {
    if (c.area < minArea) continue;
    if (c.x1 - c.x0 < minSide || c.y1 - c.y0 < minSide) continue;
    if (c.score < boxThresh) continue;
    kept.push(unclipBox(c, unclipRatio));
  }
  // Reading order: down the page, then across. A stable sort matters because two
  // boxes on the same row must come out left-to-right, never swapped.
  kept.sort((a, b) => (a.y0 - b.y0) || (a.x0 - b.x0));
  return kept;
}

/**
 * Split one detected LINE box into WORD boxes on its blank columns.
 *
 * WHY THIS EXISTS AT ALL. The recogniser is a sequence model: handed a whole
 * text line it reads the characters correctly but drops the SPACES between words
 * when the gap is wide — measured on the receipt fixture, "GREENFIELD MARKET"
 * came back as "GREENFIELDMARKET" and "12 HIGH STREET TEL 020 7946 0011" as one
 * 27-character run. That is not cosmetic here: the app extracts fields, and
 * "GREENFIELD MARKET" is the `merchant` field. A missing space in the middle of
 * a value is a wrong value.
 *
 * The detector's map will not help — it fills a line solid. The IMAGE will: a
 * column of the line box that carries no ink is a gap between words. Columns are
 * blank when the gap is at least `minGap` wide, which is a fraction of the line
 * height rather than a pixel count so it holds at every scale. Recognising the
 * words separately and joining them with one space reproduces the spacing the
 * page actually has, and yields the word boxes the brief asks to expose.
 *
 * @param {Uint8Array} luma source pixels
 * @param {number} w @param {number} h source dimensions
 * @param {object} box the line box, source pixels
 * @param {{inkThresh?:number, gapFraction?:number, minGap?:number}} [opts]
 * @returns {Array<{x0,y0,x1,y1}>} word boxes, left to right; never empty
 */
export function splitIntoWords(luma, w, h, box, opts) {
  const o = opts || {};
  const inkThresh = o.inkThresh !== undefined ? o.inkThresh : 160;
  const b = clampBox(box, w, h);
  if (!b) return [];
  const bw = b.x1 - b.x0;
  const bh = b.y1 - b.y0;
  const fraction = o.gapFraction !== undefined ? o.gapFraction : 0.34;
  const minGap = o.minGap !== undefined ? o.minGap : Math.max(3, Math.round(fraction * bh));

  // Which columns of the box carry ink at all.
  const inked = new Uint8Array(bw);
  for (let x = 0; x < bw; x++) {
    const px = b.x0 + x;
    for (let y = b.y0; y < b.y1; y++) {
      if (luma[y * w + px] < inkThresh) { inked[x] = 1; break; }
    }
  }

  // Runs of blank columns at least minGap wide become the cuts, placed at the
  // centre of the run so neither neighbouring word keeps the gap as margin.
  const cuts = [];
  let runStart = -1;
  for (let x = 0; x < bw; x++) {
    if (!inked[x]) { if (runStart < 0) runStart = x; continue; }
    if (runStart >= 0) {
      if (x - runStart >= minGap) cuts.push(Math.floor((runStart + x) / 2));
      runStart = -1;
    }
  }
  if (runStart >= 0 && bw - runStart >= minGap) cuts.push(bw);

  const words = [];
  let from = 0;
  for (const c of cuts) {
    if (c > from) words.push({ x0: b.x0 + from, y0: b.y0, x1: b.x0 + c, y1: b.y1 });
    from = c;
  }
  if (from < bw) words.push({ x0: b.x0 + from, y0: b.y0, x1: b.x1, y1: b.y1 });

  // Trim each word box to the ink it actually contains, so a word crop is not
  // mostly blank margin (which the recogniser reads as a leading/trailing space).
  const trimmed = [];
  for (const wd of words) {
    let x0 = wd.x1, x1 = wd.x0;
    for (let x = wd.x0; x < wd.x1; x++) {
      if (inked[x - b.x0]) { if (x < x0) x0 = x; if (x > x1) x1 = x; }
    }
    if (x1 >= x0) trimmed.push({ x0, y0: wd.y0, x1: x1 + 1, y1: wd.y1 });
  }
  return trimmed.length ? trimmed : [b];
}

/**
 * Group boxes into text LINES.
 *
 * The detector is trained to emit one region per text line, so for a normal page
 * this is a no-op. It matters when a line has a wide gap in it (a receipt's
 * `MILK 2L          1.85`), because DB sometimes splits that into two regions —
 * and two regions that belong on one line must be read left-to-right on one line,
 * not as two separate lines that interleave with the row below.
 *
 * Two boxes are on the same line when their vertical spans overlap by more than
 * half the shorter one.
 *
 * @param {Array} boxes axis-aligned, any order
 * @returns {Array<Array>} lines, each sorted left-to-right, lines top-to-bottom
 */
export function groupIntoLines(boxes) {
  const sorted = [...boxes].sort((a, b) => (a.y0 - b.y0) || (a.x0 - b.x0));
  const lines = [];
  for (const b of sorted) {
    const h = b.y1 - b.y0;
    let target = null;
    for (const line of lines) {
      const lh = line.bottom - line.top;
      const overlap = Math.min(line.bottom, b.y1) - Math.max(line.top, b.y0);
      if (overlap > 0.5 * Math.min(lh, h)) { target = line; break; }
    }
    if (target) {
      target.boxes.push(b);
      target.top = Math.min(target.top, b.y0);
      target.bottom = Math.max(target.bottom, b.y1);
    } else {
      lines.push({ top: b.y0, bottom: b.y1, boxes: [b] });
    }
  }
  for (const line of lines) line.boxes.sort((a, b) => a.x0 - b.x0);
  lines.sort((a, b) => a.top - b.top);
  return lines.map((l) => l.boxes);
}

// ---------------------------------------------------------------------------
// CTC decoding
// ---------------------------------------------------------------------------

/**
 * Build the CTC charset from PaddleOCR's dictionary file.
 *
 * Index 0 is the CTC BLANK and is not a character; the space is appended after
 * the dictionary's own characters (PaddleOCR's `use_space_char`). Getting the
 * offset wrong shifts every recognised character by one and still looks like
 * fluent text, so this is asserted by a test rather than trusted.
 *
 * @param {string} dictText the contents of ppocrv5_dict.txt
 */
export function buildCharset(dictText) {
  const chars = String(dictText).split('\n').map((l) => l.replace(/\r$/, ''));
  // A trailing newline is not a character.
  if (chars.length && chars[chars.length - 1] === '') chars.pop();
  return [''].concat(chars, [' ']);
}

/**
 * Greedy CTC decode of a [T,C] logit matrix — PaddleOCR's `CTCLabelDecode`.
 *
 * The confidence is the mean of the winning softmax probability over the KEPT
 * timesteps only (not over every timestep). Averaging over all T would divide
 * the score by the number of blank frames, so a short word in a wide crop would
 * score low for being short — and the app decides whether to SPEAK a value on
 * exactly this number, so it has to mean "how sure the model was about the
 * characters it emitted".
 *
 * @param {Float32Array} data T*C logits
 * @param {number} t number of timesteps
 * @param {number} c number of classes
 * @param {string[]} charset index -> character
 * @returns {{text:string, confidence:number, kept:number}}
 */
export function ctcDecode(data, t, c, charset) {
  let prev = 0;
  const out = [];
  let confSum = 0;
  let kept = 0;
  for (let i = 0; i < t; i++) {
    const base = i * c;
    let best = 0;
    let bestV = -Infinity;
    let rowSum = 0;
    let nonNegative = true;
    for (let k = 0; k < c; k++) {
      const v = data[base + k];
      if (v > bestV) { bestV = v; best = k; }
      rowSum += v;
      if (v < 0) nonNegative = false;
    }
    // THE EXPORTED MODEL ALREADY ENDS IN A SOFTMAX. Applying a second one would
    // divide an already-normalised 0.93 by the sum of ~11,000 exponentials and
    // report every character as 0.01 confident — which reads as "the engine is
    // guessing" and would make the speak-or-omit gate refuse to ever speak a
    // value. So: use the row as-is when it is a probability distribution, and
    // only fall back to softmax for a genuine (possibly negative) logit row.
    const isProb = nonNegative && rowSum > 0.5 && rowSum < 1.5;
    let p;
    if (isProb) {
      p = bestV;
    } else {
      let sumExp = 0;
      for (let k = 0; k < c; k++) sumExp += Math.exp(data[base + k] - bestV);
      p = sumExp > 0 ? 1 / sumExp : 0;
    }
    if (best !== 0 && best !== prev) {
      out.push(charset[best] !== undefined ? charset[best] : '');
      confSum += p;
      kept++;
    }
    prev = best;
  }
  return {
    text: out.join(''),
    confidence: kept > 0 ? confSum / kept : 0,
    kept,
  };
}

/** The width a detected line is resampled to before recognition. */
export function recWidthFor(boxW, boxH, recHeight, maxWidth) {
  const raw = Math.round((boxW * recHeight) / Math.max(1, boxH));
  return Math.max(8, Math.min(maxWidth, raw));
}

export default {
  resizeLuma, clampBox, cropResizeLuma, toChw, detInputSize,
  connectedComponents, unclipBox, dbPostProcess, groupIntoLines,
  buildCharset, ctcDecode, recWidthFor,
  DET_MEAN, DET_STD, REC_MEAN, REC_STD,
};
