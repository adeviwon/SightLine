/**
 * SightLine — degradation-aware restoration gating (JavaScript port).
 *
 * Mirrors ml/src/gating.py exactly. The anchors and the reasoning are
 * documented there; the short version:
 *
 *   The trained restorer is UNCONDITIONAL. Measured on the training corpus it
 *   is excellent on damaged input (median +5.3 dB vs identity on badly
 *   degraded patches) and destructive on clean input (-48 dB, because it
 *   adds its own artefacts to an already-perfect image).
 *
 *   A clean scan is the demo case, so "do nothing to clean input" is not
 *   optional. This module measures how degraded the input is and returns a
 *   blend weight: 0.0 passes the original through untouched, 1.0 uses the
 *   full restoration.
 *
 *   SCALE WARNING (this cost real debugging time): variance-of-Laplacian is a
 *   PER-PIXEL statistic, so the same content measures ~3.5x lower on a
 *   64x256 patch than on a full 900x580 render. The anchors below were
 *   calibrated on 64x256 patches. If you tile at a different size, re-measure
 *   with `bash run.sh gating` and update them, or the gate silently sends
 *   clean images through full restoration — the exact bug it prevents.
 */

const Gating = (() => {
  // Measured on 64x256 patches (see ml/src/gating.py for the full table and
  // for how these were derived). These MUST equal the Python constants:
  //   SHARP_CLEAN 0.080, SHARP_BAD 0.008, CONTENT_MIN 0.12
  // `node tools/gating_parity.js` fails if they ever drift apart — a silent
  // drift here means clean scans get restored on a phone and nobody notices.
  const SHARP_CLEAN = 0.080;  // sharpness >= this -> weight 0
  const SHARP_BAD = 0.008;    // sharpness <= this -> weight 1
  const CONTENT_MIN = 0.12;   // below this dynamic range: blank paper

  // The scale the anchors above were calibrated at. Must equal TILE_H/TILE_W
  // in restorer.js, which is what the ONNX model consumes.
  const TILE_H = 64;
  const TILE_W = 256;

  /** Grayscale Float32Array in [0,1] from an RGBA ImageData-like object. */
  function toGray(data, w, h) {
    const out = new Float32Array(w * h);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      out[p] = (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) / 255;
    }
    return out;
  }

  /**
   * Variance of the Laplacian, matching cv2.Laplacian(gray, CV_32F, ksize=3)
   * which uses the 4-neighbour kernel:
   *     [ 0  1  0 ]
   *     [ 1 -4  1 ]
   *     [ 0  1  0 ]
   * cv2.BORDER_REPLICATE at the edges.
   */
  function sharpness(g, w, h) {
    let sum = 0, sumSq = 0, n = 0;
    const at = (x, y) => g[Math.min(h - 1, Math.max(0, y)) * w +
                            Math.min(w - 1, Math.max(0, x))];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = 4 * at(x, y) - at(x - 1, y) - at(x + 1, y) -
                  at(x, y - 1) - at(x, y + 1);
        sum += v; sumSq += v * v; n++;
      }
    }
    const mean = sum / n;
    return sumSq / n - mean * mean;
  }

  /** Dynamic range (max - min). A blank patch has no structure to recover. */
  function contrast(g) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < g.length; i++) {
      if (g[i] < lo) lo = g[i];
      if (g[i] > hi) hi = g[i];
    }
    return hi - lo;
  }

  /** 0.0 at/above clean, 1.0 at/below bad. Sharpness LOW is what is bad. */
  function rampDown(v, clean, bad) {
    if (v >= clean) return 0;
    if (v <= bad) return 1;
    return (clean - v) / (clean - bad);
  }

  /**
   * Restoration weight for one tile, from its grayscale data.
   * Returns 0 for blank tiles and for undamaged tiles.
   */
  function weightFromGray(g, w, h) {
    if (contrast(g) < CONTENT_MIN) return 0;
    return rampDown(sharpness(g, w, h), SHARP_CLEAN, SHARP_BAD);
  }

  /**
   * Convenience: build from RGBA ImageData, measured at TILE scale.
   *
   * SCALE IS NOT OPTIONAL. See the SCALE WARNING at the top of this file.
   * Variance-of-Laplacian is a per-pixel statistic, so a full 900x580 render
   * measures ~3.5x LOWER than a 64x256 patch of the same content. Feeding a
   * whole image to weightFromGray() with patch-calibrated anchors produced:
   *
   *     profile          full-image weight   correct (per-tile)
   *     studio_clean            0.43               0.00
   *     off_axis                0.55               0.00
   *     glossy_glare            0.66               0.00
   *     paper_texture           0.52               0.00
   *
   * i.e. every clean profile was sent through full restoration, which costs
   * about -56 dB PSNR on clean input and measurably changed the OCR text the
   * classifier saw. That was the live behaviour of the shipped app.
   *
   * The fix is to measure the unit the model actually consumes: slice the image
   * into TILE_H x TILE_W patches and take the MEDIAN weight. Median rather than
   * mean so that a few blank margins (weight 0 by CONTENT_MIN) do not dilute a
   * genuinely blurred page, and so one noisy tile cannot force restoration of a
   * clean page.
   */
  function weightFromImageData(imgData) {
    const g = toGray(imgData.data, imgData.width, imgData.height);
    const W = imgData.width, H = imgData.height;
    if (W <= TILE_W && H <= TILE_H) return weightFromGray(g, W, H);

    const ws = [];
    for (let y = 0; y + TILE_H <= H; y += TILE_H) {
      for (let x = 0; x + TILE_W <= W; x += TILE_W) {
        // Extract this tile into a contiguous buffer.
        const tile = new Float32Array(TILE_W * TILE_H);
        for (let r = 0; r < TILE_H; r++) {
          const src = (y + r) * W + x;
          tile.set(g.subarray(src, src + TILE_W), r * TILE_W);
        }
        ws.push(weightFromGray(tile, TILE_W, TILE_H));
      }
    }
    if (!ws.length) return weightFromGray(g, W, H);
    ws.sort((a, b) => a - b);
    const mid = ws.length >> 1;
    return ws.length % 2 ? ws[mid] : (ws[mid - 1] + ws[mid]) / 2;
  }

  /**
   * Blend restored with original.
   * weight 0 -> original untouched; weight 1 -> restored.
   * Gamma 0.7 spends more of the range near the damaged end, where the
   * model's correction is largest and most valuable.
   */
  function blend(original, restored, weight) {
    if (weight <= 0) return original;
    if (weight >= 1) return restored;
    const a = Math.pow(weight, 0.7);
    const n = Math.min(original.length, restored.length);
    const out = new Float32Array(original.length);
    for (let i = 0; i < n; i++) out[i] = a * restored[i] + (1 - a) * original[i];
    return out;
  }

  function describe(imgData) {
    const g = toGray(imgData.data, imgData.width, imgData.height);
    const s = sharpness(g, imgData.width, imgData.height);
    const w = weightFromGray(g, imgData.width, imgData.height);
    return {
      sharpness: s,
      contrast: contrast(g),
      weight: w,
      band: w < 0.15 ? "clean" : w < 0.45 ? "mild" :
            w < 0.75 ? "moderate" : "severe",
    };
  }

  return { toGray, sharpness, contrast, weightFromGray, weightFromImageData,
           blend, describe, rampDown,
           TILE_H, TILE_W,
           ANCHORS: { SHARP_CLEAN, SHARP_BAD, CONTENT_MIN } };
})();

if (typeof module !== "undefined" && module.exports) module.exports = Gating;
if (typeof globalThis !== "undefined") globalThis.SightLineGating = Gating;
if (typeof window !== "undefined") window.SightLineGating = Gating;
