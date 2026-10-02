/**
 * Page layout detection: find text LINES in a camera photo.
 *
 * WHY THIS EXISTS
 * ---------------
 * Measured on real hand-photographed receipts, the detector finds 62.6% of the
 * text lines. That is a hard ceiling: line accuracy cannot exceed it no matter
 * how good the recogniser is. Detection, not recognition, is the current
 * bottleneck, which is the opposite of what we assumed when we started.
 *
 * WHAT WAS TRIED AND REJECTED, because the measurements are the useful part
 * ------------------------------------------------------------------------
 * This was going to be slice-based. The reasoning was sound: on a tilted page
 * two vertically-overlapping lines merge in a global row projection, so detect
 * bands independently in three column slices and union them. Three variants
 * were implemented and all three measured WORSE than the plain global
 * projection, over 20 real pages / 405 transcript lines:
 *
 *   3-of-3 slice vote        bands/line 0.664
 *   union of slice bands     bands/line 0.393
 *   plain global projection   bands/line 0.790   <- kept
 *
 * Why slicing loses: a short line (an item name, a right-aligned price)
 * occupies ONE slice, so requiring agreement throws it away; and unioning
 * re-merges exactly what the slices were meant to separate, because adjacent
 * slices produce bands a pixel apart and any sensible tolerance joins them
 * back into one 141px block. 31 bands taller than 90px appeared.
 *
 * So the row threshold is 0.04 because it was swept, not chosen:
 *   0.02 -> 0.73   0.04 -> 0.79   0.06 -> 0.75   0.08 -> 0.61   0.12 -> 0.31
 *
 * THE TWO THINGS THAT WERE ACTUALLY WRONG, AND ARE FIXED HERE
 * -----------------------------------------------------------
 * 1. Absolute pixel thresholds. Band limits and the median radius now scale
 *    with image height (scaleFor). Pages in this dataset span 258px to 1000px;
 *    a fixed 8px minimum band rejects every line on the short ones, and a 31px
 *    median window covers 12% of a 258px image versus 3% of a 1000px one.
 *    Measured before the fix: bands/line of 0.19 on a 348px page against 1.00
 *    on a 1000px page.
 * 2. The median radius must be ODD. h*0.03 rounds even often enough to matter
 *    (a 348px page gives 10), and cv2.medianBlur asserts ksize % 2 == 1.
 *
 * WHAT IS STILL WRONG, HONESTLY
 * -----------------------------
 * 1002-receipt has a row profile that never falls below 6% ink, so no
 * threshold separates its 26 lines; it yields 2 bands at 0.02 and 14 at 0.08.
 * A single global threshold cannot serve both that page and the easy ones.
 * Fixing this properly needs a learned detector or an adaptive projection
 * peak-finder, not another constant -- and per-page Otsu was tried and is also
 * worse (0.69 aggregate), because it moves the threshold the wrong way.
 *
 * No neural network here on purpose: this runs before the recogniser on a
 * phone with no GPU, so a deterministic, debuggable detector is the right
 * trade. It is also the honest thing to report -- we can say exactly which
 * pages break it.
 *
 * Mirrors the Python evaluator's detector so training and evaluation see the
 * same preprocessing. If these diverged, the evaluation would be measuring a
 * detector the app never ships.
 */

/** Median blur, radius in pixels. Separates local paper from local ink. */
const MEDIAN_R = 31;

/** How much darker than local paper a pixel must be to count as ink. */
const INK_DELTA = 18;

/** Column slices used for independent detection. More = less vertical merge. */
/** Fraction of page width a row must fill with ink to count as text. */
const ROW_INK_THRESHOLD = 0.04;

// Band height limits are NOT constants: they scale with image height via
// scaleFor(), because a fixed 8px minimum rejects every line on a 258px page
// while looking generous on a 1000px one. See scaleFor() for the measurement
// that forced this.

/**
 * Separable-ish median approximation.
 *
 * A true median filter is O(n*k*log k); this uses a histogram over 256 bins,
 * which is O(n) and exact for 8-bit input. Returns Uint8Array.
 */
export function medianBlur(gray, w, h, radius) {
  const out = new Uint8Array(gray.length);
  const hist = new Int32Array(256);
  const need = radius * 2 + 1;
  const half = need >> 1;
  for (let y = 0; y < h; y++) {
    hist.fill(0);
    for (let x = -radius; x <= radius; x++) {
      const xi = x < 0 ? 0 : (x >= w ? w - 1 : x);
      hist[gray[y * w + xi]]++;
    }
    for (let x = 0; x < w; x++) {
      // median = value where the cumulative count passes half
      let acc = 0, med = 0;
      for (let v = 0; v < 256; v++) {
        acc += hist[v];
        if (acc >= (need >> 1) + 1) { med = v; break; }
      }
      out[y * w + x] = med;
      // slide
      const xo = x - radius;
      const xi = x + radius + 1;
      const xoC = xo < 0 ? 0 : (xo >= w ? w - 1 : xo);
      const xiC = xi < 0 ? 0 : (xi >= w ? w - 1 : xi);
      hist[gray[y * w + xoC]]--;
      hist[gray[y * w + xiC]]++;
    }
  }
  return verticalMedian(out, w, h, radius);
}

/** Vertical pass; horizontal median from the row pass. */
function verticalMedian(src, w, h, radius) {
  const out = new Uint8Array(src.length);
  const hist = new Int32Array(256);
  const need = radius * 2 + 1;
  for (let x = 0; x < w; x++) {
    hist.fill(0);
    for (let y = -radius; y <= radius; y++) {
      const yi = y < 0 ? 0 : (y >= h ? h - 1 : y);
      hist[src[yi * w + x]]++;
    }
    for (let y = 0; y < h; y++) {
      let acc = 0, med = 0;
      for (let v = 0; v < 256; v++) {
        acc += hist[v];
        if (acc >= (need >> 1) + 1) { med = v; break; }
      }
      out[y * w + x] = med;
      const yo = y - radius;
      const yi = y + radius + 1;
      const yoC = yo < 0 ? 0 : (yo >= h ? h - 1 : yo);
      const yiC = yi < 0 ? 0 : (yi >= h ? h - 1 : yi);
      hist[src[yoC * w + x]]--;
      hist[src[yiC * w + x]]++;
    }
  }
  return out;
}

/**
 * Ink mask: 1 where the pixel is meaningfully darker than local paper.
 * Uint8Array of 0/1.
 */
export function inkMask(gray, w, h, medianRadius = MEDIAN_R, delta = INK_DELTA) {
  const paper = medianBlur(gray, w, h, medianRadius);
  const ink = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    ink[i] = gray[i] < paper[i] - delta ? 1 : 0;
  }
  return ink;
}

/** Row runs where `pred` is true, filtered to plausible heights. */
function runs(pred, minH, maxH) {
  const bands = [];
  let start = -1;
  for (let i = 0; i <= pred.length; i++) {
    const on = i < pred.length && pred[i];
    if (on && start < 0) start = i;
    else if (!on && start >= 0) {
      const hgt = i - start;
      if (hgt >= minH && hgt <= maxH) bands.push([start, i]);
      start = -1;
    }
  }
  return bands;
}

/**
 * Everything below is expressed as a FRACTION OF IMAGE HEIGHT, not in pixels.
 *
 * That change is not cosmetic. Measured over 20 real pages whose heights span
 * 258px to 1000px, absolute pixel thresholds produced bands/line ratios of
 * 0.19 and 0.41 on the short pages against 1.00 on the tall ones -- a detector
 * that only works on one capture distance. The small images were losing lines
 * because a fixed 8px minimum band height rejects a 6px line, and a 31px
 * median window covers a third of the page.
 *
 * A median filter radius of 31 is also 12% of a 258px image but 3% of a 1000px
 * one, so it was measuring "local paper" at two completely different scales.
 */
export function scaleFor(h) {
  // Text on a receipt is roughly 2-4% of page height; band limits bracket that
  // generously. Median window ~3% of height, clamped to something sane.
  //
  // MUST BE ODD. cv2.medianBlur asserts ksize % 2 == 1, and h*0.03 rounds to
  // an even number often enough to matter: a 348px page gave 10, which raised
  // cv2.error in the Python mirror of this code. An even window is also
  // mathematically ambiguous for a median, so requiring odd is not merely a
  // library quirk.
  let median = Math.max(5, Math.min(31, Math.round(h * 0.03)));
  if (median % 2 === 0) median -= 1;
  return {
    median,
    delta: 18,
    minBandH: Math.max(3, Math.round(h * 0.012)),   // ~1.2% of height
    maxBandH: Math.round(h * 0.13),                  // a merged block is far taller
  };
}

/**
 * Detect text lines. Returns [{y0, y1, x0, x1, ink, conf}].
 *
 * Each box is expanded to the full ink extent horizontally so the recogniser
 * sees the whole line rather than the slice that detected it.
 */
export function detectLines(gray, w, h, opts = {}) {
  const S = scaleFor(h);
  const minH = opts.minBandH ?? S.minBandH;
  const maxH = opts.maxBandH ?? S.maxBandH;
  // Row ink is measured as a FRACTION OF WIDTH, then thresholded at a fixed
  // fraction. Both the fraction and the threshold were swept over 20 real
  // pages (405 transcript lines): 0.02 -> 0.73, 0.04 -> 0.79, 0.06 -> 0.75,
  // 0.08 -> 0.61, 0.12 -> 0.31. 0.04 wins, so it is not a guess.
  const thr = opts.rowThreshold ?? ROW_INK_THRESHOLD;
  const ink = opts.ink || inkMask(gray, w, h, opts.medianRadius ?? S.median,
                                  opts.delta ?? S.delta);

  const rows = new Uint8Array(h);
  for (let y = 0; y < h; y++) {
    let n = 0;
    const base = y * w;
    for (let x = 0; x < w; x++) n += ink[base + x];
    rows[y] = (n / w) >= thr ? 1 : 0;
  }

  const bands = runs(Array.from(rows), minH, maxH);

  // Expand each band to its horizontal ink extent.
  const out = [];
  for (const [y0, y1] of bands) {
    let lo = -1, hi = -1, count = 0;
    for (let y = y0; y < y1; y++) {
      const base = y * w;
      for (let x = 0; x < w; x++) {
        if (ink[base + x]) {
          count++;
          if (lo < 0) lo = x;
          hi = x;
        }
      }
    }
    // A band seen by every slice spans the full page width; one seen by a
    // single slice may be a fragment that the horizontal expansion below will
    // stretch across the whole line. Both are kept -- dropping fragments is
    // exactly the failure that made the vote-threshold version worse.
    if (lo < 0 || hi - lo < 4) continue;
    const pad = Math.max(2, Math.floor((y1 - y0) * 0.10));
    const total = (y1 - y0) * w;
    out.push({
      y0, y1,
      x0: Math.max(0, lo - pad),
      x1: Math.min(w, hi + pad + 1),
      ink: count / total,
      conf: Math.min(1, (count / total) / 0.10),
    });
  }
  return out;
}

/**
 * Convert one detected box to the crop array the recogniser consumes.
 *
 * SCALES PROPORTIONALLY, THEN PADS. Matches build_handheld.prepare(): a crop
 * is fitted inside (maxW, 32) with a uniform scale and padded afterwards.
 * Resizing to a fixed 32-row strip while keeping full width, then truncating,
 * squashes characters horizontally -- and worse, labels the full line against
 * the pixels of only its first 320 columns.
 */
export function cropForRecognizer(gray, w, box, cropH = 32, maxW = 320) {
  const bh = box.y1 - box.y0;
  const bw = box.x1 - box.x0;
  if (bh <= 0 || bw <= 0) return { data: new Uint8Array(cropH * maxW), width: maxW };

  const scale = Math.min(maxW / bw, cropH / bh);
  const nw = Math.max(1, Math.min(maxW, Math.round(bw * scale)));
  const nh = Math.max(1, Math.min(cropH, Math.round(bh * scale)));

  const out = new Uint8Array(cropH * nw);
  // Box filter down to nw x nh, averaging the source footprint of each dest px.
  for (let dy = 0; dy < nh; dy++) {
    const y0 = box.y0 + Math.floor((dy * bh) / nh);
    const y1 = box.y0 + Math.max(y0 + 1 - box.y0, Math.floor(((dy + 1) * bh) / nh));
    for (let dx = 0; dx < nw; dx++) {
      const x0 = box.x0 + Math.floor((dx * bw) / nw);
      const x1 = box.x0 + Math.max(x0 + 1 - box.x0, Math.floor(((dx + 1) * bw) / nw));
      let sum = 0, n = 0;
      for (let y = y0; y < Math.min(y1, box.y1) && y < y0 + bh; y++) {
        for (let x = x0; x < Math.min(x1, box.x1); x++) {
          sum += gray[y * w + x]; n++;
        }
      }
      out[dy * nw + dx] = n ? Math.round(sum / n) : 0;
    }
  }
  return { data: out, width: nw };
}

const SightLineLayout = {
  MEDIAN_R, INK_DELTA, ROW_INK_THRESHOLD,
  medianBlur, inkMask, detectLines, cropForRecognizer, scaleFor,
};
if (typeof window !== "undefined") window.SightLineLayout = SightLineLayout;
export default SightLineLayout;