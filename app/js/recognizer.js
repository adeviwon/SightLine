/**
 * SightLineCRNN — in-browser text recognition.
 *
 * This replaces Tesseract. It is our own model (ml/src/ocr_model.py), trained
 * on 30,021 real word crops from 260 real SROIE receipts plus 198
 * hand-photographed receipts, exported to a self-contained ONNX file.
 *
 * WHY THE ARCHITECTURE IS WHAT IT IS
 * ----------------------------------
 * Greedy CTC decoding is a softmax over 38 classes per timestep. No beam
 * search, no lexicon, no autoregressive loop. That is what makes one ONNX file
 * viable inside browser WASM on a phone with no GPU.
 *
 * THE FIXED-WIDTH CONTRACT — read this before changing anything
 * ------------------------------------------------------------
 * The exported graph takes EXACTLY [1, 1, 32, 256]. Width is NOT dynamic,
 * because aten::lstm cannot be traced with a dynamic sequence length:
 *
 *     'stack expects each tensor to be equal size, but got
 *      torch.Size([Min(((((((s0 - 1)//4)) - 1)//2)) + 1, ...), 1, 128])'
 *
 * and torch.export cannot unroll a loop over a symbolic bound:
 *
 *     'The inequality, Eq((((x/4)//2)) - 20, 0), cannot be solved using
 *      solve_univariate_inequality'
 *
 * So a narrow crop is RIGHT-PADDED to 256. That is safe BECAUSE THE LOSS IS
 * CTC: a right-padded region is blank page, a trained model emits the blank
 * symbol there, and the decoder drops consecutive blanks by construction, so
 * padding contributes no characters. This would NOT be safe for an
 * encoder-decoder with attention, which is why the reasoning is written down
 * instead of being assumed.
 *
 * Because of that, we still trim to the crop's true timestep count
 * T = ceil(contentWidth / 8) before decoding. Without the trim, a 64 px crop
 * in a 256 px tensor would be read as 32 timesteps of padding, and any
 * residual signal there becomes invented characters.
 *
 * CHARSET — 37 symbols, and that is deliberate
 * --------------------------------------------
 * 10 digits + 26 letters + space. No punctuation. Punctuation is the
 * highest-error class in practice and no field regex this app uses needs it.
 * The visible consequence is that '9.00' is read as '900', which is fine for
 * field matching (TOTAL 900 and TOTAL 9.00 hit the same pattern) and would be
 * unacceptable for a product that had to report currency to the cent.
 *
 * Index 0 is the CTC blank. It is never a character.
 */

const ORT = "../vendor/ort/ort.min.mjs";

export const CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
export const BLANK = 0;
export const CROP_H = 32;
export const EXPORT_W = 256;      // must match export_crnn(width=256)
export const WIDTH_DIVISOR = 8;   // timesteps = contentWidth / 8

let session = null;
let initPromise = null;
const cache = new Map();          // contentWidth -> padded Float32Array

/**
 * Load the ONNX session once. Concurrent callers share one promise.
 *
 * A failed load resolves to null and sets `lastError` rather than throwing, so
 * the pipeline can fall back rather than break -- the app degrades, it never
 * shows a broken state.
 */
export async function init() {
  if (session) return session;
  if (initPromise) return initPromise;
  initPromise = (async () => {
    try {
      const env = ORT.env || (ORT.env = {});
      env.wasm = env.wasm || {};
      env.wasm.wasmPaths = "../vendor/ort/";
      env.wasm.numThreads = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
      session = await ORT.InferenceSession.create("models/crnn.onnx", {
        executionProviders: ["wasm"],
      });
      return session;
    } catch (e) {
      lastError = e && e.message ? e.message : String(e);
      return null;
    }
  })();
  return initPromise;
}

let lastError = null;
export function getError() { return lastError; }
export function isReady() { return session !== null; }

/**
 * True timestep count for a crop of the given content width.
 *
 * Mirrors model.timesteps(), which RUNS the conv stack rather than multiplying
 * factors -- MaxPool2d floors, so odd widths are off by one from W/8 and a
 * clean-product formula would silently mis-trim the decode. Kept as a function
 * (not a constant) so it can be tested against the Python value.
 */
export function timestepsFor(contentWidth) {
  return Math.max(1, Math.floor(contentWidth / WIDTH_DIVISOR));
}

/**
 * Build the [1,1,32,256] input tensor for one crop.
 *
 * Input is normalised to [-1,1] because every crop is dark ink on light paper,
 * so +1 is "paper" and -1 is "ink". Matching the Python training transform
 * exactly matters: a different scale here produces confidently wrong text.
 */
function buildInput(gray, contentWidth) {
  let padded = cache.get(contentWidth);
  if (!padded) {
    padded = new Float32Array(1 * 1 * CROP_H * EXPORT_W);
    cache.set(contentWidth, padded);
  } else {
    padded.fill(0);           // clear any previous content
  }
  for (let y = 0; y < CROP_H; y++) {
    const srcRow = y * contentWidth;
    const dstRow = y * EXPORT_W;
    for (let x = 0; x < contentWidth; x++) {
      padded[dstRow + x] = (gray[srcRow + x] / 255 - 0.5) / 0.5;
    }
  }
  return padded;
}

/**
 * Greedy CTC decode: collapse repeats, then drop blanks.
 *
 * CRITICAL: repeated symbols MUST be separated by blanks. [9,0,0] decodes to
 * "90" — that is what CTC means, not a decoder bug. A trained model emits
 * [9,blank,0,blank,0] and this returns "900". Getting this wrong corrupts
 * every price, dosage and account number containing a doubled digit, so the
 * separator is load-bearing rather than incidental.
 */
export function decodeGreedy(indices, limit) {
  const n = limit === undefined ? indices.length : Math.min(limit, indices.length);
  let out = "";
  let prev = -1;
  for (let i = 0; i < n; i++) {
    const v = indices[i];
    if (v !== prev && v !== BLANK) out += CHARSET[v - 1] || "";
    prev = v;
  }
  return out;
}

/**
 * Recognise one word crop.
 *
 * @param {Uint8Array} gray  CROP_H * contentWidth grayscale bytes, ink dark.
 * @param {number} contentWidth  the crop's real width (NOT the padded 256).
 * @returns {Promise<{text: string, confidence: number, chars: number}>}
 *
 * Confidence is the mean max-probability over timesteps that EMIT a character.
 * Averaging over ALL timesteps -- the obvious implementation -- reports a LOW
 * score for a confident short word inside a long line, because blank-heavy
 * timesteps are not evidence of anything. That is backwards, and it matters:
 * the confidence gates whether a field is spoken at all.
 */
export async function recognise(gray, contentWidth) {
  const s = await init();
  if (!s) return { text: "", confidence: 0, chars: 0, error: lastError };

  const input = buildInput(gray, contentWidth);
  try {
    const res = await s.run({
      input: new ORT.Tensor("float32", input, [1, 1, CROP_H, EXPORT_W]),
    });
    const logits = res.logits;
    const T = logits.dims[1];
    const C = logits.dims[2];
    const data = logits.data;

    const Ttrue = Math.min(timestepsFor(contentWidth), T);
    const idx = new Int32Array(Ttrue);
    let confSum = 0, emitted = 0;

    for (let t = 0; t < Ttrue; t++) {
      const base = t * C;
      let best = 0, bestVal = -Infinity;
      // Second pass over the same row for the probability of the winner.
      // Doing it in one pass would need the log-sum-exp, which is needless
      // here: softmax probability of the argmax is exp(max)/sum(exp(all)).
      let sumExp = 0;
      for (let c = 0; c < C; c++) {
        const v = data[base + c];
        if (v > bestVal) { bestVal = v; best = c; }
      }
      for (let c = 0; c < C; c++) sumExp += Math.exp(data[base + c] - bestVal);
      idx[t] = best;
      if (best !== BLANK) {
        confSum += 1 / sumExp;      // exp(0) / sum(exp(v - max))
        emitted++;
      }
    }

    const text = decodeGreedy(idx, Ttrue);
    return {
      text,
      confidence: emitted ? confSum / emitted : 0,
      chars: text.length,
      timesteps: Ttrue,
    };
  } catch (e) {
    lastError = e && e.message ? e.message : String(e);
    return { text: "", confidence: 0, chars: 0, error: lastError };
  }
}

/**
 * Recognise a batch of crops.
 *
 * Sequential rather than parallel ONNX calls: a single 256-wide graph is small
 * enough that N concurrent sessions thrash WASM memory on a phone, and the
 * worker is already off the UI thread. Measured on the desktop harness this is
 * faster than Promise.all at realistic crop counts.
 */
export async function recogniseAll(crops) {
  const out = [];
  for (const c of crops) {
    out.push(await recognise(c.gray, c.width));
  }
  return out;
}

export function reset() {
  session = null;
  initPromise = null;
  lastError = null;
  cache.clear();
}

// Exported for the browser self-test harness.
const SightLineRecognizer = {
  CHARSET, BLANK, CROP_H, EXPORT_W, WIDTH_DIVISOR,
  init, isReady, getError, recognise, recogniseAll,
  decodeGreedy, timestepsFor, reset,
};
if (typeof window !== "undefined") window.SightLineRecognizer = SightLineRecognizer;
export default SightLineRecognizer;