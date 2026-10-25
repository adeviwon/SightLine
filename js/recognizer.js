import { beamDecode, WordTrie, logSoftmaxRow } from "./word_beam.js";

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
 * THE WIDTH CONTRACT — read this before changing anything
 * -------------------------------------------------------
 * The graph's width axis decides how this module builds its tensor, and it is
 * READ FROM THE SESSION at init (engineWidth()) instead of being assumed:
 *
 *   fixed width N      build the tensor and the row stride at N, right-pad
 *                      shorter crops, trim the decode to
 *                      Ttrue = min(timestepsFor(contentWidth), T).
 *   symbolic ('width') build them at contentWidth and feed the crop at its
 *                      NATURAL width — no padding at all — and decode all T.
 *                      T already IS this crop's timestep count, so the trim has
 *                      nothing to mask; it must NOT be reused as one either:
 *                      timestepsFor FLOORS w/8 while the conv stack gives
 *                      ceil(w/8) at some widths (T=6 at w=47, 8 at w=63, against
 *                      floor(47/8)=5, floor(63/8)=7), so masking the dynamic
 *                      path would drop a real timestep and read short.
 *
 * BOTH modes live here at once on purpose, because the fix is meant to be a
 * FILE REPLACEMENT, not a code patch. app/models/crnn.onnx is fixed-320 today
 * and stays fixed-320 until the running retrain finishes: models/ocr/crnn.pt is
 * rewritten every epoch, so the dynamic graph must be re-exported from the
 * FINAL checkpoint. `artifacts/crnn_dynamic.onnx` is a proof artifact — copying
 * it over app/models/crnn.onnx is a separate, later action, and nothing in this
 * module depends on it having happened.
 *
 * WHY THE WIDTH IS NOT FIXED FOREVER, AND WHY PADDING IS NOT SELF-CORRECTING
 * 320 IS THE TRAINING CLAMP AND IT IS NOT NEGOTIABLE. build_dataset.crop_to_array
 * builds every training crop at its natural width CLAMPED to MAX_W = 320, and
 * the corpus median IS that clamp: of 3,270 held-out crops only 40 (1.2%) are
 * <= 256px, and the median and max are both 320. This graph was exported at 256
 * before, on the belief that 256 sat "above the median crop". It sat BELOW it, so
 * the app truncated the tail off ~99% of lines before the model saw them. On 200
 * held-out crops, exact line match was 44.7% at natural width and 0.0% cut to
 * 256, with the read changed on 200 of 200: 'PATIENT JANE DOE DOB 04021979' came
 * back as 'PERIENT JANE DOE 2O08010', 'DATE 01102026' as 'DATE 01020'.
 *
 * The root cause of the FIXED width was never the LSTM op: `SightLineCRNN.
 * set_export_mode(True)` runs an explicit per-timestep Python LSTM loop, and
 * tracing THAT unrolls the recurrence and bakes T into the graph (the torch and
 * dynamo errors quoted in ml/src/export_dynamic.py). Tracing the FUSED nn.LSTM
 * lowers to a real ONNX LSTM op whose sequence axis is dynamic natively, so the
 * graph accepts [B, 1, 32, W] for any W and computes what eager PyTorch
 * computes. That is the whole mechanism, re-verified in the browser by
 * app/tools/dynamic_width_check.js.
 *
 * Padding up to a fixed width is NOT inert, which is the reason to prefer the
 * natural width at all. The loss being CTC does not save it: a padded region
 * emits blanks so the LOSS ignores it, but this CRNN carries a two-layer
 * BIDIRECTIONAL LSTM whose backward pass starts at the RIGHT EDGE, i.e. inside
 * the padding, so padded zeros move the hidden states across the VALID frames
 * (measured max |delta| 9.10 at content 64, 15.28 at 200, in logit units --
 * enough to change the decoded string). 74.7% of held-out crops are padded at
 * all against the 320 clamp, and on crops the clamp does NOT truncate the padded
 * serve path scores crop-level exact 19.0% / CER 0.308 against 30.5% / 0.198 at
 * natural width, agreeing with the natural decode on only 43% of crops. The
 * fixed path is what the app runs until the swap; it is a known loss, not a
 * neutral default.
 *
 * The fixed path's trim is still required: without it a narrow crop would be
 * read as timesteps of padding, and any residual signal there becomes invented
 * characters. timestepsFor FLOORS (MaxPool2d floors, so a clean w/8 formula
 * mis-trims odd widths); it is a mask, not the model's own timestep count.
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

const ORT_URL = "../vendor/ort/ort.min.mjs";   // module-relative: import() resolves
                                               // against THIS module, not the page
let ORT = null; // ort module namespace — set by init() via dynamic import below

// onnxruntime-web resolves wasmPaths (and the model URL) against the DOCUMENT
// — a relative prefix becomes a BARE import specifier, which browsers reject
// ("Failed to resolve module specifier"), and a page-relative one breaks under
// subdirectory deploys (/SightLine/ on GitHub Pages). Build absolute URLs from
// THIS module's location: correct in every mount point — the same problem
// classifier.js solves with base().
const WASM_DIR = new URL("../vendor/ort/", import.meta.url).href;
const MODEL_URL = new URL("../models/crnn.onnx", import.meta.url).href;

export const CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
export const BLANK = 0;
export const CROP_H = 32;

// The TRAINING CLAMP (realdata.MAX_W) — NOT the served width. build_dataset
// clamps every training crop to it, so it is the widest crop the model was ever
// trained on AND the crop cap app/js/ocr_engine.js passes to cropForRecognizer.
// Nothing in this module builds a tensor at it: the tensor width is the graph's
// own (engineWidth()). Serving at the clamp is what re-introduces the padding
// loss below, so this constant must never travel back into the tensor shape.
export const TRAINING_CLAMP = 320;

// Legacy name, kept because app/js/ocr_engine.js imports it and passes it as
// cropForRecognizer's maxW. That crop cap is all it is now — it is NOT the
// tensor width, and buildInput does not take it.
export const EXPORT_W = TRAINING_CLAMP;

export const WIDTH_DIVISOR = 8;   // timesteps = contentWidth / 8

// ── the word-LM decode stage ────────────────────────────────────────────────
// Applied AFTER the graph, to the logits the graph returns (see word_beam.js).
// Measured on 3,375 held-out crops of THIS checkpoint: greedy 67.97% exact ->
// beam 4 + word LM at alpha 0.25 = 70.67% (+114 / -23 crops, exact McNemar
// p < 1e-4). Beam width 4 rather than 8 because the two tied on exact match
// (71.10% on a 1,000-crop sweep, 70.67% vs 70.70% here) at half the cost.
export const WORD_BEAM = 4;
export const WORD_ALPHA = 0.25;
const WORD_TRIE_URL = new URL("../data/word_trie.json", import.meta.url).href;

let session = null;
let initPromise = null;
let wordTrie = null;          // null until loaded; greedy decode while absent
let wordTriePromise = null;
let wordTrieError = null;

/**
 * The trie is optional BY DESIGN: a phone that cannot fetch it still reads text,
 * it just reads it with the older greedy decoder. So this never throws and never
 * blocks init() — it resolves to null and the scan proceeds.
 */
export function loadWordTrie() {
  if (!wordTriePromise) {
    wordTriePromise = WordTrie.load(WORD_TRIE_URL)
      .then((t) => { wordTrie = t; return t; })
      .catch((e) => { wordTrieError = e && e.message ? e.message : String(e); return null; });
  }
  return wordTriePromise;
}

/** Which decoder the next scan will use, and why — for the harnesses and the UI. */
export function decoderState() {
  return { wordLM: wordTrie ? "on" : "off", words: wordTrie ? wordTrie.words : 0,
           greedy: wordTrie === null, error: wordTrieError };
}

// The graph's width axis, read from the session itself at init:
//   undefined  not read yet (before init, and after reset())
//   null       the axis is SYMBOLIC -> serve every crop at its natural width
//   number     the axis is FIXED at that width -> pad up to it and trim
let graphWidth;

/**
 * The width this session is actually served at: 'dynamic' when the graph's
 * width axis is symbolic, otherwise the fixed number.
 *
 * Exported so the app's meta line and the browser harness assert what was
 * CHOSEN rather than trusting a constant — a silent fallback to a hardcoded
 * width is the exact failure class this module is being changed to remove.
 */
export function engineWidth() { return graphWidth === null ? "dynamic" : graphWidth; }

const cache = new Map();          // tensor width -> Float32Array scratch buffer

/**
 * WebAssembly SIMD feature probe.
 *
 * onnxruntime-web 1.20.1 ships exactly ONE non-JSEP wasm binary --
 * `ort-wasm-simd-threaded.wasm` -- and it requires WASM SIMD (ORT dropped its
 * non-SIMD builds after 1.19: "ONNX Runtime Web is dropping support for
 * non-SIMD and non-threaded builds ... since v1.19.0"). Safari shipped
 * WebAssembly SIMD in 16.4, so on iOS 15 and iOS 16.0-16.3 the whole neural
 * path cannot instantiate, while the app's own docs advertise iOS 15+.
 *
 * There is no Tesseract left to fall back to, so this must be detected BEFORE
 * the runtime is imported: a failure inside ort surfaces as an opaque
 * instantiation error minutes into the scan, and the user's phone looks
 * broken. Exported so the UI can state the real reason.
 *
 * The module bytes below are the canonical wasm-feature-detect SIMD snippet:
 * `(func (result v128) (v128.const i32x4 0 0 0 0))` with a shuffle op. It
 * validates only on a SIMD-capable engine -- confirmed true under WebKit
 * (tools/ios_webkit_check.js prints the probe result).
 */
export function simdSupported() {
  try {
    return typeof WebAssembly === "object" && WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123,
      3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
    ]));
  } catch (e) {
    return false;
  }
}

/** The message the UI shows when the hardware/browser cannot run the model. */
export const NO_SIMD_MESSAGE =
  "This device's browser has no WebAssembly SIMD, so the text recogniser " +
  "cannot run. It needs Safari 16.4 or newer (iOS 16.4+, March 2023). " +
  "Update iOS, or open the app in a newer browser.";

/**
 * The probe widths used to read the graph's width axis from the session.
 *
 * Deliberately NOT round numbers: an exported graph's fixed width is the clamp
 * or a bucket (256, 320, 192, ...), and a probe that happened to equal it would
 * read as "dynamic" off a single run that only proved the graph accepts its own
 * width. Both are above MIN_W, so neither can be refused for being degenerate.
 */
const PROBE_WIDTHS = [41, 63];

/**
 * Read the graph's declared input width FROM the session.
 *
 * onnxruntime-web 1.20.1 exposes NO input-shape metadata. VERIFIED, not assumed
 * (webkit and chromium, the vendored app/vendor/ort/ort.min.mjs 1.20.1): an
 * InferenceSession carries only `handler`, `run`, `release`, `inputNames` and
 * `outputNames`; `inputNames` is ["input"] with no dims, `inputMetadata` does
 * not exist, and the wasm handler has no shape accessor either.
 *
 * What the runtime DOES do is validate tensor dims at run() and report the
 * graph's OWN number — the same declaration the exporter wrote:
 *
 *   failed to call OrtRun(). ERROR_CODE: 2, ERROR_MESSAGE: Got invalid
 *   dimensions for input: input for the following indices
 *    index: 3 Got: 63 Expected: 320
 *
 * So ask the session the only way it answers: run a zero tensor at a width the
 * graph is not. If BOTH probes run, the axis is symbolic; a refusal names the
 * fixed width. Two probes because one successful run cannot tell a dynamic axis
 * from a fixed one that happens to equal the probe width. A refused run does not
 * damage the session — verified: the same session returns a correct 320-wide
 * [1,40,38] result immediately after two refusals, in both engines.
 *
 * @returns {Promise<number|null>} the fixed width, or null when it is symbolic.
 */
async function readGraphWidth(s) {
  for (const w of PROBE_WIDTHS) {
    try {
      await s.run({
        input: new ORT.Tensor("float32", new Float32Array(CROP_H * w), [1, 1, CROP_H, w]),
      });
    } catch (e) {
      const msg = e && e.message ? e.message : String(e);
      const m = /index:\s*3\s+Got:\s*\d+\s+Expected:\s*(\d+)/.exec(msg);
      if (!m) {
        throw new Error("the graph refused width " + w
          + " but did not declare its own: " + msg);
      }
      return Number(m[1]);
    }
  }
  return null;                    // every probe ran: the width axis is symbolic
}

/**
 * Load the ONNX session once. Concurrent callers share one promise.
 *
 * A failed load resolves to null and sets `lastError` rather than throwing, so
 * the pipeline can fall back rather than break -- the app degrades, it never
 * shows a broken state.
 *
 * @param {string} [modelUrl]  overrides MODEL_URL. Only the browser harness
 *   (app/tools/dynamic_width_check.js) passes this, to drive THIS module
 *   against both a fixed-width and a dynamic-width graph; the app calls init()
 *   bare. Pair with reset(), which clears the memoised session.
 */
export async function init(modelUrl = MODEL_URL) {
  if (session) return session;
  if (initPromise) return initPromise;
  initPromise = (async () => {
    try {
      if (!simdSupported()) {
        lastError = NO_SIMD_MESSAGE;
        return null;
      }
      // Load the runtime the way classifier.js and restorer.js do: dynamic
      // import into a module namespace. (The previous code named the URL
      // string `ORT` and read `ORT.env` off it — a TypeError in strict-mode
      // modules, so the recogniser could never load in ANY browser. Found by
      // the selftest's ONNX group, which exists precisely to catch this.)
      ORT = await import(ORT_URL);
      const env = ORT.env || (ORT.env = {});
      env.wasm = env.wasm || {};
      // onnxruntime-web resolves wasmPaths (and the model URL) against the
      // DOCUMENT — a relative prefix becomes a BARE import specifier, which
      // browsers reject ("Failed to resolve module specifier"), and an
      // absolute prefix is needed under subdirectory deploys (/SightLine/ on
      // GitHub Pages). Build both from THIS module's location: correct in
      // every mount point, no DOM inspection — the same problem
      // classifier.js solves with base().
      env.wasm.wasmPaths = WASM_DIR;
      // Static hosting cannot send the cross-origin-isolation headers, so
      // SharedArrayBuffer is unavailable and pthreads WASM cannot start.
      // Request threads only when the page really is isolated; ort's
      // threaded wasm otherwise fails depending on version. A phone visiting
      // the deployed PWA is never COI, so this is the deployed case, not the
      // rare one.
      const threads = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
      env.wasm.numThreads = (typeof SharedArrayBuffer !== "undefined" && globalThis.crossOriginIsolated) ? threads : 1;
      session = await ORT.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"],
      });
      // Fire the word-LM fetch now and DO NOT await it: the first scan must not
      // wait on a 195 KB sibling, and a missing/slow trie degrades to greedy
      // rather than failing the scan (see loadWordTrie).
      loadWordTrie();
      // Ask the graph what width it takes, ONCE, before any crop is served.
      try {
        graphWidth = await readGraphWidth(session);
      } catch (e) {
        // The session will not say. TRAINING_CLAMP is what the shipped artifact
        // is, so assuming it keeps this build working rather than disabling OCR
        // — but say so loudly: a wrong width corrupts the tensor silently.
        graphWidth = TRAINING_CLAMP;
        console.warn("recognizer: could not read the graph's width ("
          + (e && e.message ? e.message : String(e)) + "); assuming the training clamp "
          + TRAINING_CLAMP);
      }
      console.info("recognizer: engine.width = " + engineWidth()
        + (graphWidth === null
          ? " (symbolic width axis: every crop is served at its natural width)"
          : " (fixed width axis: crops are padded up to it and the decode trimmed)"));
      return session;
    } catch (e) {
      lastError = e && e.message ? e.message : String(e);
      // Clear the failed promise so a later call can retry (a transient
      // runtime/compile failure must not disable OCR until page reload) —
      // same policy as pipeline.js getEngine().
      initPromise = null;
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
 * Build the [1,1,32,w] input tensor for one crop.
 *
 * `w` is the SESSION's width (engineWidth()) — never a constant. In the fixed
 * mode it is the graph's own number and a shorter crop is right-padded with
 * zeros; in the dynamic mode it IS contentWidth, so nothing is padded. The row
 * stride is `w`, so the tensor shape and the stride cannot disagree with each
 * other or with the graph.
 *
 * Input is normalised to [-1,1] because every crop is dark ink on light paper,
 * so +1 is "paper" and -1 is "ink". Matching the Python training transform
 * exactly matters: a different scale here produces confidently wrong text.
 * A zero tail in NORMALISED space is mid-grey, which is what train_ocr.collate
 * pads with — deliberately not paper.
 */
function buildInput(gray, contentWidth, w) {
  // ponytail: keyed by tensor width, so the dynamic mode keeps one buffer per
  // distinct crop width (<= 40 KB each). Same entry count as the old
  // EXPORT_W-sized cache and smaller buffers; cap it if a page ever segments
  // thousands of distinct widths.
  let buf = cache.get(w);
  if (!buf) {
    buf = new Float32Array(CROP_H * w);
    cache.set(w, buf);
  } else {
    buf.fill(0);              // clear the previous crop, padding included
  }
  for (let y = 0; y < CROP_H; y++) {
    const srcRow = y * contentWidth;
    const dstRow = y * w;
    for (let x = 0; x < contentWidth; x++) {
      buf[dstRow + x] = (gray[srcRow + x] / 255 - 0.5) / 0.5;
    }
  }
  return buf;
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
 * @param {number} contentWidth  the crop's real width (never the tensor width).
 * @returns {Promise<{text: string, confidence: number, chars: number,
 *   timesteps: number}>}
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

  // Serve at the width the GRAPH declared: its own fixed number (right-pad the
  // crop up to it) or the crop's natural width (nothing to pad). engineWidth()
  // is the same decision, made once, in one place.
  const w = graphWidth === null ? contentWidth : graphWidth;
  const input = buildInput(gray, contentWidth, w);
  try {
    const res = await s.run({
      input: new ORT.Tensor("float32", input, [1, 1, CROP_H, w]),
    });
    const logits = res.logits;
    const T = logits.dims[1];
    const C = logits.dims[2];
    const data = logits.data;

    // The padded tail must not be decoded: training never supervised it.
    // (This exact line is pinned by app/tools/ocr_input_contract.js.)
    const Ttrue = Math.min(timestepsFor(contentWidth), T);
    // That trim is a MASK FOR PADDING, so it applies to the fixed path only. At
    // natural width there is no padding and T IS this crop's timestep count --
    // and timestepsFor is not that count: it floors w/8 while the conv stack
    // gives ceil(w/8) at some widths (T=6 at w=47 against floor(47/8)=5), so
    // masking here would drop a real timestep and read the tail short.
    const decodeT = graphWidth === null ? T : Ttrue;
    const idx = new Int32Array(decodeT);
    let confSum = 0, emitted = 0;

    for (let t = 0; t < decodeT; t++) {
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

    // The word-LM stage: same logits, searched over prefixes instead of read
    // greedily. It runs only when the trie loaded — greedy is the fallback, and
    // greedy is what the confidence below describes either way (the gates that
    // consume confidence were calibrated on this emission probability, so it is
    // deliberately NOT swapped for a beam score).
    let text = decodeGreedy(idx, decodeT);
    let decodedBy = "greedy";
    if (wordTrie) {
      const lps = new Float64Array(decodeT * C);
      const row = new Float64Array(C);
      for (let t = 0; t < decodeT; t++) {
        logSoftmaxRow(data, t * C, C, row);
        lps.set(row, t * C);
      }
      // The weight comes from the loaded artifact (see WordTrie.alpha) so a retuned
      // lexicon cannot ship against the old literal; WORD_ALPHA stays the documented
      // fallback for tries exported before meta.alpha existed.
      const beam = beamDecode(lps, decodeT, C, CHARSET,
                              { beam: WORD_BEAM, blank: BLANK, lm: wordTrie,
                                alpha: wordTrie.alpha === undefined ? WORD_ALPHA
                                                                    : wordTrie.alpha });
      // A beam that returns nothing usable must not replace a readable greedy
      // string: the fallback only ever fires on an EMPTY read.
      if (beam.text) { text = beam.text; decodedBy = "wordLM"; }
    }
    return {
      text,
      confidence: emitted ? confSum / emitted : 0,
      chars: text.length,
      timesteps: decodeT,
      decoder: decodedBy,
    };
  } catch (e) {
    lastError = e && e.message ? e.message : String(e);
    return { text: "", confidence: 0, chars: 0, error: lastError };
  }
}

/**
 * Recognise a batch of crops.
 *
 * Sequential rather than parallel ONNX calls: one graph is small enough that N
 * concurrent sessions thrash WASM memory on a phone, and the worker is already
 * off the UI thread. Measured on the desktop harness this is faster than
 * Promise.all at realistic crop counts.
 */
export async function recogniseAll(crops) {
  const out = [];
  for (const c of crops) {
    out.push(await recognise(c.gray, c.width));
  }
  return out;
}

/**
 * Release the session. Called from pipeline.dispose().
 *
 * On iOS this is the difference between a working scan and a jettisoned tab:
 * ort's wasm heap cannot shrink once grown, but release() returns this graph's
 * allocations to the runtime, so the next scan reuses them instead of growing
 * the heap again. Dropping the JS reference alone does none of that. The call
 * is async and deliberately not awaited -- nothing waits on a free, and a
 * rejected release just means it was already released.
 */
export function reset() {
  const s = session;
  session = null;
  initPromise = null;
  lastError = null;
  graphWidth = undefined;      // next init() re-reads it from ITS session
  cache.clear();
  if (s && typeof s.release === "function") {
    try {
      const p = s.release();
      if (p && typeof p.catch === "function") p.catch(() => { /* already gone */ });
    } catch (e) { /* already gone */ }
  }
}

// Exported for the browser self-test harness.
const SightLineRecognizer = {
  CHARSET, BLANK, CROP_H, EXPORT_W, TRAINING_CLAMP, WIDTH_DIVISOR,
  init, isReady, getError, recognise, recogniseAll,
  decodeGreedy, timestepsFor, engineWidth, reset,
  loadWordTrie, decoderState, WORD_BEAM, WORD_ALPHA,
  simdSupported, NO_SIMD_MESSAGE,
};
if (typeof window !== "undefined") window.SightLineRecognizer = SightLineRecognizer;
export default SightLineRecognizer;