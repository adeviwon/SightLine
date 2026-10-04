/**
 * The OCR stage: our own CRNN+CTC recogniser, not a library.
 *
 * This module is what replaced tesseract.js. The difference is not "we swapped
 * one dependency for another" -- there is no OCR dependency left. The
 * recogniser is a 942,166-parameter CNN + bi-LSTM trained from scratch on
 * ICDAR 2019 SROIE, exported to a self-contained ONNX graph, and run here
 * through onnxruntime-web. Nothing about the glyphs is inferred by a
 * third-party binary: the weights are in `models/crnn.onnx` because we put
 * them there.
 *
 * WHY THIS IS THE RIGHT SHAPE, and why it is also the hard one:
 *
 *   The recogniser consumes 32px-tall crops. A phone photo is a whole page.
 *   So the line DETECTOR (`layout.js`) has to find text lines before the
 *   recogniser can read anything, and detection recall is a hard ceiling on
 *   what this module can achieve. That ceiling is measured and reported in
 *   docs/03_RESULTS.md rather than assumed away.
 *
 *   Tesseract hid this. It does its own page segmentation internally, so the
 *   failure mode was invisible -- a bad page just produced bad text. Splitting
 *   the problem into detect-then-recognise makes the ceiling explicit and
 *   therefore fixable.
 *
 * WHAT THE FIXED WIDTH MEANS. `crnn.onnx` takes exactly [1,1,32,256]. Width is
 * fixed because `aten::lstm` cannot export with a dynamic sequence length and
 * an explicit gate loop cannot unroll over a symbolic bound. So each line crop
 * is scaled proportionally to preserve aspect ratio, right-padded to 256, and
 * the decoder trims to the crop's true timestep count (width/8). Padding is
 * safe because a trained CTC model emits the BLANK symbol there -- which is
 * verified, not assumed: see `verify_crnn`'s padding invariant in
 * `ml/src/export_onnx.py`, which compares a real crop padded to 256 against the
 * same crop at its own width.
 *
 * NO MULTI-PASS. The previous implementation ran tesseract.js four times per
 * candidate (PSM 3, 6, 11, 4) and ranked the results. There is no segmentation
 * mode to choose here: the detector produces the segmentation. One pass, one
 * result, and `structure_score` still ranks candidates -- on the product
 * metric, not on confidence.
 */

import {
  init as initRecognizer,
  recogniseAll,
  isReady,
  reset as resetRecognizer,
} from "./recognizer.js";
import { detectLines, cropForRecognizer } from "./layout.js";
import { EXPORT_W } from "./recognizer.js";

/** Canvas -> flat grayscale Uint8Array. */
function toGray(canvas) {
  const w = canvas.width;
  const h = canvas.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const d = ctx.getImageData(0, 0, w, h).data;
  const g = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < g.length; i++, p += 4) {
    // Rec. 601 luma, integer form. The model was trained on grayscale crops
    // converted with this weighting, so reusing it keeps train/inference
    // consistent.
    g[i] = (d[p] * 77 + d[p + 1] * 150 + d[p + 2] * 29) >> 8;
  }
  return { gray: g, w, h };
}

/**
 * Recognise one candidate image.
 *
 * Returns the same shape the rest of the pipeline already consumes, so this is
 * a drop-in replacement:
 *   { text, confidence, words, psm, structure, psmRank, lines, engine }
 */
async function recogniseCanvas(canvas, onProgress) {
  const { gray, w, h } = toGray(canvas);

  if (onProgress) onProgress({ status: "detecting text lines" });
  const lines = detectLines(gray, w, h);
  if (!lines || lines.length === 0) {
    return {
      text: "",
      confidence: 0,
      words: [],
      psm: "crnn-detect",
      structure: 0,
      psmRank: 0,
      lines: 0,
      engine: "crnn",
    };
  }

  // maxW MUST be EXPORT_W. cropForRecognizer defaults to 320, and feeding a
  // 320-wide crop to a graph that wants exactly 256 is a shape error -- not a
  // silent truncation.
  const crops = lines.map((box) => {
    const c = cropForRecognizer(gray, w, box, 32, EXPORT_W);
    // cropForRecognizer returns `data`; recogniseAll reads `gray`. Renaming
    // here rather than changing either module's contract keeps both usable by
    // their own tests.
    return { gray: c.data, width: c.width };
  });

  if (onProgress) {
    onProgress({ status: "reading " + crops.length + " lines" });
  }
  const out = await recogniseAll(crops);

  // Keep only lines that produced something. A blank line is a detection
  // artefact, and concatenating it would insert phantom words into the text
  // the field extractor then parses.
  const kept = out.filter((r) => r.text && r.text.trim().length > 0);
  const text = kept.map((r) => r.text.trim()).join("\n");
  const confidence = kept.length
    ? kept.reduce((s, r) => s + r.confidence, 0) / kept.length
    : 0;

  return {
    text,
    confidence,
    // Empty by design: the CRNN emits line strings, not word boxes. Field
    // extraction works on the text. Claiming word boxes we do not produce
    // would be the same kind of lie as the old manifest's "tesseract": false.
    words: [],
    psm: "crnn-detect",
    structure: 0,          // filled in by the caller via structure_score
    psmRank: 0,
    lines: kept.length,
    engine: "crnn",
  };
}

/**
 * Load the recogniser. Mirrors the old initWorker() contract so the pipeline
 * call site is unchanged.
 */
export async function initWorker(onProgress) {
  if (onProgress) onProgress({ status: "loading our OCR model" });
  await initRecognizer();
  if (!isReady()) {
    throw new Error("crnn.onnx failed to initialise (see recognizer.js getError())");
  }
  return { recognise: recogniseCanvas };
}

/**
 * Run OCR over every candidate and return the best.
 *
 * Ranking is on `structure` -- the count of safety-critical tokens the product
 * actually extracts -- then confidence, then candidate order. This is
 * inherited from the previous implementation and kept deliberately: the lesson
 * recorded in docs/10_ACCURACY_GAP.md is that confidence and field recovery can
 * disagree, and the product metric must win.
 */
export async function runOCR(candidates, onProgress, structureScore) {
  await initWorker(onProgress);
  let best = null;
  const seen = new Set();
  const unique = candidates.filter((img) => {
    if (seen.has(img)) return false;
    seen.add(img);
    return true;
  });

  for (let ci = 0; ci < unique.length; ci++) {
    const res = await recogniseCanvas(unique[ci], onProgress);
    if (!res.text) continue;
    const structure = structureScore ? structureScore(res.text) : 0;
    const cand = { ...res, structure, psmRank: ci };
    if (
      !best ||
      cand.structure > best.structure ||
      (cand.structure === best.structure && cand.confidence > best.confidence)
    ) {
      best = cand;
    }
  }

  return best || {
    text: "",
    confidence: 0,
    words: [],
    psm: "crnn-detect",
    structure: 0,
    psmRank: 0,
    lines: 0,
    engine: "crnn",
  };
}

/** Release the onnxruntime-web session. Called from pipeline.dispose(). */
export function reset() {
  resetRecognizer();
}

export default { initWorker, runOCR, reset };