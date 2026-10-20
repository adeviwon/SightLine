/**
 * The recogniser's input contract, on the app side.
 *
 * WHAT THIS PINS, and why each one is here:
 *
 *   1. THE BEST INPUT WAS UNREACHABLE. In the arms table
 *      (artifacts/eval_results.json, engine "crnn") the raw deskewed grayscale
 *      scores best on our own rendered documents — 5.6% field accuracy / 26.9%
 *      word — against classical 5.6/23.9, restorer 7.4/26.4 and restorer_clahe
 *      3.7/23.5. app/js/pipeline.js applied CLAHE to everything and offered the
 *      engine only [clahe, clahe+otsu]: its two WORST inputs. Restore is not why
 *      the words are wrong (crop-level, with no preprocessing at all, our own
 *      documents still read 4.4% exact — that is the model), but it was a net
 *      negative that cost points for no reason.
 *
 *   2. THE TENSOR TAIL MUST NOT BE DECODED. Padding the crop up to the graph's
 *      fixed width is legitimate ONLY because the padded timesteps are dropped
 *      before decoding (`Ttrue = min(timestepsFor(contentWidth), T)`), and
 *      because training excluded the same region from the CTC loss
 *      (`train_ocr.collate` passes per-sample true `input_lengths`). Remove that
 *      limit and the model's opinion about padding — which training never
 *      supervised — starts emitting characters.
 *
 *      A NOTE ON THE PADDING VALUE, because it is a trap: this file used to
 *      assert the tail is filled with paper (+1) rather than mid-grey (0). THAT
 *      WAS WRONG and the assertion was removed after reading the trainer.
 *      `collate` builds batches with `np.zeros(...)` in NORMALISED space, so
 *      training's padding is 0 = mid-grey; the tail carries no gradient and is
 *      never decoded. "Fixing" the JS to paper would have introduced exactly the
 *      train/serve mismatch this file exists to prevent.
 *
 * Source-level checks, because these are cross-module contracts no unit test of
 * either module would catch: pipeline.js decides the candidate list,
 * recognizer.js decides the tensor and the decode window.
 *
 * Run: node app/tools/ocr_input_contract.js   (exit 0 = the contract holds)
 */
"use strict";

const fs = require("fs");
const path = require("path");

const JS = path.resolve(__dirname, "..", "js");
const pipelineSrc = fs.readFileSync(path.join(JS, "pipeline.js"), "utf8");
const recSrc = fs.readFileSync(path.join(JS, "recognizer.js"), "utf8");

let checks = 0, fails = 0;
function check(name, cond, detail) {
  checks++;
  if (!cond) fails++;
  console.log("  " + (cond ? "ok  " : "FAIL") + "  " + name
    + (cond || detail === undefined ? "" : "  -> " + detail));
  return cond;
}

console.log("the raw image must reach the recogniser:");
check("preprocess builds a `raw` candidate from the deskewed grayscale",
  /const raw = deskewed;/.test(pipelineSrc),
  "app/js/pipeline.js has no unprocessed candidate — the best-measured input is "
    + "unreachable");
check("the candidate list offers raw alongside the processed ones",
  /const candidates = \[raw, enhanced, binary\]/.test(pipelineSrc),
  "candidates must be [raw, enhanced, binary] so structure_score can choose");
check("the reporting arm names match the candidate list",
  /const arms = \[\s*"raw",/.test(pipelineSrc),
  "the arms array still mislabels the candidates it reports");

console.log("\nthe tensor tail must not be decoded:");
check("recognise() limits decoding to the crop's true timesteps",
  /const Ttrue = Math\.min\(timestepsFor\(contentWidth\), T\)/.test(recSrc),
  "app/js/recognizer.js decodes the padded tail, which training never supervised");

console.log("\n" + (checks - fails) + "/" + checks + " checks passed");
if (fails) {
  console.log("OCR INPUT CONTRACT FAILED — the recogniser is being fed an input it "
    + "was not trained on");
  process.exit(1);
}
console.log("OCR INPUT CONTRACT OK");
