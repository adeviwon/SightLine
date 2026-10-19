/**
 * psm_sweep — RETIRED. This file now says why instead of throwing.
 *
 * It swept Tesseract's full page-segmentation space (PSM 0-13) against every OCR
 * candidate on every rendered sample, to answer one question: were the 0% field
 * scores on the handheld profiles a Tesseract ceiling, or was our own pass list
 * (psm 6/3/11) too narrow? The recorded answer was "no configuration in the full
 * PSM space recovers any field" — which is why `runOCR`'s pass list stayed as it
 * was, and why docs 03/04 describe the handheld profiles as a ceiling.
 *
 * The app no longer contains Tesseract. The recogniser is our own CRNN,
 * `layout.detectLines` produces the segmentation, and there is no mode to choose
 * — see the note at the top of `js/ocr_engine.js`. So this harness cannot run,
 * and it used to fail with `TypeError: SightLine.setPSM is not a function`, which
 * reads like a broken tool rather than a retired concept: the obvious "fix" is to
 * reintroduce a knob the recogniser does not have.
 *
 * What asks the equivalent question now:
 *   - `bash run.sh ios-full`            field recovery + classification per
 *                                       rendered sample, WebKit and Chromium
 *   - `bash run.sh eval-ocr`            the recogniser's own line-level accuracy
 *   - `node tools/crop_contract.js`     what the recogniser is actually fed
 *
 * Run: node tools/psm_sweep.js  ->  exit 2 (skip) with this explanation.
 */
"use strict";

console.log("SKIP: psm_sweep swept Tesseract's page-segmentation modes.");
console.log("      The app ships our own CRNN — no Tesseract, no PSM; the line");
console.log("      detector IS the segmentation. Its historical verdict ('no PSM");
console.log("      recovers the handheld profiles') still stands and is why the");
console.log("      pass list stayed as it was.");
console.log("      Replacements: bash run.sh ios-full · bash run.sh eval-ocr ·");
console.log("      node tools/crop_contract.js. See this file's header.");
process.exit(2);
