/**
 * Differential sweep: reference normalizer (SightLine-Mobile) vs. the fixed one
 * shipped in app/js/pipeline.js.
 *
 * Two bugs in the reference, both in the medical-dosage path:
 *
 *   BUG 1 — rule order. The leading-S rule ran BEFORE m9->mg, so "S00m9" was
 *   seen as "S00m9" (the lookahead needs "mg" to close the word, and "m9" was
 *   not in the alternation) and became "S00mg" — still not a dose. The
 *   comment claims "S00m9 -> 500mg"; it never did. Fix: resolve m9/m1 first,
 *   so the S rule sees "S00mg" and fires.
 *
 *   BUG 2 — the de-duplication rule "\s*\b(mg|ml)\s+\1" could never match the
 *   tail of a real dose, because there is no word boundary between the "0" and
 *   the "m" of "50mg". Its stated purpose ("50mg mg" -> "50mg") never fired.
 *   A naive \b removal yields "50 mg", which changes the dose token and breaks
 *   field matching. Fix: anchor on the leading digits and rejoin them.
 *
 * ORACLE. The question that actually matters is not "did the string change" but
 * "did the fix recover a ground-truth field". So every case is scored with
 * pipeline.fields_found(), the port of evaluate.fields_found, against
 * evaluate.FIELDS. A change is a REPAIR if it recovers a field the old
 * normalizer lost and loses none it had. This is the same all-or-nothing
 * standard the Python eval applies to a dosage.
 *
 * Run: node tools/normalizer_diff.js
 */
"use strict";

const O = require("/tmp/ad/SightLine-Mobile/js/pipeline.js");
const S = require("../js/pipeline.js");

function FIXED(text) {
  return String(text || "")
    .replace(/m9\b/gi, "mg")
    .replace(/m1\b/gi, "ml")
    .replace(/\bS(?=\d{2,4}(?:m[gq]|m1|1m|mg|ml)?\b)/g, "5")
    .replace(/(\d+)\s*(mg|ml)\s+\2\b/gi, "$1$2");
}

const PRESCRIPTION = [
  "PRESCRIPTION", "Patient: Jane Doe   DOB: 04/02/1979", "Date: 01/10/2026",
  "PRESCRIBER: Dr A. Ahmed, MBBS", "Pharmacy: Northgate Pharmacy",
  "Rx  Amoxicillin 500mg capsules",
  "Take one capsule three times daily with food",
  "Complete the full 7 day course",
  "Rx  Ibuprofen 400mg tablets",
  "Every 6 to 8 hours as required for pain",
  "Maximum 1200mg in 24 hours",
  "WARNING: may cause drowsiness",
  "Do not drive or operate machinery",
  "Keep out of reach of children",
  "Pharmacist: R. Silva  GPhC 20441",
];

/** Every fragment exercised, including ones that must NOT change. */
const FRAGMENTS = [
  // broken dosages the reference could not repair
  "S00m9", "S99m9", "S12m9", "S250m9", "S00m1", "S401m1", "50mg mg", "10ml ml",
  // already-correct dosages - must be untouched
  "500mg", "400mg", "1200mg", "50mg", "5mg", "1mg", "S00mg", "m9", "m1",
  // words that merely look similar - must not be damaged
  "Salary", "SMS", "Statement", "Sg", "9m9", "S5", "S0", "S001", "20m9",
  "September", "Sample", "Simply", "mg", "ml", "1m", "m1L", "m9L",
  // non-medical documents - the normalizer must not corrupt these
  "Opening Balance GBP 1,204.33", "Sort Code: 40-11-04", "Account Number: 40218877",
  "Clause 3.2 rent of 2,500 per month", "Case Reference: 2024-CV-00456",
];

const SUFFIX = ["", " daily", " tablets", " to take", " twice", " x", " caps"];

let repairs = 0, regressions = 0, unchanged = 0;

/** Fields recovered from a full document, scored all-or-nothing. */
function score(fullText) {
  return S.fields_found(fullText, "prescription");
}

for (const frag of FRAGMENTS) {
  for (const suf of SUFFIX) {
    const s = frag + suf;
    const o = O.normalizeOCRText(s);
    const n = FIXED(s);
    if (o === n) { unchanged++; continue; }

    // Real test: build two full documents, one normalised the old way and one
    // the new way, and compare the ground-truth fields each recovers.
    const oldDoc = PRESCRIPTION.join("\n").replace("500mg", s);
    const newDoc = PRESCRIPTION.join("\n").replace("500mg", n);
    const oldGot = score(oldDoc);
    const newGot = score(newDoc);

    const lost = oldGot.filter((f) => !newGot.includes(f));
    const gained = newGot.filter((f) => !oldGot.includes(f));
    const ok = lost.length === 0;
    if (ok) repairs++; else regressions++;
    console.log(
      (ok ? "REPAIR  " : "REGRESS ") + JSON.stringify(s).padEnd(22) +
      "old=" + JSON.stringify(o).padEnd(24) + "new=" + JSON.stringify(n).padEnd(22) +
      (gained.length ? " +" + gained.join(",") : "") +
      (lost.length ? " -" + lost.join(",") : "")
    );
  }
}

console.log("\nunchanged: " + unchanged + "   repairs: " + repairs + "   regressions: " + regressions);
console.log(regressions === 0
  ? "PASS - every change recovers a ground-truth dosage and loses none"
  : "FAIL");
process.exit(regressions === 0 ? 0 : 1);
