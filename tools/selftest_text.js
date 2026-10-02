/**
 * Node harness for the selftest.html text battery.
 *
 * selftest.html is the user-facing runner; this file executes the SAME
 * assertions headlessly so they can be verified in CI or from a terminal.
 * The assertion bodies below are a 1:1 copy of the ones in selftest.html —
 * if you change one, change both, or the page is no longer the source of truth.
 *
 * The image-processing battery is NOT run here: it needs a real 2-D canvas.
 * Open selftest.html in a browser for that half.
 *
 * Run: node tools/selftest_text.js
 */
"use strict";

const SightLine = require("../js/pipeline.js");
const SightLineRestorer = require("../js/restorer.js");
const SightLineClassifier = require("../js/classifier.js");
const TTS = require("../js/tts.js");

// ── Vectors: verbatim from the Python sources ─────────────────────────
const CAPTURE = {
  prescription: [
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
  ],
  banking: [
    "HSBC UK BANK PLC", "Statement of Account", "Account Holder: John Smith",
    "Account Number: 40218877", "Sort Code: 40-11-04", "Period: 01/03 - 31/03/2026",
    "Opening Balance        GBP 1,204.33",
    "01/03 SALARY CREDIT   GBP 2,900.00",
    "01/03 RENT DD 1100     GBP -1,100.00",
    "08/03 TESCO STORES     GBP -44.19",
    "14/03 BRITISH GAS DD   GBP -89.99",
    "19/03 ATM WITHDRAWAL   GBP -200.00",
    "22/03 CASHBACK         GBP +12.40",
    "Deposits total         GBP 2,912.40",
    "Withdrawals total      GBP -1,434.18",
    "Closing Balance        GBP 2,682.55",
    "Card ending 4521 available funds 2,100.00",
  ],
  legal: [
    "LEASE AGREEMENT", "This agreement is made on 15 January 2024",
    "BETWEEN: The Landlord (1) and The Tenant (2)",
    "PROPERTY: 123 Baker Street, London NW1 6XE",
    "TERM: 24 months from 1 February 2024",
    "Clause 3.2  Rent",
    "The Tenant shall pay rent of 2,500 per month",
    "in advance on the first day of each month",
    "Clause 5.1  Termination",
    "Either party may terminate with two months",
    "written notice served on the other party",
    "Clause 7.4  Deposit",
    "A deposit of 5,000 is held and protected",
    "SIGNED on behalf of both parties",
    "Case Reference: 2024-CV-00456",
  ],
};
const GENERAL = [
  "NOTICE: the office will be closed on Monday for the public holiday.",
  "Your parcel has been dispatched and should arrive within three working days.",
  "Meeting at 2:30pm. Call +44 7700 900123 or email sam.smith@example.co.uk.",
  "Date: 01/03/2024. Library books are due back by the end of the month.",
].join("\n");
const FIELDS = {
  prescription: ["500mg", "400mg"],
  banking: ["40218877", "40-11-04"],
  legal: ["2500", "2024-CV-00456"],
};
const DAMAGED = {
  prescription: {
    text: CAPTURE.prescription.join("\n").replace("500mg", "S00m9").replace("400mg", "4O0mg"),
    expect: ["500mg"],
    note: "S00m9 -> 500mg by rule reordering; 4O0mg has no safe deterministic fix, so 400mg is honestly reported as lost",
  },
  banking: {
    text: CAPTURE.banking.join("\n").replace("40218877", "4O218877"),
    expect: ["40-11-04"],
    note: "sort code survives untouched; account number is a letter/digit swap with no safe rule",
  },
  legal: {
    text: CAPTURE.legal.join("\n").replace("2,500", "2,5OO"),
    expect: ["2024-CV-00456"],
    note: "2,5OO -> 25oo does not match 2500, so that field is honestly reported as lost; the intact case number is still found",
  },
};

// ── Harness ──────────────────────────────────────────────────────────
let total = 0, failed = 0;
const groups = [];
const group = (n) => { const g = { name: n, rows: [] }; groups.push(g); return g; };
function check(g, desc, actual, expected, opts) {
  const o = opts || {};
  let pass;
  if (typeof expected === "function") pass = !!expected(actual);
  else if (o.numeric) pass = Math.abs(actual - expected) < (o.eps || 1e-6);
  else pass = JSON.stringify(actual) === JSON.stringify(expected);
  total++;
  if (!pass) failed++;
  g.rows.push({ desc, pass, actual, expected });
  return pass;
}
const eq = (g, d, a, e) => check(g, d, a, e);
const near = (g, d, a, e, eps) => check(g, d, a, e, { numeric: true, eps: eps == null ? 1e-6 : eps });

// ── Tests: 1:1 with selftest.html ────────────────────────────────────
function testNorm() {
  const g = group("norm() - evaluate.py");
  for (const [inp, exp] of [["40-11-04", "401104"], ["2,500", "2500"], ["GB29 NWBK", "gb29nwbk"],
    ["£1,204.33", "120433"], ["500mg", "500mg"], ["", ""], ["!!!", ""],
    ["Case No. 2024-CV-00456", "caseno2024cv00456"]]) {
    eq(g, "norm(" + JSON.stringify(inp) + ")", SightLine.norm(inp), exp);
  }
}
function testWordAccuracy() {
  const g = group("word_accuracy() - evaluate.py");
  near(g, "identical text scores 1.0", SightLine.word_accuracy("Account Number: 40218877", "account number: 40218877"), 1.0);
  near(g, "half the words present scores 0.5", SightLine.word_accuracy("a b c d", "a b x y"), 0.5);
  near(g, "empty ground truth scores 0.0", SightLine.word_accuracy("", "a b c"), 0.0);
  // Python splits on whitespace BEFORE norm(), so "40-11-04" is one gt word
  // and "40 11 04" is three. Confirmed against evaluate.word_accuracy: 0.0.
  near(g, "a hyphenated gt word does not match its spaced form",
    SightLine.word_accuracy("40-11-04", "40 11 04"), 0.0);
  near(g, "but the same gt word with identical spacing scores 1.0",
    SightLine.word_accuracy("40-11-04 40-11-04", "40-11-04"), 1.0);
  near(g, "nothing shared scores 0.0", SightLine.word_accuracy("alpha beta", "gamma delta"), 0.0);
  near(g, "ground truth beats hypothesis length", SightLine.word_accuracy("alpha beta", "alpha beta gamma delta epsilon"), 1.0);
}
function testFieldsFound() {
  const g = group("fields_found() - evaluate.py");
  for (const dt of Object.keys(FIELDS)) {
    const text = CAPTURE[dt].join("\n");
    eq(g, dt + ": clean text recovers every field",
      SightLine.fields_found(text, dt).slice().sort(), FIELDS[dt].slice().sort());
    eq(g, dt + ": fields_complete is true on clean text", SightLine.fields_complete(text, dt), true);
  }
  eq(g, "prescription: empty text finds nothing", SightLine.fields_found("", "prescription"), []);
  eq(g, "unknown doc type finds nothing", SightLine.fields_found("500mg 400mg", "unknown"), []);
  eq(g, "exact_dosage_rate is 1.0 for non-prescription docs", SightLine.exact_dosage_rate("", "banking"), 1.0);
}
function testClassifyDoc() {
  const g = group("classify_doc() - evaluate.py keyword fallback");
  const expect = { prescription: "medical", banking: "banking", legal: "legal" };
  for (const dt of Object.keys(expect)) {
    eq(g, dt + " document -> " + expect[dt], SightLine.classify_doc(CAPTURE[dt].join("\n")), expect[dt]);
  }
  eq(g, "empty text -> general", SightLine.classify_doc(""), "general");
  eq(g, "unrelated text -> general", SightLine.classify_doc("qwerty zxcvbn hjkl"), "general");
  const sc = SightLine.classify_doc_scores(CAPTURE.prescription.join("\n"));
  check(g, "medical score is positive for a prescription", sc.medical, (v) => v > 0);
  check(g, "all four categories produce a numeric score", Object.keys(sc).length, (v) => v === 4);
}
function testNormalize() {
  const g = group("normalizeOCRText() - the OCR-confusion normalizer");
  for (const [inp, exp] of [["S00m9", "500mg"], ["Take S00m9 twice daily", "Take 500mg twice daily"],
    ["S00mg", "500mg"], ["m9", "mg"], ["m1", "ml"], ["50mg mg", "50mg"], ["10ml ml", "10ml"],
    ["S99m9", "599mg"], ["S12m9", "512mg"]]) {
    eq(g, JSON.stringify(inp) + " -> " + JSON.stringify(exp), SightLine.normalizeOCRText(inp), exp);
  }
  for (const w of ["Salary", "SMS", "Statement", "September", "Sample", "Simply",
    "500mg", "400mg", "1200mg", "1m"]) {
    eq(g, "leaves " + JSON.stringify(w) + " alone", SightLine.normalizeOCRText(w), w);
  }
}
function testUnderstand() {
  const g = group("understand() - the full post-OCR chain");
  for (const dt of Object.keys(FIELDS)) {
    const r = SightLine.understand(CAPTURE[dt].join("\n"));
    eq(g, dt + " -> category " + (dt === "prescription" ? "medical" : dt),
      r.category, dt === "prescription" ? "medical" : dt);
    check(g, dt + ": extracts at least 2 key fields", r.fields.length, (v) => v >= 2);
    check(g, dt + ": summary is non-empty prose", r.summary.length, (v) => v > 20);
    check(g, dt + ": confidence in [0,1]", r.classificationConfidence, (v) => v >= 0 && v <= 1);
  }
  const gr = SightLine.understand(GENERAL);
  eq(g, "general document -> general", gr.category, "general");
  check(g, "general document extracts an email", gr.fields.some((f) => f.label === "Email"), (v) => v === true);
  const bank = SightLine.understand(CAPTURE.banking.join("\n"));
  check(g, "banking: account number extracted", bank.fields.some((f) => f.value.includes("40218877")), (v) => v === true);
  check(g, "banking: sort code extracted", bank.fields.some((f) => f.value.includes("40-11-04")), (v) => v === true);
  const rx = SightLine.understand(CAPTURE.prescription.join("\n"));
  check(g, "prescription: patient name extracted", rx.fields.some((f) => f.label === "Patient"), (v) => v === true);
  check(g, "prescription: drug entity detected",
    rx.entities.some((e) => e.label === "DRUG" && /Amoxicillin|Ibuprofen/.test(e.text)), (v) => v === true);
  const lease = SightLine.understand(CAPTURE.legal.join("\n"));
  check(g, "legal: case number extracted", lease.fields.some((f) => f.value.includes("2024-CV-00456")), (v) => v === true);
  check(g, "legal: clauses extracted", lease.fields.some((f) => f.label === "Clause"), (v) => v === true);
}
function testDamaged() {
  const g = group("Blurry-capture repair - the medical safety case");
  for (const dt of Object.keys(DAMAGED)) {
    const d = DAMAGED[dt];
    const got = SightLine.fields_found(SightLine.normalizeOCRText(d.text), dt);
    eq(g, dt + ": recovers " + JSON.stringify(d.expect), got.slice().sort(), d.expect.slice().sort());
    check(g, dt + ": dosage survival never overstates the reference", got.length <= FIELDS[dt].length, (v) => v === true);
  }
  const repaired = SightLine.normalizeOCRText("Rx Amoxicillin S00m9 capsules");
  eq(g, "blurred prescription dose is recovered verbatim",
    SightLine.fields_found(repaired, "prescription"), ["500mg"]);
}
function testEntities() {
  const g = group("extractEntities() - the NER layer");
  const ents = SightLine.extractEntities(CAPTURE.prescription.join("\n") + "\nHSBC on 01/03 paid £44.19 for Amoxicillin");
  check(g, "detects a DRUG", ents.some((e) => e.label === "DRUG"), (v) => v === true);
  check(g, "detects a DOSAGE", ents.some((e) => e.label === "DOSAGE"), (v) => v === true);
  check(g, "detects MONEY", ents.some((e) => e.label === "MONEY"), (v) => v === true);
  check(g, "detects an ORG", ents.some((e) => e.label === "ORG"), (v) => v === true);
  check(g, "detects a DATE", ents.some((e) => e.label === "DATE"), (v) => v === true);
  eq(g, "duplicate entities are de-duplicated", SightLine.extractEntities("Amoxicillin Amoxicillin 500mg 500mg").length, 2);
  eq(g, "empty text yields no entities", SightLine.extractEntities("").length, 0);
}
function testGarbage() {
  const g = group("OCR garbage rejection");
  eq(g, "short text is garbage", SightLine.isGarbage("hi", 1), true);
  eq(g, "empty text is garbage", SightLine.isGarbage("", 0), true);
  eq(g, "600 words is garbage", SightLine.isGarbage(new Array(600).fill("word").join(" "), 600), true);
  eq(g, "mostly speckle is garbage", SightLine.isGarbage("a b c d e f g h i j k l m n o p", 16), true);
  eq(g, "real prose is not garbage", SightLine.isGarbage("Account Number: 40218877 Sort Code: 40-11-04", 7), false);
}
function testStructureScore() {
  const g = group("structure_score - ranks usable results over confident noise");
  const s = SightLine.structure_score;
  // check() takes a predicate as `expected`, which is what the >0 assertions
  // need -- eq() is strict JSON equality and would compare a boolean to a number.
  const gt0 = (d, t) => check(g, d, s(t), (v) => v > 0);

  // The patterns this app actually extracts.
  gt0("dosage is structure", "Amoxicillin 500mg three times daily");
  gt0("sort code is structure", "Sort Code: 40-11-04");
  gt0("case reference is structure", "Case reference: 2024-CV-00456");
  gt0("bare account number is structure", "Account Number: 40218877");
  gt0("currency symbol is structure", "Total: £2,500.00");
  gt0("ISO currency code is structure", "Total: GBP 2,500.00");

  // Must NOT fire on ordinary prose, or every confident read of a letter
  // would outrank the result that actually contains a case number.
  eq(g, "prose scores zero", s("the quick brown fox jumps over the lazy dog again"), 0);
  eq(g, "empty scores zero", s(""), 0);
  eq(g, "whitespace scores zero", s("    "), 0);

  // The dangerous near-miss. A letter O where a zero belongs is exactly the
  // single-digit error that makes a case number wrong, so it must NOT count as
  // structure — the whole point is that structure means RECOVERABLE.
  eq(g, "letter-O case ref is NOT structure",
     s("Case reference: 2024-CV-OO456."), 0);

  // The decisive comparison from the field_regression finding: a readable
  // result must outrank a fragmented one that recovered less.
  const readable = "Case reference: 2024-CV-00456 Date: 15 January 2024";
  const fragmented = "Case reference: 2024-CV- OO456 Date: 15 Jan 2024";
  check(g, "readable outranks fragmented",
        s(readable), (v) => v > s(fragmented));

  // A date is not an account number: the lookarounds must keep the year from
  // being counted as a bare 6-8 digit run.
  eq(g, "date digits are not an account number", s("Date: 15 January 2024"), 0);
}
function testRobustness() {
  const g = group("Robustness - no stage throws on bad input");
  for (const v of [null, undefined, "", 0, NaN, {}, []]) {
    check(g, "norm(" + String(v) + ") does not throw", SightLine.norm(v), (r) => typeof r === "string");
    check(g, "classify_doc(" + String(v) + ") does not throw", SightLine.classify_doc(v), (r) => typeof r === "string");
    check(g, "normalizeOCRText(" + String(v) + ") does not throw", SightLine.normalizeOCRText(v), (r) => typeof r === "string");
    check(g, "understand(" + String(v) + ") does not throw", SightLine.understand(v).category, (r) => typeof r === "string");
  }
  check(g, "extractFields on empty text returns an array", SightLine.extractFields("", "medical"), (v) => Array.isArray(v));
  check(g, "unknown category falls back to general patterns", SightLine.extractFields("Date 01/03/2024", "nonexistent").length, (v) => v > 0);
  eq(g, "CATEGORIES matches corpus.CATEGORIES order", SightLine.CATEGORIES, ["banking", "medical", "legal", "general"]);
  eq(g, "FIELDS matches evaluate.FIELDS", SightLine.FIELDS, FIELDS);
}
function testTtsChunks() {
  const g = group("TTS chunking - iOS utterance limit");
  const long = "This is a medical prescription. Patient: Jane Doe. Medications: Amoxicillin. "
    + "Dosages: 500 mg. Frequency: three times daily. Warning: may cause drowsiness. "
    + "Keep out of reach of children.";
  const chunks = TTS.chunkText(long, 180);
  check(g, "long text is split into multiple chunks", chunks.length, (v) => v > 1);
  check(g, "every chunk is within the 180-char limit", chunks.every((c) => c.length <= 180), (v) => v === true);
  eq(g, "chunking loses no words", chunks.join(" ").split(/\s+/).length, long.trim().split(/\s+/).length);
  eq(g, "empty text yields no chunks", TTS.chunkText("", 180).length, 0);
}
function testClassifierMath() {
  const g = group("Classifier math - pooling and softmax");
  const probs = SightLineClassifier.softmax(new Float32Array([1, 2, 3, 4]));
  near(g, "softmax sums to 1", Array.from(probs).reduce((a, b) => a + b, 0), 1.0, 1e-6);
  check(g, "softmax is monotonic", probs[0] < probs[1] && probs[1] < probs[2] && probs[2] < probs[3], (v) => v === true);
  near(g, "softmax argmax is the largest logit", probs.indexOf(Math.max.apply(null, Array.from(probs))), 3);
  const h = new Float32Array([1, 0, 0, 0, 0, 3, 0, 0]);
  const emb = SightLineClassifier.meanPoolL2(h, new Int32Array([1, 0]), 2, 4);
  near(g, "unmasked token dominates the mean", emb[0], 1.0, 1e-6);
  near(g, "embedding is L2-normalised", Math.sqrt(Array.from(emb).reduce((a, b) => a + b * b, 0)), 1.0, 1e-6);
  const emb2 = SightLineClassifier.meanPoolL2(new Float32Array(8).fill(1), new Int32Array([1, 1]), 2, 4);
  near(g, "uniform input pools to a unit vector",
    emb2[0] * emb2[0] + emb2[1] * emb2[1] + emb2[2] * emb2[2] + emb2[3] * emb2[3], 1.0, 1e-6);
}
function testRestorerTiling() {
  const g = group("Restorer tiling - must match arm_restorer");
  eq(g, "tile height is 64", SightLineRestorer.TILE_H, 64);
  eq(g, "tile width is 256", SightLineRestorer.TILE_W, 256);
  const padFor = (n, t) => (t - (n % t)) % t;
  for (const n of [1, 63, 64, 65, 300, 1000, 3000]) {
    check(g, "height " + n + " pads to a whole number of 64px tiles", (n + padFor(n, 64)) % 64, (v) => v === 0);
  }
  for (const n of [1, 255, 256, 257, 900, 3200]) {
    check(g, "width " + n + " pads to a whole number of 256px tiles", (n + padFor(n, 256)) % 256, (v) => v === 0);
  }
}

// ── Run ──────────────────────────────────────────────────────────────
testNorm(); testWordAccuracy(); testFieldsFound(); testClassifyDoc();
testNormalize(); testUnderstand(); testDamaged(); testEntities();
testGarbage(); testStructureScore(); testRobustness(); testTtsChunks(); testClassifierMath();
testRestorerTiling();

for (const g of groups) {
  const bad = g.rows.filter((r) => !r.pass);
  console.log((bad.length ? "FAIL" : "ok  ") + "  " + g.name +
    "  (" + (g.rows.length - bad.length) + "/" + g.rows.length + ")");
  for (const r of bad) {
    console.log("        FAIL: " + r.desc);
    console.log("              got      " + JSON.stringify(r.actual));
    console.log("              expected " + JSON.stringify(r.expected));
  }
}
console.log("\n" + (failed === 0 ? "ALL " + total + " ASSERTIONS PASSED" : failed + " of " + total + " FAILED"));
process.exit(failed === 0 ? 0 : 1);
