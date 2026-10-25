import { test } from "node:test";
import assert from "node:assert/strict";
import { receiptResult } from "../js/receipt.mjs";
import { classifyDocument } from "../js/aria/fields.mjs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url),
  pipeline = require("../js/pipeline.js");
function read(text, confidence = 0.94) {
  return {
    text,
    fieldText: text,
    speechText: text,
    confidence,
    words: text.split("\n").map((text) => ({ text, confidence })),
  };
}
test("clear receipt gets its own category and source-backed total", () => {
  const { result } = receiptResult(
    read("GREENFIELD MARKET\nRECEIPT 12345\nTotal £9.00\nCASH"),
  );
  assert.equal(result.category, "receipt");
  assert.equal(result.fields.find((f) => f.label === "Total").value, "£9.00");
});
test("literal bare total preserves amount without inventing a currency", () => {
  const { result } = receiptResult(
    read("BOOK STORE\nCASH RECEIPT\nTotal 9.00"),
  );
  assert.equal(result.category, "receipt");
  assert.ok(result.fields.some((f) => f.label === "Total"));
  assert.ok(!JSON.stringify(result.fields).includes("MYR"));
});
test("a 94 percent OCR score does not force unreadable text into receipt classification", () => {
  assert.equal(receiptResult(read("XQ ZP 123 00 AA")).result, null);
  assert.equal(
    classifyDocument("CARD PAYMENT AMOUNT DUE").type,
    "unrecognised",
  );
});
test("prescription stays on existing medical path with its own medicine/dose guards", () => {
  const text =
    "PRESCRIPTION\nAmoxicillin 500 mg capsules\nTake one capsule twice a day\nPatient: Jane Example";
  assert.equal(receiptResult(read(text)).result, null);
  assert.equal(classifyDocument(text).type, "prescription");
  assert.equal(pipeline.understand(text).category, "medical");
});
test("uncertain source lines never become confirmed receipt fields", () => {
  const o = read("BOOK STORE\nCASH RECEIPT\nTotal 9.00");
  o.fieldText = "BOOK STORE\nCASH RECEIPT";
  const { result, diagnostics } = receiptResult(o);
  assert.ok(!result.fields.some((f) => f.label === "Total"));
  assert.ok(diagnostics.rejected.some((f) => f.key === "total"));
});
test("low local confidence overrules high page confidence", () => {
  const o = read("BOOK STORE\nCASH RECEIPT\nTotal 9.00");
  o.words.at(-1).confidence = 0.6;
  const { result } = receiptResult(o);
  assert.ok(!result.fields.some((f) => f.label === "Total"));
});
test("product barcodes are never represented as shop phone numbers", () => {
  const { result } = receiptResult(
    read("BOOK STORE\nCASH RECEIPT\n9556939040118 MODELLING CLAY\nTotal 9.00"),
  );
  assert.ok(!result.fields.some((f) => f.label === "Phone"));
});
test("missing OCR confidence fails closed", () => {
  const o = read("BOOK STORE\nCASH RECEIPT\nTotal 9.00");
  delete o.confidence;
  assert.deepEqual(receiptResult(o).result.fields, []);
});

test("conflicting totals are omitted rather than choosing one", () => {
  const { result } = receiptResult(
    read("BOOK STORE\nCASH RECEIPT\nTotal 9.00\nTotal 90.00"),
  );
  assert.ok(!result.fields.some((f) => f.label === "Total"));
});

test("bank statements with totals and shop names retain the banking pipeline", () => {
  const text =
    "HSBC BANK\nStatement of Account\nAccount Number: 40218877\nSort Code: 40-11-04\nOpening Balance GBP 1204.33\nTESCO STORES GBP 44.19\nWithdrawals total GBP 1434.18\nClosing Balance GBP 2682.55\nCard ending 4521";
  assert.equal(receiptResult(read(text)).result, null);
  assert.equal(pipeline.understand(text).category, "banking");
});

test("invalid local confidence cannot bypass the field gate", () => {
  const o = read("BOOK STORE\nCASH RECEIPT\nTotal 9.00");
  o.words.at(-1).confidence = NaN;
  assert.ok(!receiptResult(o).result.fields.some((f) => f.label === "Total"));
});
