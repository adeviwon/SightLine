const { test } = require("node:test"),
  a = require("node:assert/strict"),
  C = require("../js/cantonese_summary.js");
function receipt(value = "HK$155.36") {
  return {
    category: "receipt",
    fields: [
      {
        label: "Shop",
        value: "PARKnSHOP",
        sourceLine: "PARKnSHOP",
        confidence: 0.9,
      },
      { label: "Total", value, sourceLine: "TOTAL " + value, confidence: 0.9 },
    ],
  };
}
test("receipt uses Hong Kong Traditional Chinese and exact HKD cents", () => {
  let r = C.build(receipt());
  a.ok(r.eligible);
  a.match(r.text, /百佳/);
  a.match(r.text, /一百五十五蚊三毫六仙/);
});
test("unknown brand is preserved verbatim", () => {
  let r = receipt();
  r.fields[0].value = "Unlisted Store";
  a.match(C.build(r).text, /Unlisted Store/);
});
test("uncertain totals are never spoken", () => {
  let r = receipt();
  r.fields[1].confidence = 0.7;
  a.ok(!C.build(r).text.includes("一百五十五"));
});
test("unmarked currency is not invented as HKD", () => {
  a.match(C.amount("155.36"), /貨幣未確認/);
  a.ok(!C.amount("155.36").includes("蚊"));
});
test("fractional medicine strengths retain their decimal", () => {
  let r = {
    category: "medical",
    speechText: "AMOXICILLIN\n2.5 mg\nTake one capsule twice daily",
    entities: [{ label: "DRUG", text: "Amoxicillin", confidence: 0.9 }],
    fields: [{ label: "Dosage", value: "2.5 mg", confidence: 0.9 }],
  };
  a.match(C.build(r).text, /二點五毫克/);
  a.match(C.build(r).text, /每次一粒膠囊/);
});
test("medical incompleteness remains a refusal", () =>
  a.equal(
    C.build({ category: "medical", speechText: "", ocrText: "500mg" }).eligible,
    false,
  ));
test("multiple medicines cannot be paired to one dose", () =>
  a.equal(
    C.build({
      category: "medical",
      speechText: "DrugA DrugB 500mg",
      entities: [
        { label: "DRUG", text: "DrugA", confidence: 0.9 },
        { label: "DRUG", text: "DrugB", confidence: 0.9 },
      ],
      fields: [],
    }).eligible,
    false,
  ));
test("negated instructions never match a positive phrase", () => {
  let r = {
    category: "medical",
    speechText: "AMOXICILLIN\nDo not take one capsule twice daily",
    entities: [{ label: "DRUG", text: "Amoxicillin", confidence: 0.9 }],
    fields: [],
  };
  a.ok(!C.build(r).text.includes("每日兩次"));
});
test("malformed prices and uncertain unknown document types are refused", () => {
  a.equal(C.amount("1.2.3"), null);
  a.equal(C.build({ category: "general" }).eligible, false);
});

test("malformed separators are never repaired into a different price", () =>
  a.equal(C.amount("1,55.36"), null));

test("Cantonese medical copy does not guess whether a quantity is strength or dose", () => {
  const r = {
    category: "medical",
    speechText: "AMOXICILLIN\nTake 5 mg",
    entities: [{ label: "DRUG", text: "Amoxicillin", confidence: 0.9 }],
    fields: [{ label: "Dosage", value: "5 mg", confidence: 0.9 }],
  };
  a.match(C.build(r).text, /強度或用量/);
});
test("PDF scope is retained in the Cantonese summary", () => {
  let r = receipt();
  r.pageNote = "Page 1 of 2";
  a.match(C.build(r).text, /只包括目前辨識嘅文件頁面/);
});

test("Cantonese receipt speaks an unambiguous ISO date in Chinese", () => {
  const r = {
    category: "receipt",
    fields: [{ label: "Date", value: "2025-04-09", sourceLine: "2025-04-09", confidence: 0.9 }],
  };
  a.match(C.build(r).text, /二千零二十五年四月九日/);
});

test("Cantonese receipt summary supports subtotal and discount", () => {
  const r = {
    category: "receipt",
    fields: [
      { label: "Subtotal", value: "HK$100.00", sourceLine: "HK$100.00", confidence: 0.9 },
      { label: "Discount", value: "HK$5.00", sourceLine: "HK$5.00", confidence: 0.9 },
    ],
  };
  const text = C.build(r).text;
  a.match(text, /小計係一百蚊/);
  a.match(text, /折扣係五蚊/);
});

test("Cantonese receipt speaks supported quantities without losing the count", () => {
  const r = {
    category: "receipt",
    fields: [{ label: "Quantity", value: "2 pcs", sourceLine: "Qty 2 pcs", confidence: 0.9 }],
  };
  a.match(C.build(r).text, /數量係二件/);
});
test("missing IndexedDB fails honestly without claiming a save", async () => {
  const H = require("../js/history.js");
  a.equal(typeof indexedDB, "undefined");
  await a.rejects(H.list(), /unavailable/);
});
