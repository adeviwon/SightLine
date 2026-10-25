const { test } = require("node:test");
const a = require("node:assert/strict");
const L = require("../js/field_localization.js");

function rec(category, field) {
  return {
    category,
    ocrText: field.sourceLine || field.value,
    fields: [field],
  };
}
function field(label, value, extra = {}) {
  return { label, value, confidence: 0.9, sourceLine: String(value), ...extra };
}

test('new medicine sections localize form, each dose and duration without changing source', () => {
  for (const [label,value,expected] of [['Medicine type','capsules','膠囊'],['Amount each time','one capsule','一粒膠囊'],['How long','for 7 days','持續 7 日']]) {
    const f=field(label,value), before=JSON.stringify(f);
    a.equal(L.localize(rec('medical',f),f).value,expected);
    a.equal(JSON.stringify(f),before);
  }
});

test('unsupported timing conditions remain complete English instead of being shortened', () => {
  const f=field('When to take it','every 6 hours as required for pain');
  const localized=L.localize(rec('medical',f),f);
  a.equal(localized.value,f.value);
  a.equal(localized.needsVerification,true);
});

test("receipt label and verified PARKnSHOP display localize with English preserved", () => {
  const f = field("Shop", "PARKnSHOP");
  const out = L.localize(rec("receipt", f), f);
  a.equal(out.label, "商店");
  a.equal(out.value, "百佳超級市場（PARKnSHOP）");
  a.equal(out.translated, true);
  a.equal(f.value, "PARKnSHOP");
});

test("unknown merchant and unsupported language remain verbatim and marked", () => {
  const f = field("Shop", "NORTH HARBOUR MARKET");
  const out = L.localize(rec("receipt", f), f);
  a.equal(out.value, f.value);
  a.equal(out.needsVerification, true);
  a.equal(out.lang, "en");
});

test("HK currency spelling and numeric precision are not changed", () => {
  const f = field("Total", "HK$155.36");
  const out = L.localize(rec("receipt", f), f);
  a.equal(out.label, "總金額");
  a.equal(out.value, "HK$155.36");
  a.equal(out.translated, false);
});

test("ISO date localized without changing its digits", () => {
  const f = field("Date", "2025-04-09");
  const out = L.localize(rec("receipt", f), f);
  a.equal(out.value, "2025年04月09日");
  a.deepEqual(out.value.match(/\d+/g), ["2025", "04", "09"]);
});

test("verified payment method and quantity map without changing numbers", () => {
  const cash = field("Payment method", "cash");
  const qty = field("Quantity", "2 pcs");
  a.equal(L.localize(rec("receipt", cash), cash).value, "現金");
  const q = L.localize(rec("receipt", qty), qty);
  a.equal(q.value, "2 件");
  a.equal(q.value.match(/\d+/)[0], "2");
});

test("verified shop category maps while arbitrary categories do not", () => {
  const pharmacy = field("Kind of shop", "Pharmacy");
  const other = field("Kind of shop", "Night market on Queen's Road");
  a.equal(L.localize(rec("receipt", pharmacy), pharmacy).value, "藥房");
  const out = L.localize(rec("receipt", other), other);
  a.equal(out.value, other.value);
  a.equal(out.needsVerification, true);
});

test("dosage units localize while preserving numeric tokens and field confidence", () => {
  const f = field("Dosage", "2.5 mg", { sourceLine: "AMOXICILLIN 2.5 mg" });
  const r = { category: "medical", ocrText: f.sourceLine };
  const out = L.localize(r, f);
  a.equal(out.label, "強度或用量");
  a.equal(out.value, "2.5 毫克");
  a.equal(f.confidence, 0.9);
});

test("only exact supported medical instructions are translated", () => {
  const supported = field("Dosage", "Take one capsule twice daily", {
    sourceLine: "Take one capsule twice daily",
  });
  const unsupported = field("Dosage", "Take one capsule when needed", {
    sourceLine: "Take one capsule when needed",
  });
  a.equal(L.localize({ category: "medical", ocrText: supported.value }, supported).value,
    "每日兩次，每次一粒膠囊");
  const out = L.localize({ category: "medical", ocrText: unsupported.value }, unsupported);
  a.equal(out.value, unsupported.value);
  a.equal(out.needsVerification, true);
});

test("unknown medical warning remains English with a truthful label", () => {
  const f = field("Warning", "Consult your doctor before use");
  const out = L.localize(rec("medical", f), f);
  a.equal(out.value, f.value);
  a.equal(out.needsVerification, true);
});

test("low confidence or values absent from OCR text are never localized", () => {
  const weak = field("Payment method", "cash", { confidence: 0.74 });
  const invented = field("Payment method", "cash", { sourceLine: "CASH" });
  a.equal(L.localize(rec("receipt", weak), weak).value, "cash");
  a.equal(L.localize({ category: "receipt", ocrText: "Card" }, invented).value, "cash");
  a.equal(L.localize({ category: "receipt", ocrText: "Card" }, invented).needsVerification, true);
});

test("numeric source matching rejects a value embedded in a different amount", () => {
  const f = field("Dosage", "5 mg", { sourceLine: "15 mg" });
  const out = L.localize({ category: "medical", ocrText: "AMOXICILLIN 15 mg" }, f);
  a.equal(out.value, "5 mg");
  a.equal(out.needsVerification, true);
});

test("English display preserves original labels, values, and language", () => {
  const f = field("Total", "HK$155.36");
  a.deepEqual(L.localize(rec("receipt", f), f, "en-GB"), {
    label: "Total", value: "HK$155.36", lang: "en", translated: false, retained: true,
  });
});
