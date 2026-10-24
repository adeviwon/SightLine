// Aria's pure document/field layer, adapted to SightLine's existing result API.
// No OCR engine replacement. Medical extraction and whole-document speech stay intact.
import { extractAll, gateField } from "./aria/fields.mjs";
const labels = {
  merchant: "Shop",
  shop_address: "Address",
  shop_phone: "Phone",
  category: "Kind of shop",
  purchase_date: "Date",
  purchase_time: "Time",
  subtotal: "Before tax",
  tax: "Tax",
  total: "Total",
  payment_method: "Payment method",
  card_last4: "Card ending",
  receipt_number: "Receipt number",
  transaction_id: "Transaction number",
};
export function receiptResult(ocr) {
  const text = String(ocr.text || "");
  const measured = Number.isFinite(ocr.confidence)
    ? Math.max(0, Math.min(1, ocr.confidence))
    : 0;
  const extraction = extractAll(text, {
    ocr: {
      mean_confidence: measured * 100,
      boxes: (ocr.words || []).map((w) => ({
        ...w,
        confidence: w.confidence * 100,
      })),
    },
  });
  const decision = extraction.docType;
  const diagnostics = {
    source: "aria-weighted-keywords",
    ...decision,
    ocrConfidence: measured,
    confidenceSource: "live-inference",
    rejected: [],
  };
  // Statements contain merchant names, totals and card entries too. They must
  // retain SightLine's banking path rather than borrowing the receipt layout.
  const statement =
    /\b(?:statement\s+of\s+account|bank\s+statement)\b/i.test(text) ||
    (/\bopening\s+balance\b/i.test(text) &&
      /\bclosing\s+balance\b/i.test(text)) ||
    (/\bsort\s+code\b/i.test(text) && /\baccount\s+number\b/i.test(text));
  if (statement) {
    diagnostics.integrationRejection =
      "bank statement evidence takes priority over shared receipt keywords";
    return { result: null, diagnostics };
  }
  if (decision.type !== "receipt" || decision.confidence < 0.75)
    return { result: null, diagnostics };
  const candidates = Object.values(extraction.fields);
  const safeLines = String(ocr.fieldText ?? ocr.speechText ?? "").split("\n");
  // A company-marked letterhead is stronger shop evidence than a personal heading.
  const company = safeLines
    .slice(0, 6)
    .find((l) => /\b(?:sdn\s*bhd|limited|ltd|inc|llc)\b/i.test(l));
  if (company) {
    const merchant = candidates.find((f) => f.key === "merchant");
    if (merchant)
      Object.assign(merchant, {
        value: company,
        raw: company,
        line: company,
        confidence: Math.min(0.86, measured),
      });
  }
  // Aria requires currency-marked amounts. Receipts often print only "Total 9.00".
  // Accept a literal, labelled decimal; never infer its currency or repair digits.
  for (const [key, re] of [
    [
      "total",
      /^\s*(?:grand\s+total|total|amount\s+paid)\s*[:=]?\s*((?:(?:HK\$|US\$|RM|GBP|USD|HKD|MYR|£|\$|€)\s*)?\d+(?:,\d{3})*\.\d{2})\s*$/i,
    ],
  ]) {
    const matches = text
      .split("\n")
      .map((line) => ({ line, match: line.match(re) }))
      .filter((x) => x.match);
    if (matches.length && new Set(matches.map((x) => x.match[1])).size === 1) {
      const existing = candidates.findIndex((f) => f.key === key);
      if (existing !== -1) candidates.splice(existing, 1);
      candidates.push({
        key,
        kind: "money",
        value: matches[0].match[1],
        raw: matches[0].match[1],
        line: matches[0].line,
        valid: true,
        confidence: Math.min(0.85, measured),
      });
    } else if (new Set(matches.map((x) => x.match[1])).size > 1) {
      const existing = candidates.findIndex((f) => f.key === key);
      if (existing !== -1) candidates.splice(existing, 1);
      diagnostics.rejected.push({ key, reason: "conflicting labelled totals" });
    }
  }
  diagnostics.missing = ["purchase_date", "total"]
    .filter((key) => !candidates.some((f) => f.key === key))
    .map((key) => ({ key, reason: "no valid, unambiguous recognised value" }));
  const fields = [];
  for (const f of candidates) {
    let reason = null;
    if (!safeLines.includes(f.line))
      reason = "source line contains uncertain or missing recognition";
    if (
      f.key === "shop_phone" &&
      !/\b(?:tel|phone|telephone|mobile|call)\b/i.test(f.line || "")
    )
      reason = "number has no phone label";
    // Use source spelling for dates/times, rather than quietly normalising an uncertain year or AM/PM.
    const local = (ocr.words || []).filter((w) => f.line?.includes(w.text));
    const confidence = Math.min(
      f.confidence,
      ...local.map((w) =>
        Number.isFinite(w.confidence)
          ? Math.max(0, Math.min(1, w.confidence))
          : 0,
      ),
    );
    const gate = gateField({ ...f, confidence });
    if (!gate.speak) reason = gate.reason;
    if (reason) {
      diagnostics.rejected.push({ key: f.key, reason, confidence });
      continue;
    }
    fields.push({
      label: labels[f.key] || f.key,
      value:
        /date|time/.test(f.key) || f.kind === "money" ? f.raw : String(f.value),
      confidence,
      sourceLine: f.line,
    });
  }
  return {
    diagnostics,
    result: {
      ocrText: text,
      normalized: text,
      category: "receipt",
      categoryLabel: "Receipt",
      classificationConfidence: decision.confidence,
      classificationSource: diagnostics.source,
      documentEvidence: decision.scores.receipt,
      fields,
      entities: [],
      summary: fields.length
        ? "Receipt details. Check important amounts against the original."
        : "Receipt recognised. Some details need a clearer photo.",
      receiptDiagnostics: diagnostics,
    },
  };
}
