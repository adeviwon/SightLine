/** One pure eligibility policy shared by receipt rendering and playback.
 * Medical whole-document verification remains separate from receipt fields.
 */
const SightLineSpeech = (() => {
  function plan(result) {
    if (!result)
      return {
        eligible: false,
        text: "",
        reason: "Scan a document before reading aloud.",
      };
    if (result.category === "receipt") {
      const fields = (result.fields || []).filter(
        (f) =>
          Number.isFinite(f.confidence) &&
          f.confidence >= 0.75 &&
          String(f.value || "").trim() &&
          String(f.sourceLine || "").trim(),
      );
      if (!fields.length)
        return {
          eligible: false,
          text: "",
          reason:
            "No receipt details are clear enough to read aloud. Try a clearer photo.",
        };
      const details = fields.map((f) => `${f.label}: ${f.value}.`).join(" ");
      const missing =
        (result.receiptDiagnostics?.rejected || []).length ||
        (result.receiptDiagnostics?.missing || []).length;
      return {
        eligible: true,
        source: "verified-receipt-fields",
        text:
          "Receipt. " +
          details +
          (missing
            ? " Some details were unclear and have not been read. Check important amounts against the original receipt."
            : " Check important amounts against the original receipt."),
      };
    }
    if (result.engine?.ocr === "ppocrv5" && !result.speechText)
      return {
        eligible: false,
        text: "",
        reason:
          "Medicine details or other words need checking. Try a clearer photo, or ask someone to check the document.",
      };
    const text = result.speechText || result.ocrText || "";
    return {
      eligible: !!text.trim(),
      source: "complete-document",
      text,
      reason: text.trim() ? "" : "No clear words are available to read.",
    };
  }
  return { plan };
})();
if (typeof module !== "undefined" && module.exports)
  module.exports = SightLineSpeech;
if (typeof window !== "undefined") window.SightLineSpeech = SightLineSpeech;
