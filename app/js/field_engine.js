/**
 * SightLine — the FIELD ENGINE: the two already-ported halves, composed into the
 * measured best extraction path.
 *
 * WHY THIS EXISTS. The app ships the WEAKEST of the three measured extraction
 * paths. On the correct ground truth (the value the renderer actually DREW,
 * after the label that was spotted), 120 held-out pages / 696 true label-value
 * pairs:
 *
 *   (a) whole-page regex         173 spoken  precision 0.965  recall 0.240
 *   (b) spot + typed validation  704 spoken  precision 0.608  recall 0.615
 *   (c) the speak/refuse policy  686 spoken  precision 0.793  recall 0.705
 *       and wrong-but-spoken 276/704 (39%) -> 142; typed-only 14/323 (4.3%),
 *       with ZERO wrong dosages.
 *
 * (a) is what pipeline.js runs today. (b) and (c) are already IN the app —
 * app/js/field_spot.js (label spotting + the proximity read) and
 * app/js/field_guard.js (validate + decide) — each parity-proven against the
 * Python it was ported from, and NOTHING CALLS THEM. This module is the one
 * missing piece: the composition. It re-derives NO rule; every rule is the
 * ported modules' own.
 *
 * WHAT IT IS, precisely: for each line, field_spot spots labels; field_spot's
 * proximity read takes the value anchored to the label (same line to the right
 * of the span, else the line just below where the box geometry allows);
 * field_guard.validate checks the kind and carries the human-readable reason;
 * field_guard.decide makes the speak/refuse call over BOTH paths.
 *
 * THE DEFAULT IS TYPED-ONLY (speakFreeText=false). That is Lane N's measured
 * recommendation, not a preference: turning free text back on buys +0.26 recall
 * for 128 extra wrong free-text values, so free text is an opt-in mode. See
 * ml/src/field_policy.py, whose docstring carries the whole table.
 *
 * THE LINE BOX IS OPTIONAL, AND ITS ABSENCE IS MEANINGFUL. The proximity read's
 * cross-line direction needs real page geometry; a line with no `box` falls back
 * to "the next line in reading order", which is field_proximity's own documented
 * box-is-None branch (`_next_line_text`). The app's OCR layer currently returns
 * only a joined string (see pipeline.js), so in the shipped wiring every line has
 * NO box and the li+1 branch is what runs. `extract()` therefore accepts boxes
 * and proves the geometry direction works, while the app path exercises the
 * fallback — the behaviour is a property of the INPUT, not a second detector.
 *
 * PORTING RULES: none of this module's own. No regex at all, so no look-behind
 * (a parse-time SyntaxError on Safari < 16.4) and nothing for
 * app/tools/ios_compat_scan.js to find. IIFE + globalThis.SightLine* +
 * module.exports, like every other file in app/js (there is no package.json, so
 * a bare `export` could not be require()d by the tools).
 *
 * VERIFIED BY app/tools/field_engine_contract.js (synthetic pages, known lines).
 */

"use strict";

const FieldEngine = (() => {

  const FS = (typeof module !== "undefined" && module.exports)
    ? require("./field_spot.js")
    : (typeof globalThis !== "undefined" ? globalThis.SightLineFieldSpot : undefined);
  const FG = (typeof module !== "undefined" && module.exports)
    ? require("./field_guard.js")
    : (typeof globalThis !== "undefined" ? globalThis.SightLineFieldGuard : undefined);

  /**
   * Lane N's recommendation, made an executable default rather than a comment.
   * `threshold: null` means field_spot's own default (0.75, the precision-first
   * operating point); `docType: null` disables field_guard's vocabulary gate.
   */
  const DEFAULT_OPTS = {
    threshold: null,
    docType: null,
    regexFields: null,
    kinds: null,
    speakFreeText: false,      // typed-only: free text is OPT-IN
    strictCritical: false,
  };

  /** A line is {text, box?}; a bare string is read as {text}. */
  function asLine(ln) {
    if (ln === null || ln === undefined) return { text: "" };
    if (typeof ln === "string") return { text: ln };
    return ln;
  }

  /** Accept a page as [{text, box?}], [string], or a newline-separated string. */
  function linesOf(page) {
    if (Array.isArray(page)) return page.map(asLine);
    if (typeof page === "string") {
      return String(page).split("\n").map(function (t) { return { text: t }; });
    }
    return [];
  }

  /**
   * spot -> read -> validate -> decide, on one page.
   *
   * Returns { spots, fields, decisions, spoken }:
   *   spots      field_spot.spotInLines output (label/start/end/score/line_index)
   *   fields     field_spot.extractFields output
   *              {name: {value, ok, kind, reason, provenance}} — `reason` is the
   *              human-readable refusal the guard's validator produced
   *   decisions  field_guard.decide output, each carrying its why
   *   spoken     the values a caller would read out, in decision order
   */
  function extract(page, opts) {
    const o = Object.assign({}, DEFAULT_OPTS, opts || {});
    const lines = linesOf(page);

    // field_spot.spotInLines is pinned to field_spot's own default threshold.
    // Passing one through means re-running its spotLabels per line, which is the
    // same call spotInLines makes — no rule is re-derived, only the threshold.
    const spots = (o.threshold === null || o.threshold === undefined)
      ? FS.spotInLines(lines)
      : lines.reduce(function (acc, ln, i) {
          for (const s of FS.spotLabels(ln.text, o.threshold)) {
            acc.push({ label: s.label, start: s.start, end: s.end, score: s.score,
                       line_index: i });
          }
          return acc;
        }, []);

    const fields = FS.extractFields(lines, spots);
    const decisions = FG.decide(o.regexFields || {}, fields, {
      kinds: o.kinds,
      doc_type: o.docType,
      strict_critical: o.strictCritical,
      speak_free_text: o.speakFreeText,
    });
    return { spots: spots, fields: fields, decisions: decisions, spoken: FG.spoken(decisions) };
  }

  /** extract() from a page of TEXT (newline-separated lines, no boxes). */
  function extractFromText(text, opts) {
    return extract(text === null || text === undefined ? "" : String(text), opts);
  }

  return { DEFAULT_OPTS, extract, extractFromText, linesOf, asLine };
})();

if (typeof module !== "undefined" && module.exports) module.exports = FieldEngine;
if (typeof globalThis !== "undefined") globalThis.SightLineFieldEngine = FieldEngine;
