/**
 * SightLine — closed-vocabulary LABEL SPOTTING + the label-PROXIMITY value read
 * (JS port).
 *
 * Ports, semantically exactly:
 *   ml/src/label_spot.py      — spot_labels(text, threshold) -> [{label, start,
 *                               end, score}] (start/end are character offsets of
 *                               the MATCHED REGION, end exclusive) and
 *                               spot_in_lines(lines).
 *   ml/src/field_proximity.py — the PROXIMITY READ only: _trim_index, value_after,
 *                               _next_line_text and extract_fields.
 *
 * The VALIDATORS and the SPEAK/REFUSE policy are NOT here and are NOT duplicated:
 * extract_fields() calls app/js/field_guard.js (SightLineFieldGuard), which is
 * the already-ported half (its own 898 golden vectors prove it).
 *
 * WHY THE APP NEEDS THIS: the measured best path does not exist in the app yet.
 * label-anchored extraction + typed validation + the speak/refuse policy gives
 * 61.5% recall of the values the renderer actually drew, against 24.0% for the
 * whole-page regex, with the paired exact McNemar IMPROVED at p=1.6e-33 — and
 * the policy cut wrong-but-spoken from 276/704 (39%) to 14/323 (4.3%) with ZERO
 * wrong dosages. The shipped regex speaks 173 of 696 values. This module is the
 * missing candidate finder; wiring it in is a separate, reviewed change.
 *
 * PORTING RULES THAT MATTER (each earned porting label_spot/field_proximity):
 *   1. NO REGEX LOOK-BEHIND. A look-behind is a PARSE-TIME SyntaxError on
 *      Safari < 16.4 and kills the whole script (see app/tools/ios_compat_scan.js).
 *      None is used here; label_spot has none and the whole-word boundary is a
 *      consuming `start/end inside a longer token` test, not an assertion.
 *   2. THE VOCABULARY IS GENERATED, NOT TYPED. VOCAB / CONFUSE / SEP below are a
 *      @generated block written by ml/tests/test_field_spot_vectors.py straight
 *      from ml/src/label_spot.py (DOC_LABELS, _CONFUSE) and
 *      ml/src/field_proximity.py (_SEP), and a pytest fails if the block drifts
 *      from the Python. Hand-editing it is a bug.
 *   3. PYTHON WHITESPACE IS NOT JS WHITESPACE. str.strip()/rstrip() use the same
 *      29-character set (\x1c-\x1f and \x85 in, \ufeff out) and JS has the
 *      opposite set. pyRstrip below is that exact set; line-text stripping uses
 *      SightLineFieldGuard.pyStrip, which is the same 29 characters.
 *   4. OFFSETS ARE UTF-16 INDICES. Python indexes by CODE POINT; JS by UTF-16
 *      unit. Identical for the recogniser's alphabet (0-9 A-Z space); differs for
 *      astral input, which is excluded from the vectors (see UNPORTABLE there).
 *
 * PARITY: app/tools/field_spot_vectors.json is generated FROM the Python, and
 * app/tools/field_spot_contract.js asserts this module returns the same spots
 * (label, span, score) and the same extracted values for every vector.
 * Run:  node app/tools/field_spot_contract.js      (exit 0 = parity)
 */

"use strict";

const FieldSpot = (() => {

  const FG = (typeof module !== "undefined" && module.exports)
    ? require("./field_guard.js")
    : (typeof globalThis !== "undefined" ? globalThis.SightLineFieldGuard : undefined);

  /* @generated:vocab:begin */
  // ── VOCABULARY — GENERATED from ml/src/label_spot.py (DOC_LABELS,
  //    _CONFUSE), traced through provenance() to ml/src/field_lexicon.py
  //    PATTERNS + capture.DOCS + evaluate.classify_doc, and
  //    ml/src/field_proximity.py (_SEP). DO NOT HAND-EDIT. Regenerate:
  //      ./.venv/bin/python ml/tests/test_field_spot_vectors.py --write
  const VOCAB = [
    "ACCOUNT HOLDER",
    "ACCOUNT NUMBER",
    "AGREEMENT",
    "ATM WITHDRAWAL",
    "BRITISH GAS",
    "CAPSULES",
    "CASHBACK",
    "CLAUSE",
    "COURT",
    "DATE",
    "DEED",
    "DOB",
    "DOSAGE",
    "HSBC UK BANK PLC",
    "LANDLORD",
    "LEASE",
    "LEASE AGREEMENT",
    "MG",
    "NOTICE",
    "OPENING BALANCE",
    "PATIENT",
    "PERIOD",
    "PHARMACY",
    "PRESCRIBER",
    "PRESCRIPTION",
    "PROPERTY",
    "RENT",
    "RX",
    "SALARY CREDIT",
    "SORT CODE",
    "STATEMENT",
    "TABLETS",
    "TAKE",
    "TENANT",
    "TERM",
    "TESCO",
    "WARNING",
  ];
  const CONFUSE = {"0":"O","1":"I","5":"S","7":"T","6":"G","8":"B","L":"I"};
  const SEP = " \t:#=|.,\u00b7\u2014";
  /* @generated:vocab:end */

  const DEFAULT_THRESHOLD = 0.75;

  // ── Python-compatible primitives ───────────────────────────────────────────
  //
  // The 29 code points Python calls whitespace, for str.rstrip()'s default set.
  // SightLineFieldGuard owns the same string but does not export it, so a
  // trailing-only strip needs its own copy here (field_guard.js is not ours to
  // edit). WS is the class; WS_TRAIL is the trailing run.
  const WS = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
  const WS_TRAIL = new RegExp("[" + WS + "]+$");

  /** Python str.rstrip() with the default (29-code-point) whitespace set. */
  function pyRstrip(s) {
    return String(s).replace(WS_TRAIL, "");
  }

  // Python str.isalnum() is Unicode (L* + N*); JS's \w is ASCII. \p{L}\p{N}
  // matches the same characters, so a token boundary lands where Python's does.
  const ALNUM_U = /[\p{L}\p{N}]/u;

  function isAlnum(ch) {
    return ch !== "" && ALNUM_U.test(ch);
  }

  /** Python str.isdigit() over the recogniser's alphabet (ASCII digits only). */
  const DIGIT_ASCII = /[0-9]/;

  // ── normalisation ──────────────────────────────────────────────────────────
  //
  // OCR's digit-for-letter confusions folded to one canonical class each, plus
  // case and punctuation. Folding is free (no edit budget spent).

  const ALNUM_ASCII = /[0-9A-Z]/;

  /** (folded string, index map into `text`) — punctuation and case dropped. */
  function normOf(text) {
    const up = String(text).toUpperCase();
    let out = "", idx = [];
    for (let i = 0; i < up.length; i++) {
      const ch = up.charAt(i);
      if (!ALNUM_ASCII.test(ch)) continue;
      out += Object.prototype.hasOwnProperty.call(CONFUSE, ch) ? CONFUSE[ch] : ch;
      idx.push(i);
    }
    return [out, idx];
  }

  const NORM_LABELS = {};
  for (const lab of VOCAB) NORM_LABELS[lab] = normOf(lab)[0];

  // ── Levenshtein, abandoned early once it exceeds `cap` (-> cap + 1) ─────────

  function lev(a, b, cap) {
    if (Math.abs(a.length - b.length) > cap) return cap + 1;
    let prev = [];
    for (let j = 0; j <= b.length; j++) prev.push(j);
    for (let i = 1; i <= a.length; i++) {
      const ca = a.charAt(i - 1);
      const cur = [i];
      for (let j = 1; j <= b.length; j++) {
        const cb = b.charAt(j - 1);
        cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca === cb ? 0 : 1)));
      }
      let mn = Infinity;
      for (const x of cur) if (x < mn) mn = x;
      if (mn > cap) return cap + 1;
      prev = cur;
    }
    return prev[prev.length - 1];
  }

  /**
   * round(x, 4) as Python does it: nearest over the EXACT value of the double,
   * ties to even. Number(FG.fmtF(x, 4)) is that: fmtF is CPython's "%.4f" (an
   * exact decimal string, ties to even), and re-reading it gives the same double
   * CPython's round() returns. JS toFixed() would round ties away from zero.
   */
  function round4(x) {
    return Number(FG.fmtF(x, 4));
  }

  // ── spot_labels ────────────────────────────────────────────────────────────

  /** Sort key (-L, start, -score, label), exactly Python's c[:4]. */
  function cmpCand(a, b) {
    if (a[0] !== b[0]) return a[0] < b[0] ? -1 : 1;
    if (a[1] !== b[1]) return a[1] < b[1] ? -1 : 1;
    if (a[2] !== b[2]) return a[2] < b[2] ? -1 : 1;
    if (a[3] !== b[3]) return a[3] < b[3] ? -1 : 1;
    return 0;
  }

  /** Sort key (start, end, label). */
  function cmpSpot(a, b) {
    if (a.start !== b.start) return a.start < b.start ? -1 : 1;
    if (a.end !== b.end) return a.end < b.end ? -1 : 1;
    if (a.label !== b.label) return a.label < b.label ? -1 : 1;
    return 0;
  }

  /**
   * Find known labels in `text`; return spotted regions, longest first.
   *
   * A candidate is a same-length window of the folded text whose edit distance
   * to a folded label is within budget, PROVIDED the window starts and ends on a
   * token boundary — that is what stops "TERM" firing inside "TERMINAL"/"TERMS".
   *
   * The budget is the score gate: accepted iff 1 - dist/len >= threshold, so the
   * default 0.75 means <=1.5 edits per 6 characters and the conservative
   * 1-edit-per-6 rule is threshold=0.833.
   */
  function spotLabels(text, threshold) {
    const th = threshold === undefined || threshold === null ? DEFAULT_THRESHOLD : threshold;
    const s = typeof text === "string" ? text : "";
    const nrm = normOf(s), norm = nrm[0], idx = nrm[1];
    if (!norm) return [];

    const cands = [];
    for (const label of VOCAB) {
      const lab = NORM_LABELS[label];
      const L = lab.length;
      if (L > norm.length) continue;
      const cap = Math.floor((1.0 - th) * L + 1e-9);
      // Window = the folded label's length, or one shorter: OCR DROPS characters
      // far more often than it invents them, and a shorter window is the only
      // way to align that. One LONGER window is deliberately not tried — it is
      // what would let TERM fire inside "TERMS".
      const wls = L > 2 ? [L, L - 1] : [L];
      for (const wl of wls) {
        if (wl > norm.length) continue;
        for (let i = 0; i <= norm.length - wl; i++) {
          const prv = idx[i] - 1;
          if (prv >= 0 && isAlnum(s.charAt(prv))) continue;   // starts inside a token
          const j = i + wl;
          const nxt = idx[j - 1] + 1;
          if (nxt < s.length && isAlnum(s.charAt(nxt))) continue;  // ends inside a token
          const d = lev(lab, norm.slice(i, j), cap);
          if (d > cap) continue;
          const score = 1.0 - d / L;
          if (score < th) continue;
          cands.push([-L, idx[i], -score, label, idx[i], nxt, score]);
        }
      }
    }

    // Longest match first, then earliest start, then best score: two spots never
    // cover the same characters, and the result does not depend on dict order.
    cands.sort(cmpCand);
    const taken = [], kept = [];
    for (const c of cands) {
      const label = c[3], st = c[4], en = c[5], score = c[6];
      let overlap = false;
      for (const t of taken) if (st < t[1] && t[0] < en) { overlap = true; break; }
      if (overlap) continue;
      taken.push([st, en]);
      kept.push({ label: label, start: st, end: en, score: round4(score) });
    }
    kept.sort(cmpSpot);
    return kept;
  }

  /** Spot labels line by line, tagging each spot with its `line_index`. */
  function spotInLines(lines) {
    const out = [];
    const arr = lines || [];
    for (let i = 0; i < arr.length; i++) {
      const line = arr[i] || {};
      const t = typeof line.text === "string" ? line.text : "";
      for (const sp of spotLabels(t)) {
        out.push({ label: sp.label, start: sp.start, end: sp.end, score: sp.score,
                   line_index: i });
      }
    }
    return out;
  }

  // ── proximity read ─────────────────────────────────────────────────────────

  /**
   * First index >= i that is not a leading separator.
   *
   * A '-' is a separator only when it does NOT start a digit: 'Sort Code: 40-11-04'
   * trims the colon and space, while a value written '-1,100.00' keeps its sign.
   */
  function trimIndex(s, i, limit) {
    const n = (limit === undefined || limit === null) ? s.length : Math.min(limit, s.length);
    while (i < n) {
      const c = s.charAt(i);
      if (SEP.indexOf(c) >= 0) {
        i++;
      } else if (c === "-" && !(i + 1 < n && DIGIT_ASCII.test(s.charAt(i + 1)))) {
        i++;
      } else {
        break;
      }
    }
    return i;
  }

  /** Text to the RIGHT of a label span on the same line, leading separators
   *  trimmed ('Sort Code: 40-11-04' + span of 'Sort Code' -> '40-11-04'). */
  function valueAfter(lineText, span) {
    const text = (lineText === null || lineText === undefined) ? "" : String(lineText);
    const end = Math.trunc(Number(span[1]));
    return pyRstrip(text.slice(trimIndex(text, end, undefined)));
  }

  /** Python `ln.get("text", "")`; a line's text is always a string from the OCR
   *  layer, and a missing one reads as "". */
  function textOf(ln) {
    return (ln && typeof ln.text === "string") ? ln.text : "";
  }

  /**
   * The value line just below a label's line, or null.
   *
   * "just below" = smallest box.y0 at or under the label line's y1, tie-broken by
   * x0 then index, so two stacked lines never flip between runs.
   */
  function nextLineText(lines, li, box) {
    if (box !== null && box !== undefined) {
      const y1 = box[3];
      const cands = [];
      for (let j = 0; j < lines.length; j++) {
        if (j === li) continue;
        const ln = lines[j] || {};
        const b = ln.box;
        if (b && b[1] >= y1 && FG.pyStrip(textOf(ln))) cands.push([b[1], b[0], j]);
      }
      if (cands.length) {
        cands.sort((a, b) => a[0] !== b[0] ? (a[0] < b[0] ? -1 : 1)
                            : a[1] !== b[1] ? (a[1] < b[1] ? -1 : 1)
                            : (a[2] < b[2] ? -1 : a[2] > b[2] ? 1 : 0));
        const j = cands[0][2];
        return [j, String(textOf(lines[j]))];
      }
      return null;
    }
    const j = li + 1;
    if (j >= 0 && j < lines.length && FG.pyStrip(textOf(lines[j]))) {
      return [j, String(textOf(lines[j]))];
    }
    return null;
  }

  function intOf(v) {
    if (v === undefined || v === null) return 0;
    const n = Number(v);
    return Number.isFinite(n) ? Math.trunc(n) : 0;
  }

  /** Sort key (line_index, start, end, label). */
  function cmpSpotIn(a, b) {
    if (a.line_index !== b.line_index) return a.line_index < b.line_index ? -1 : 1;
    if (a.start !== b.start) return a.start < b.start ? -1 : 1;
    if (a.end !== b.end) return a.end < b.end ? -1 : 1;
    if (a.label !== b.label) return a.label < b.label ? -1 : 1;
    return 0;
  }

  /** {field_name: {value, ok, kind, reason, provenance}} from label spots. */
  function extractFields(lines, spots) {
    const ls = lines ? Array.from(lines) : [];
    const ordered = (spots || [])
      .filter(s => s && typeof s === "object" && !Array.isArray(s))
      .map(s => ({
        label: s.label === undefined || s.label === null ? "" : String(s.label),
        start: intOf(s.start), end: intOf(s.end), line_index: intOf(s.line_index),
      }))
      .sort(cmpSpotIn);

    const out = {};
    const claimed = new Set();          // (line_index, value_start) already spoken for
    for (let i = 0; i < ordered.length; i++) {
      const sp = ordered[i];
      const li = sp.line_index;
      if (!(li >= 0 && li < ls.length)) continue;
      const rl = FG.resolveLabel(sp.label);
      const name = rl[0], kind = rl[1];
      if (Object.prototype.hasOwnProperty.call(out, name)) continue;   // first wins

      const text = textOf(ls[li]);
      const end = sp.end;
      let nxt = text.length;
      for (let k = i + 1; k < ordered.length; k++) {
        if (ordered[k].line_index === li && ordered[k].start > end) {
          nxt = ordered[k].start;
          break;
        }
      }

      const j = trimIndex(text, end, nxt);
      let value = pyRstrip(text.slice(j, nxt));
      let provenance = "same-line-right";
      let region = li + "|" + j;

      if (!value) {
        const found = nextLineText(ls, li, ls[li] ? ls[li].box : undefined);
        if (found) {
          const nli = found[0], ntext = found[1];
          const k = trimIndex(ntext, 0, undefined);
          value = pyRstrip(ntext.slice(k));
          provenance = "next-line";
          region = nli + "|" + k;
        } else {
          out[name] = { value: "", ok: false, kind: kind,
                        reason: "no value found after label", provenance: "none" };
          continue;
        }
      }

      if (claimed.has(region)) continue;         // one value, one field. no dupes.
      claimed.add(region);

      const vr = FG.validate(kind, value);
      out[name] = { value: value, ok: vr[0], kind: kind, reason: vr[1],
                    provenance: provenance };
    }
    return out;
  }

  return {
    VOCAB, CONFUSE, SEP, DEFAULT_THRESHOLD,
    // spotting
    spotLabels, spotInLines,
    // proximity read
    trimIndex, valueAfter, nextLineText, extractFields,
    // exposed so the contract can prove the porting rules hold
    normOf, lev, pyRstrip,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = FieldSpot;
if (typeof globalThis !== "undefined") globalThis.SightLineFieldSpot = FieldSpot;
