/**
 * SightLine — typed field validation + the speak / refuse decision (JS port).
 *
 * Ports, semantically exactly:
 *   ml/src/field_proximity.py — validate(kind, value) -> (ok, reason), one
 *       validator per kind, every rejection carrying a human-readable REASON
 *       (the reason is what the product says INSTEAD of the value).
 *   ml/src/field_policy.py    — decide(regex_fields, spot_fields, ...) -> the
 *       list of values to SPEAK, each refusal carrying its why.
 *
 * WHY THE APP NEEDS IT: the whole-page regex path (app/js/pipeline.js PATTERNS)
 * is precise but has no idea what shape a value should be, so it can speak a
 * value that is the wrong shape entirely. The policy below drops wrong-but-spoken
 * from 39% to 4.3% on the measured test split, and to zero wrong dosages: a
 * confidently spoken wrong dosage is the failure this product exists to prevent.
 *
 * PARITY: app/tools/field_guard_vectors.json is generated FROM the Python, and
 * app/tools/field_guard_contract.js asserts this module returns the same
 * decision and the same kind for every vector. Run it with plain node:
 *     node app/tools/field_guard_contract.js
 *
 * THREE PORTING RULES THAT MATTER (each earned by a mismatch found by vectors):
 *   1. NO REGEX LOOK-BEHIND. A look-behind is a PARSE-TIME SyntaxError on
 *      Safari < 16.4 and kills the whole script (see app/tools/ios_compat_scan.js).
 *      Python uses one in resolve_label; it is translated below into a consuming
 *      alternative plus a look-AHEAD, which accepts exactly the same positions.
 *   2. PYTHON WHITESPACE IS NOT JS WHITESPACE. str.strip()/str.split()/r"\s" all
 *      use the same 29-character set, which includes \x1c-\x1f and \x85 and
 *      excludes \ufeff — JS has the opposite set on both counts. WS below is the
 *      exact Python set and is used everywhere Python wrote \s or called strip.
 *   3. NUMBERS ARE FORMATTED EXACTLY. canonical() compares money and dosage via
 *      "%.2f" / "%.10g". CPython rounds the EXACT binary value to nearest with
 *      ties-to-even; JS toFixed/toExponential round ties away from zero, so
 *      '%.2f' % 0.125 is '0.12' in Python and would be '0.13' in JS. Both
 *      formatters below are exact (BigInt over the double's mantissa).
 *
 * The app's values arrive as strings; the parts of the Python modules that read
 * geometry or a spot list (extract_fields, value_after) are deliberately NOT
 * ported here — the app has no label provider yet. Wire-in is a separate change.
 */

"use strict";

const FieldGuard = (() => {

  // ── Python-compatible primitives ───────────────────────────────────────────
  //
  // The 29 code points Python calls whitespace: str.isspace(), str.strip()'s
  // default set, re "\s", and the separator set of str.split() are all this.
  const WS = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
  const WS_LEAD = new RegExp("^[" + WS + "]+");
  const WS_TRAIL = new RegExp("[" + WS + "]+$");
  const WS_RUN = new RegExp("[" + WS + "]+");

  /** Python str.strip(). */
  function pyStrip(s) {
    return String(s).replace(WS_LEAD, "").replace(WS_TRAIL, "");
  }

  /** Python str.split() with no argument: whitespace runs, empties dropped. */
  function pySplit(s) {
    const t = pyStrip(s);
    return t === "" ? [] : t.split(WS_RUN);
  }

  /** Python len(): CODE POINTS, not UTF-16 units (matters for the 40-char cap). */
  function pyLen(s) {
    let n = 0;
    for (const _ of s) n++;
    return n;
  }

  /** Python str[:n] slicing, by code point. */
  function pySlice(s, n) {
    let out = "", i = 0;
    for (const c of s) {
      if (i++ >= n) break;
      out += c;
    }
    return out;
  }

  /** Python re.fullmatch: the match must cover the whole string. */
  function fullMatch(re, s) {
    const m = re.exec(s);
    return m && m[0] === s ? m : null;
  }

  /** re.fullmatch over a source string, no anchors needed. */
  function fullMatchSrc(src, flags, s) {
    return fullMatch(new RegExp(src, flags), s);
  }

  // Python str.isdigit()/r"\d" are Unicode-aware (Nd, plus isdigit's No
  // supersets); everything here is ASCII. See UNPORTABLE below.
  const DIGIT = /[0-9]/;
  function hasDigit(s) {
    return DIGIT.test(s);
  }
  function digits(s) {
    return String(s).replace(/[^0-9]/g, "");
  }
  /** Python _alnum(): lowercased first — 'JANE DOE' is 'janedoe', not '' — then
   *  everything outside [a-z0-9] dropped. */
  function alnum(s) {
    return String(s).toLowerCase().replace(/[^a-z0-9]/g, "");
  }
  function firstToken(s) {
    const p = pySplit(s);
    return p.length ? p[0] : "";
  }

  // ── exact decimal formatting ("%.2f" / "%.10g") ────────────────────────────

  const BUF = new DataView(new ArrayBuffer(8));

  /** |x| as mantissa * 2^exp, exactly, for a finite non-zero double. */
  function exactParts(x) {
    BUF.setFloat64(0, x);
    const hi = BUF.getUint32(0), lo = BUF.getUint32(4);
    let e = (hi >>> 20) & 0x7ff;
    let m = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
    if (e === 0) e = 1; else m |= 1n << 52n;
    return { m, e: e - 1075 };
  }

  /**
   * round(ax * 10^f) as a BigInt, exact, ties to EVEN — the rounding CPython's
   * %f and %g do on the exact value of the double. f may be negative.
   */
  function roundScaled(ax, f) {
    const { m, e } = exactParts(ax);
    let num = m, den = 1n;
    if (e >= 0) num <<= BigInt(e); else den <<= BigInt(-e);
    if (f >= 0) num *= 10n ** BigInt(f); else den *= 10n ** BigInt(-f);
    let q = num / den;
    const r = num % den, twice = r * 2n;
    if (twice > den || (twice === den && (q & 1n) === 1n)) q += 1n;   // ties to even
    return q;
  }

  /** C "%.*f" / CPython "%.2f". */
  function fmtF(x, f) {
    if (Number.isNaN(x)) return "nan";
    if (x === Infinity) return "inf";
    if (x === -Infinity) return "-inf";
    const neg = x < 0 || Object.is(x, -0);
    let d = roundScaled(Math.abs(x), f).toString();
    if (f > 0) {
      d = d.padStart(f + 1, "0");
      d = d.slice(0, -f) + "." + d.slice(-f);
    }
    return (neg ? "-" : "") + d;
  }

  /** C "%.*g" / CPython "%.10g": p significant digits, trailing zeros stripped. */
  function fmtG(x, p) {
    if (Number.isNaN(x)) return "nan";
    if (x === Infinity) return "inf";
    if (x === -Infinity) return "-inf";
    if (x === 0) return Object.is(x, -0) ? "-0" : "0";
    const neg = x < 0, ax = Math.abs(x);
    let X = Math.floor(Math.log10(ax));          // decimal exponent, correct to +/-1
    let f = 0, d = "";
    for (let i = 0; i < 4; i++) {
      f = p - 1 - X;
      d = roundScaled(ax, f).toString();
      const X2 = d.length - 1 - f;
      if (X2 === X) break;
      X = X2;
    }
    if (X < -4 || X >= p) {                      // %e branch
      const frac = d.slice(1).replace(/0+$/, "");
      const exp = Math.abs(X).toString().padStart(2, "0");
      return (neg ? "-" : "") + d[0] + (frac ? "." + frac : "") +
             "e" + (X < 0 ? "-" : "+") + exp;
    }
    const intPart = f > 0 ? d.padStart(f + 1, "0").slice(0, -f) : d + "0".repeat(-f);
    const fracPart = f > 0 ? d.padStart(f + 1, "0").slice(-f).replace(/0+$/, "") : "";
    return (neg ? "-" : "") + intPart + (fracPart ? "." + fracPart : "");
  }

  // Python float(): ValueError -> the empty string, and no other way in. The
  // input alphabet is digits, "." and "-" (the caller stripped the rest), so
  // this one regex IS float()'s grammar over that alphabet. Huge digit strings
  // become Infinity exactly as Python's float() does.
  const FLOAT_SRC = /^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)$/;

  function pyNum(x, fmt) {
    const s = String(x).replace(/,/g, "");
    if (!FLOAT_SRC.test(s)) return "";
    const v = Number(s);
    return fmt === "%.2f" ? fmtF(v, 2) : fmtG(v, 10);
  }

  // ── the kinds ──────────────────────────────────────────────────────────────

  const UNITS = new Set(["mg", "mcg", "g", "ml", "tablet", "tablets", "capsule",
                         "capsules", "drop", "drops", "unit", "units"]);

  /** Dosing frequency phrases: the product's Frequency pattern + UK abbrevs. */
  const FREQ_SRC =
    "(?:once|twice|three|four|five|six|[0-9]+)[" + WS + "]*times?[" + WS + "]*" +
    "(?:a[" + WS + "]*|per[" + WS + "]*)?(?:day|daily|week|weekly)" +
    "|every[" + WS + "]*[0-9]+(?:[" + WS + "]*(?:to|-)" + "[" + WS + "]*[0-9]+)?" +
    "[" + WS + "]*(?:hours?|hrs?|h)(?![\\p{L}\\p{N}_])" +
    // Python's \b is Unicode-word; JS's \b is ASCII-word, so the boundary is
    // spelled out instead: (?:^|[^\w]) consumes the char before, and the
    // look-ahead asserts the char after. Same set of matching positions.
    "|(?:^|[^\\p{L}\\p{N}_])(?:bd|b\\.d\\.|tds|t\\.d\\.s\\.|qds|q\\.d\\.s\\.|" +
    "prn|p\\.r\\.n\\.|nocte|mane|od|qid|tid|bid)(?![\\p{L}\\p{N}_])" +
    "|at[" + WS + "]+bedtime|as[" + WS + "]+(?:required|needed)|" +
    "in[" + WS + "]+the[" + WS + "]+morning";
  const FREQ_RE = new RegExp(FREQ_SRC, "iu");

  const CURRENCY_SRC = "(?:[\u00a3$\u20ac]|GBP|USD|EUR)";
  const CUR_LEAD = new RegExp("^[" + WS + "]*" + CURRENCY_SRC + "[" + WS + "]*", "i");
  const CUR_TRAIL = new RegExp("[" + WS + "]*" + CURRENCY_SRC + "[" + WS + "]*$", "i");
  const MONEY_RE = new RegExp("([+-]?)[" + WS + "]*([0-9,]+(?:\\.[0-9]+)?)");
  const THOUSANDS_OK = /^[0-9]{1,3}(?:,[0-9]{3})+$/;
  const DOSE_RE = new RegExp("([0-9]+(?:\\.[0-9]+)?)[" + WS + "]*([A-Za-z]+)", "gi");
  const DMY_RE = /^([0-9]{1,2})[/-]([0-9]{1,2})[/-]([0-9]{2}|[0-9]{4})$/;
  const YMD_RE = /^([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})$/;
  const H8 = /^[0-9]{8}$/;

  // Long-form month-name dates: '15 January 2024', '3 Mar 2026', '15 Jan 24',
  // '15th January, 2024'. ASCII month names only (no locale handling). The day
  // may carry an English ordinal suffix because pipeline.js's own legal Date
  // pattern emits '\d{1,2}(?:st|nd|rd|th)?'. Full and abbreviated month names,
  // an optional comma before the year, 2- or 4-digit year, case-insensitive.
  const MONTHS = {
    january: 1, february: 2, march: 3, april: 4, may: 5, june: 6, july: 7,
    august: 8, september: 9, october: 10, november: 11, december: 12,
    jan: 1, feb: 2, mar: 3, apr: 4, jun: 6, jul: 7, aug: 8,
    sep: 9, sept: 9, oct: 10, nov: 11, dec: 12,
  };
  const DMONTH_RE = new RegExp(
    "^([0-9]{1,2})(?:st|nd|rd|th)?[" + WS + "]+([A-Za-z]{3,9})\\.?[" + WS +
    "]*,?[" + WS + "]*([0-9]{2}|[0-9]{4})$", "i");

  const ID_BAD = /[^A-Za-z0-9-]/;
  const ID_ALNUM = /[A-Za-z0-9]/;
  const SUBSTANCE_ONLY = new RegExp("^[" + WS + "0-9:.,/-]+$");

  // ── validators: (ok, reason), one per kind ─────────────────────────────────

  /**
   * Exactly 6 digits: '40-11-04', '40 11 04', '401104'. Takes the LEADING run
   * of digits/spaces/hyphens, not the first token: for a space-separated sort
   * code the space IS the separator, so '40 11 04' would otherwise be just '40'.
   */
  function validateSortCode(v) {
    const s = pyStrip(v);
    let cut = s.length;
    for (let i = 0; i < s.length; i++) {
      const c = s[i];
      if (!DIGIT.test(c) && c !== " " && c !== "-") { cut = i; break; }
    }
    const cleaned = s.slice(0, cut).replace(/^[ -]+|[ -]+$/g, "")
      .replace(new RegExp("[-" + WS + "]", "g"), "");
    if (/^[0-9]{6}$/.test(cleaned)) return [true, "ok"];
    return [false, "6 digits required, got " + cleaned.replace(/[^0-9]/g, "").length];
  }

  /** 4-12 digits. A value masked to its last 4 digits is ACCEPTED and flagged. */
  function validateAccount(v) {
    const s = pyStrip(v);
    const tokens = pySplit(s);
    const cand = tokens.find(hasDigit) || "";
    const d = digits(cand);
    const masked = d.length === 4 || /[*xX#]/.test(s) || s.toLowerCase().indexOf("ending") >= 0;
    if (!(d.length >= 4 && d.length <= 12)) {
      return [false, "4-12 digits required, got " + d.length];
    }
    if (masked) return [true, "masked, last 4 digits only"];
    return [true, "ok"];
  }

  function isRealDate(y, mo, d) {
    if (!(y >= 1 && y <= 9999)) return false;
    if (!(mo >= 1 && mo <= 12)) return false;
    if (d < 1) return false;
    const leap = y % 4 === 0 && (y % 100 !== 0 || y % 400 === 0);
    const per = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    return d <= per[mo - 1];
  }

  /** A real calendar date in DD/MM/YYYY, DD/MM/YY, YYYY-MM-DD, DD-MM-YYYY or a
   *  long-form month-name date ('15 January 2024', '3 Mar 2026', '15 Jan 24'). */
  function validateDate(v) {
    const s = pyStrip(v);
    const tok = firstToken(s);
    let d, mo, y, m = fullMatch(DMY_RE, tok);
    if (m) {
      d = Number(m[1]); mo = Number(m[2]); y = Number(m[3]);
    } else {
      m = fullMatch(YMD_RE, tok);
      if (m) {
        y = Number(m[1]); mo = Number(m[2]); d = Number(m[3]);
      } else {
        // Long-form month-name dates: the whole value, not the first token,
        // because the value is multi-token. The app's own Date pattern reads
        // these, so refusing them would refuse a value it reads correctly.
        m = fullMatch(DMONTH_RE, s);
        if (m) {
          mo = Object.prototype.hasOwnProperty.call(MONTHS, m[2].toLowerCase())
            ? MONTHS[m[2].toLowerCase()] : undefined;
          if (mo === undefined) return [false, "not a calendar date"];   // '15 Foo 2024'
          d = Number(m[1]); y = Number(m[3]);
        } else if (!H8.test(tok)) {
          // SEPARATOR-FREE DATES, and this is not a nicety: the recogniser's
          // alphabet is 0-9 A-Z and space — it CANNOT emit '/' or '-'. Every
          // date this product reads arrives as 8 digits (04021979 for
          // 04/02/1979), so refusing them refuses every date the model can
          // physically produce.
          return [false, "not a calendar date"];
        } else {
          d = Number(tok.slice(0, 2)); mo = Number(tok.slice(2, 4)); y = Number(tok.slice(4));
          if (d > 31 || mo > 12) {                     // DDMMYYYY first (HK/UK),
            d = Number(tok.slice(6));                  // then YYYYMMDD
            mo = Number(tok.slice(4, 6));
            y = Number(tok.slice(0, 4));
          }
        }
      }
    }
    if (y < 100) y += 2000;
    if (!isRealDate(y, mo, d)) return [false, "not a calendar date"];
    return [true, "ok"];
  }

  /** Optional currency either side, thousands separators, optional sign, and
   *  exactly two decimals when a decimal point is present. */
  function validateMoney(v) {
    let s = pyStrip(v), prev = null;
    while (s !== prev) {
      prev = s;
      s = s.replace(CUR_LEAD, "").replace(CUR_TRAIL, "");
    }
    const m = fullMatch(MONEY_RE, s);
    if (!m) return [false, "not a monetary amount: " + pyStrip(v)];
    const num = m[2];
    if (num.indexOf(".") >= 0) {
      const dec = num.split(".")[1];
      if (dec.length !== 2) {
        return [false, "money needs two decimals, got " + dec.length];
      }
    }
    if (num.indexOf(",") >= 0 && !THOUSANDS_OK.test(num.split(".")[0])) {
      return [false, "bad thousands grouping: " + pyStrip(v)];
    }
    return [true, "ok"];
  }

  /** A number followed by a KNOWN unit, optionally a frequency. */
  function validateDosage(v) {
    const m = DOSE_RE.exec(String(v));
    DOSE_RE.lastIndex = 0;                    // findall semantics only; keep it clean
    if (!m) return [false, "no dosage in value: " + pyStrip(v)];
    const unit = m[2].toLowerCase();
    if (!UNITS.has(unit)) return [false, "unknown unit: " + m[2]];
    const rest = String(v).slice(m.index + m[0].length).replace(/^[ .,;]+|[ .,;]+$/g, "");
    if (rest && !UNITS.has(rest.toLowerCase()) && !FREQ_RE.test(rest)) {
      return [false, "unknown frequency: " + rest];
    }
    return [true, "ok"];
  }

  /** Case references like 2024-CV-00456: letters, digits, hyphens, 4-20 chars. */
  function validateIdentifier(v) {
    const tok = firstToken(pyStrip(v));
    const bad = ID_BAD.exec(tok);
    if (bad) return [false, "invalid character in identifier: " + bad[0]];
    if (!(tok.length >= 4 && tok.length <= 20)) {
      return [false, "identifier must be 4-20 chars, got " + tok.length];
    }
    if (!ID_ALNUM.test(tok)) return [false, "identifier needs a letter or digit"];
    return [true, "ok"];
  }

  /** Any non-empty string under 40 chars with at least one alphanumeric. */
  function validateGeneral(v) {
    const s = pyStrip(v);
    if (!s) return [false, "empty value"];
    if (pyLen(s) >= 40) return [false, "too long: " + pyLen(s) + " chars"];
    if (!ID_ALNUM.test(s)) return [false, "no alphanumeric"];
    return [true, "ok"];
  }

  const VALIDATORS = {
    sort_code: validateSortCode,
    account: validateAccount,
    date: validateDate,
    money: validateMoney,
    dosage: validateDosage,
    identifier: validateIdentifier,
    general: validateGeneral,
  };

  /** (ok, reason) for `value` under `kind`. Never throws on bad input. */
  function validate(kind, value) {
    const fn = VALIDATORS[String(kind)];
    if (!fn) return [false, "unknown kind: " + String(kind)];
    if (value === null || value === undefined) return [false, "empty value"];
    return fn(String(value));
  }

  // ── label -> (field name, kind) ────────────────────────────────────────────
  //
  // Ordered most-specific-first; the first needle matching as a whole word wins.
  // Derived from pipeline.js PATTERNS and capture.py's own document labels.

  const FIELD_LABELS = [
    ["sort code", "sort_code", "sort_code"],
    ["account number", "account_number", "account"],
    ["account no", "account_number", "account"],
    ["card ending", "card_ending", "account"],
    ["ending in", "card_ending", "account"],
    ["ending", "card_ending", "account"],
    ["account holder", "account_holder", "general"],
    ["opening balance", "opening_balance", "money"],
    ["closing balance", "closing_balance", "money"],
    ["available funds", "available_funds", "money"],
    ["balance", "balance", "money"],
    ["amount", "amount", "money"],
    ["date of birth", "date_of_birth", "date"],
    ["dob", "date_of_birth", "date"],
    ["date", "date", "date"],
    ["case reference", "case_number", "identifier"],
    ["case number", "case_number", "identifier"],
    ["case ref", "case_number", "identifier"],
    ["case no", "case_number", "identifier"],
    ["matter", "case_number", "identifier"],
    ["iban", "iban", "general"],
    ["dosage", "dosage", "dosage"],
    ["dose", "dosage", "dosage"],
    ["prescription for", "medication", "general"],
    ["medication", "medication", "general"],
    ["rx", "medication", "general"],
    ["frequency", "frequency", "general"],
    ["patient", "patient", "general"],
    ["name", "patient", "general"],
    ["warning", "warning", "general"],
    ["caution", "warning", "general"],
    ["clause", "clause", "general"],
    ["section", "clause", "general"],
    ["phone", "phone", "general"],
    ["email", "email", "general"],
    ["time", "time", "general"],
    ["period", "period", "general"],
    ["property", "property", "general"],
    ["term", "term", "general"],
  ];

  // Python's whole-word test for a needle is "not preceded by a-z" / "not
  // followed by a-z". The trailing half is a look-AHEAD, kept as is. The leading
  // half is spelled (?:^|[^a-z]): either the needle starts the string, or the
  // character before it is not a lowercase letter. That accepts exactly the same
  // positions, and it avoids the construct this repo has been burned by before —
  // the backwards one is a PARSE-TIME SyntaxError on Safari below 16.4 and takes
  // the entire script with it (see app/tools/ios_compat_scan.js, which scans
  // this file).
  const LABEL_RES = FIELD_LABELS.map(([needle, name, kind]) => [
    new RegExp("(?:^|[^a-z])" + needle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "(?![a-z])"),
    name,
    kind,
  ]);

  /** (field name, kind) for a spotted label; unknown labels read as general. */
  function resolveLabel(label) {
    const low = pyStrip(label === null || label === undefined ? "" : String(label)).toLowerCase();
    for (const rx of LABEL_RES) if (rx[0].test(low)) return [rx[1], rx[2]];
    const slug = low.replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
    return [slug || "field", "general"];
  }

  // ── the decision layer ─────────────────────────────────────────────────────

  const CRITICAL_KINDS = new Set(["dosage", "money", "account", "sort_code", "date"]);
  const MIN_SPEAK_CHARS = 4, MIN_SPEAK_WORD = 5;

  // evaluate.fields_found returns bare tokens from evaluate.FIELDS; a token
  // means nothing without its field and kind, so the mapping lives here.
  const REGEX_TOKENS = {
    prescription: { "500mg": ["dosage", "dosage"], "400mg": ["dosage", "dosage"] },
    banking: { "40218877": ["account_number", "account"], "40-11-04": ["sort_code", "sort_code"] },
    legal: { "2500": ["amount", "money"], "2024-CV-00456": ["case_number", "identifier"] },
  };
  const REGEX_KIND = {};
  for (const per of Object.values(REGEX_TOKENS)) {
    for (const entry of Object.values(per)) REGEX_KIND[entry[0]] = entry[1];
  }

  // label_spot.DOC_LABELS — the field names a document type's OWN labels
  // resolve to. The vocabulary gate (R5) needs it: on a prescription the
  // spotter fires CLAUSE inside "CAUSE", and that is not a prescription field.
  const DOC_LABELS = {
    prescription: ["PRESCRIPTION", "PATIENT", "DOB", "DATE", "PRESCRIBER", "PHARMACY",
                   "RX", "TAKE", "WARNING", "DOSAGE", "MG", "TABLETS", "CAPSULES"],
    banking: ["HSBC UK BANK PLC", "STATEMENT", "ACCOUNT HOLDER", "ACCOUNT NUMBER",
              "SORT CODE", "PERIOD", "OPENING BALANCE", "SALARY CREDIT", "RENT",
              "TESCO", "BRITISH GAS", "ATM WITHDRAWAL", "CASHBACK"],
    legal: ["LEASE AGREEMENT", "LEASE", "AGREEMENT", "CLAUSE", "TENANT",
            "LANDLORD", "PROPERTY", "TERM", "NOTICE", "COURT", "DEED"],
  };

  // ── normalisation: what "the same value" means per kind ────────────────────

  function digitsCommas(v) {
    return String(v).replace(/[^0-9.\-]/g, "");
  }

  /**
   * A sorted array of comparable tokens for `value` under `kind`, or [].
   * Two paths AGREE when their sets intersect. Dosage is set-valued; every
   * other kind is scalar. A form this cannot normalise yields [] — it can then
   * neither agree NOR contradict, so the error direction is silence.
   */
  function canonical(kind, value) {
    if (value === null || value === undefined) return [];
    const v = pyStrip(String(value));
    if (!v) return [];
    if (kind === "dosage") {
      const out = new Set();
      for (const m of v.matchAll(DOSE_RE)) {
        const unit = m[2].toLowerCase();
        if (UNITS.has(unit)) out.add(pyNum(m[1], "%.10g") + unit);
      }
      return [...out].sort();
    }
    if (kind === "date") {
      let d = digits(v.split(WS_RUN)[0]);
      if (d.length === 6) d = d.slice(0, 4) + "20" + d.slice(4);   // DDMMYY -> DDMM20YY
      return d.length === 8 ? [d] : [];
    }
    if (kind === "money") {
      const n = pyNum(digitsCommas(v), "%.2f");
      return n ? [n] : [];
    }
    if (kind === "account" || kind === "sort_code") {
      const d = digits(v);                       // '40 11 04' and '40-11-04' agree
      return d ? [d] : [];
    }
    const a = alnum(v);
    return a ? [a] : [];
  }

  // ── candidate construction ─────────────────────────────────────────────────

  /** A regex field may carry one value or several (a page has two doses). */
  function asValues(raw) {
    if (Array.isArray(raw)) return raw.map(String);
    return raw === null || raw === undefined ? [] : [String(raw)];
  }

  function kindOf(field, record, kinds) {
    if (kinds && Object.prototype.hasOwnProperty.call(kinds, field)) return String(kinds[field]);
    if (record && record.kind) return String(record.kind);
    const k = REGEX_KIND[field];
    return k === undefined ? "general" : k;
  }

  function candidates(regexFields, spotFields, kinds) {
    const out = [];
    for (const field0 of Object.keys(regexFields || {})) {
      const field = String(field0);
      for (const v of asValues(regexFields[field0])) {
        out.push({ field, value: pyStrip(v), kind: kindOf(field, null, kinds),
                   source: "regex", ok: null, reason: "" });
      }
    }
    for (const field0 of Object.keys(spotFields || {})) {
      const field = String(field0);
      const raw = spotFields[field0];
      const rec = raw && typeof raw === "object" && !Array.isArray(raw) ? raw : { value: raw };
      const val = rec.value === undefined || rec.value === null ? "" : String(rec.value);
      out.push({ field, value: pyStrip(val),
                 kind: kindOf(field, rec, kinds), source: "spot",
                 ok: rec.ok === undefined ? null : rec.ok,
                 reason: String(rec.reason || "") });
    }
    for (const c of out) {
      if (c.ok === null) {                    // the regex path arrives unvalidated
        const r = validate(c.kind, c.value);
        c.ok = r[0];
        c.reason = r[1];
      }
      c.canon = canonical(c.kind, c.value);
      c.agreed = false;
    }
    return out;
  }

  /** The field names a document type's OWN printed labels resolve to. */
  function vocabulary(docType) {
    const labels = DOC_LABELS[String(docType === null || docType === undefined ? "" : docType)] || [];
    const out = new Set();
    for (const lab of labels) out.add(resolveLabel(lab)[0]);
    return out;
  }

  function intersects(a, b) {
    for (const x of a) if (b.indexOf(x) >= 0) return true;
    return false;
  }

  /**
   * Enough there to be worth speaking: a real word, or a number we can read.
   * 'MAY' / 'MANY' / 'BMAY' are label + 4-character fragments left when a
   * false-positive spot cut the line short — 74 of the 276 wrong values the old
   * path spoke. 'JANE DOE' (two tokens, no long word) must survive.
   */
  function substantial(value) {
    const v = pyStrip(String(value));
    if (alnum(v).length < MIN_SPEAK_CHARS) return false;
    const words = v.match(/[A-Za-z]{2,}/g) || [];
    if (words.length >= 2 || words.some(w => w.length >= MIN_SPEAK_WORD)) return true;
    return SUBSTANCE_ONLY.test(v) && digits(v).length >= MIN_SPEAK_CHARS;
  }

  /**
   * What to SPEAK from both extractions. The rule set, first that fires wins:
   *   R1 empty · R2 validation · R3 disagreement (both refused, no winner)
   *   R4 safety-critical: agreement, else validated + uncontradicted
   *   R5 non-critical: vocabulary + substance gates
   *   R6 deterministic, deduplicated by (field, canonical value)
   */
  function decide(regexFields, spotFields, opts) {
    const o = opts || {};
    const kinds = o.kinds || null;
    const docType = o.doc_type === undefined ? null : o.doc_type;
    const strictCritical = !!o.strict_critical;
    const speakFreeText = o.speak_free_text === undefined ? true : !!o.speak_free_text;

    const cands = candidates(regexFields, spotFields, kinds);
    const vocab = docType ? vocabulary(docType) : null;

    // R3 — disagreement, per field, both paths present and no shared value.
    const byField = new Map();
    for (const c of cands) {
      if (!byField.has(c.field)) byField.set(c.field, []);
      byField.get(c.field).push(c);
    }
    const conflicts = new Map();
    for (const [field, group] of byField) {
      const rs = group.filter(c => c.source === "regex" && c.canon.length);
      const ss = group.filter(c => c.source === "spot" && c.canon.length);
      if (rs.length && ss.length && !rs.some(a => ss.some(b => intersects(a.canon, b.canon)))) {
        conflicts.set(field, [rs[0].value, ss[0].value]);
      }
    }

    // Agreement — the same canonical value from the other path. A candidate of a
    // CRITICAL kind asks the question the other way round too: "does the other
    // path's value CARRY my token?" — that is how the regex's bare '500mg'
    // corroborates the spot path's 'AMOXICILLIN 500MG' read as free text.
    for (const c of cands) {
      if (!c.canon.length) continue;
      for (const other of cands) {
        if (other === c || other.source === c.source) continue;
        if (intersects(other.canon, c.canon)) c.agreed = true;
        else if (CRITICAL_KINDS.has(c.kind) &&
                 intersects(canonical(c.kind, other.value), c.canon)) c.agreed = true;
        if (c.agreed) break;
      }
    }

    const decisions = [];
    cands.forEach((c, i) => {
      const field = c.field, value = c.value, kind = c.kind;
      let decision, reason = "";
      if (!value) {
        decision = "refuse"; reason = "label spotted but no value present";
      } else if (!c.ok) {
        decision = "refuse"; reason = c.reason || "failed validation";
      } else if (conflicts.has(field)) {
        decision = "refuse";
        reason = "two paths disagree: " + conflicts.get(field)[0] + " vs " + conflicts.get(field)[1];
      } else if (CRITICAL_KINDS.has(kind)) {
        if (c.agreed) {
          decision = "speak"; reason = "both paths agree on this value";
        } else if (strictCritical) {
          decision = "refuse";
          reason = "safety-critical " + kind + " from one path only; agreement required";
        } else {
          decision = "speak";
          reason = "validated " + kind + "; the other path had no value to confirm or contradict it";
        }
      } else if (!speakFreeText) {
        decision = "refuse";
        reason = "free text (kind " + kind + ") is not spoken by this policy";
      } else if (vocab !== null && !vocab.has(field)) {
        decision = "refuse";
        reason = "'" + field + "' is not in the " + docType + " label vocabulary " +
                 "(the spot matched another type's label)";
      } else if (!substantial(value)) {
        decision = "refuse"; reason = "value too short to speak: '" + pySlice(value, 20) + "'";
      } else if (c.agreed) {
        decision = "speak"; reason = "both paths agree on this value";
      } else {
        decision = "speak"; reason = "validated " + kind + " text from the label path";
      }
      const part = c.canon.length ? c.canon.join("|") : value.toLowerCase();
      decisions.push({ field, value, decision, why: reason,
                       source: c.agreed ? "both" : c.source, _i: i, _part: part });
    });

    // R6 — deterministic, deduplicated, never the same field+value twice.
    decisions.sort((a, b) =>
      (a.field < b.field ? -1 : a.field > b.field ? 1 : 0) ||
      (a._part < b._part ? -1 : a._part > b._part ? 1 : 0) ||
      (a.decision < b.decision ? -1 : a.decision > b.decision ? 1 : 0) ||
      a._i - b._i);
    const seen = new Set(), out = [];
    for (const d of decisions) {
      const key = d.field + "\u0000" + d._part;   // never the same field+value twice
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ field: d.field, value: d.value, decision: d.decision,
                 why: d.why, source: d.source });
    }
    return out;
  }

  /** The values a caller would read out, in decision order. */
  function spoken(decisions) {
    return decisions.filter(d => d.decision === "speak").map(d => d.value);
  }

  /** {field: value or [values]} from evaluate.fields_found's bare tokens. */
  function fromRegexTokens(docType, tokens) {
    const table = REGEX_TOKENS[String(docType === null || docType === undefined ? "" : docType)] || {};
    const out = {};
    for (const t0 of tokens || []) {
      const t = String(t0);
      const entry = table[t];
      const field = entry ? entry[0] : t.toLowerCase();
      const cur = out[field];
      if (cur === undefined) out[field] = t;
      else if (Array.isArray(cur)) { if (cur.indexOf(t) < 0) cur.push(t); }
      else if (cur !== t) out[field] = [cur, t];
    }
    return out;
  }

  return {
    // primitives (exposed so the contract can prove the porting rules hold)
    pyStrip, pySplit, pyLen, pySlice, fmtF, fmtG, pyNum,
    // validators
    UNITS, CRITICAL_KINDS, MIN_SPEAK_CHARS, MIN_SPEAK_WORD,
    validate, validateSortCode, validateAccount, validateDate, validateMoney,
    validateDosage, validateIdentifier, validateGeneral,
    // labels + vocabulary
    resolveLabel, FIELD_LABELS, DOC_LABELS, REGEX_TOKENS,
    // policy
    canonical, decide, spoken, fromRegexTokens,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = FieldGuard;
if (typeof globalThis !== "undefined") globalThis.SightLineFieldGuard = FieldGuard;