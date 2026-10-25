// js/fields.js — BRIEF.md §6.4. Field extraction and the speak-or-omit gate.
//
// Everything in this file is a pure function over a string. No DOM, no fetch, no
// clock except where a reference date is injected. That is what makes gate G2
// (BRIEF.md §8) able to assert on it in plain Node.
//
// THE PRODUCT RULE THIS FILE EXISTS TO ENFORCE (BRIEF.md §1):
//   "Never speak a number you are not sure about."
// A field is spoken only if `confidence >= CONFIG.confidenceMin` AND it validates.
// Everything else is *omitted* and reported as unclear. `buildSpokenSummary()`
// can physically not emit a number that failed the gate — it reads the gated list,
// not the raw candidates — which is what the G2 unit test asserts.

import { CONFIG } from './config.mjs';
import {
  DOC_TYPES,
  UNRECOGNISED_TYPE,
  CLASSIFIABLE_TYPES,
  GROUPS_BY_TYPE,
  FIELDS,
  groupsForType,
} from './taxonomy.mjs';

// Re-exported so existing importers (tests, tools) keep working. The value itself
// now lives in js/taxonomy.js — ONE definition, so the classifier, the extractor
// and the results screen cannot disagree about what a document type is.
export { DOC_TYPES, UNRECOGNISED_TYPE, CLASSIFIABLE_TYPES, GROUPS_BY_TYPE, FIELDS, groupsForType };

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

/** Alphanumeric character count — the §6.3 "better result" metric, shared. */
export function countAlnum(text) {
  if (!text) return 0;
  const m = String(text).match(/[0-9A-Za-z]/g);
  return m ? m.length : 0;
}

/** Collapse runs of whitespace; keeps newlines meaningful for line-based rules. */
export function squeezeWhitespace(text) {
  return String(text || '').replace(/[ \t\u00a0]+/g, ' ').trim();
}

/**
 * Line-scoped context around a match.
 *
 * WHY THIS EXISTS (found by running the extractors over realistic fixtures):
 * a fixed character window crosses line boundaries, so on a prescription the
 * duration "for 7 days" on the line *below* "Bisoprolol 2.5mg once daily" was
 * being attached to that dose, and on a bank statement every amount inherited
 * "amount due" from a neighbouring row. Both produced confidently wrong fields.
 * Context must come from the same line as the value.
 *
 * @returns {{before:string, after:string, line:string}}
 */
export function lineContext(text, start, end, beforeChars, afterChars) {
  const b = beforeChars === undefined ? 60 : beforeChars;
  const a = afterChars === undefined ? 90 : afterChars;
  let lineStart = text.lastIndexOf('\n', start - 1);
  lineStart = lineStart === -1 ? 0 : lineStart + 1;
  let lineEnd = text.indexOf('\n', end);
  if (lineEnd === -1) lineEnd = text.length;
  return {
    before: squeezeWhitespace(text.slice(Math.max(lineStart, start - b), start)),
    after: squeezeWhitespace(text.slice(end, Math.min(lineEnd, end + a))),
    line: squeezeWhitespace(text.slice(lineStart, lineEnd)),
  };
}

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const round4 = (v) => Math.round(v * 10000) / 10000;

/** ISO calendar-day validity. */
export function isRealDate(year, month, day) {
  if (!(year >= 1900 && year <= 2100)) return false;
  if (!(month >= 1 && month <= 12)) return false;
  if (!(day >= 1 && day <= 31)) return false;
  const days = [31, (isLeapYear(year) ? 29 : 28), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  return day <= days[month - 1];
}

export function isLeapYear(y) {
  return (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;
}

export function toIsoDate(year, month, day) {
  const p = (n) => String(n).padStart(2, '0');
  return `${year}-${p(month)}-${p(day)}`;
}

// ---------------------------------------------------------------------------
// MONEY — §6.4 "currency symbol/code + amount, thousands separators, negatives,
// 'balance'/'total'/'amount due' context words"
// ---------------------------------------------------------------------------

const CURRENCY_SYMBOLS = {
  $: 'USD', '£': 'GBP', '€': 'EUR', '¥': 'JPY', '₹': 'INR', '₩': 'KRW', '₽': 'RUB',
};

const CURRENCY_CODES = ['USD', 'GBP', 'EUR', 'AUD', 'CAD', 'NZD', 'CHF', 'JPY', 'SGD', 'HKD', 'INR', 'ZAR', 'SEK', 'NOK', 'DKK', 'PLN', 'MXN', 'BRL', 'CNY', 'KWD', 'AED', 'ILS', 'TRY'];

// Context words that make a bare number a *money* number, and how much they
// raise confidence.
//
// THE ORDERING IS THE PRODUCT DECISION (§6.5 "the single most important field").
// On a document that states an obligation and a balance, the number the reader
// acts on is the obligation — what is owed — not a balance that merely describes
// the account. So "amount due" / "minimum payment" carry the strongest weight,
// deliberately ABOVE a balance. On a bank statement the summary therefore reads
// the amount due; on a statement with no obligation (a savings statement) the
// balance is still the only money signal and still wins.
const MONEY_CONTEXT = [
  { re: /\bamount\s+due\b/i, weight: 0.3, label: 'amount due' },
  { re: /\bminimum\s+payment\b/i, weight: 0.3, label: 'minimum payment' },
  { re: /\b(?:total|balance|subtotal|sub-total)\b/i, weight: 0.18, label: 'total/balance' },
  { re: /\b(?:new\s+balance|closing\s+balance|opening\s+balance|account\s+balance)\b/i, weight: 0.22, label: 'balance' },
  { re: /\b(?:payment|paid|debit|credit|charge|fee|invoice|receipt|refund|deposit|withdrawal|tax|vat|gst)\b/i, weight: 0.12, label: 'money word' },
  { re: /\b(?:due|owing|outstanding|overdue)\b/i, weight: 0.12, label: 'due/owing' },
];

/**
 * Parse the digits of a money token, honouring both `1,234.56` and `1.234,56`.
 *
 * The ambiguous case is a single separator with three digits after it
 * ("1,234"): that is a thousands separator in en-GB/US, a decimal in some
 * locales, and a typo in others. We take thousands (the overwhelmingly common
 * reading on a document) but charge a confidence penalty, so an ambiguous
 * amount has to be corroborated by context before it is ever spoken.
 *
 * @returns {{value:number|null, amount:string|null, decimalSep:'.'|','|null,
 *            thousandsSep:string|null, ambiguous:boolean, reason:string}}
 */
export function parseAmountToken(token) {
  const raw = String(token || '').trim();
  if (!raw) return { value: null, amount: null, decimalSep: null, thousandsSep: null, ambiguous: false, reason: 'empty' };
  const cleaned = raw.replace(/[^\d.,'\s-]/g, '');
  const digitsOnly = cleaned.replace(/[^\d]/g, '');
  if (!digitsOnly) return { value: null, amount: null, decimalSep: null, thousandsSep: null, ambiguous: false, reason: 'no digits' };

  const hasDot = cleaned.includes('.');
  const hasComma = cleaned.includes(',');
  const hasSpace = /\s/.test(cleaned);

  let decimalSep = null;
  let thousandsSep = null;
  let ambiguous = false;
  let reason = '';

  if (hasDot && hasComma) {
    // The rightmost separator is the decimal one. This is locale-independent and
    // is the rule that makes "1,234.56" and "1.234,56" both work.
    decimalSep = cleaned.lastIndexOf('.') > cleaned.lastIndexOf(',') ? '.' : ',';
    thousandsSep = decimalSep === '.' ? ',' : '.';
  } else if (hasDot || hasComma) {
    const sep = hasDot ? '.' : ',';
    const parts = cleaned.split(sep);
    const tail = parts[parts.length - 1];
    if (parts.length > 2) {
      thousandsSep = sep;             // 1.234.567
    } else if (tail.length === 3) {
      thousandsSep = sep;             // 1,234 -> thousand (ambiguous with 1.234 decimal)
      ambiguous = true;
      reason = 'single separator followed by three digits could be a decimal comma';
    } else if (tail.length === 2) {
      decimalSep = sep;               // 12,34 / 12.34
    } else if (tail.length === 1) {
      decimalSep = sep;               // 12.5
      ambiguous = true;
      reason = 'single digit after the separator';
    } else {
      thousandsSep = sep;
    }
  } else if (hasSpace) {
    thousandsSep = ' ';
  }

  // Normalise to a plain "1234.56" before Number() sees it. Number('1,234.56')
  // is NaN, and Number('1 234') is NaN too — this is the step that usually gets
  // skipped and silently turns every amount into null.
  let normalised = cleaned.replace(/[-\s']/g, '');
  if (thousandsSep) normalised = normalised.split(thousandsSep).join('');
  if (decimalSep && decimalSep !== '.') normalised = normalised.split(decimalSep).join('.');
  if (normalised.startsWith('.')) normalised = `0${normalised}`;

  const value = Number(normalised);
  if (!isFinite(value)) {
    return { value: null, amount: null, decimalSep, thousandsSep, ambiguous, reason: `unparseable amount "${raw}"` };
  }
  const decimals = decimalSep ? (normalised.split('.')[1] || '').length : 0;
  return {
    value,
    amount: value.toFixed(decimals || 0),
    decimalSep,
    thousandsSep,
    ambiguous,
    reason: reason || 'parsed',
  };
}

/**
 * Find money in a document.
 * @returns {Array<object>} candidates, best first
 */
export function extractMoney(text, opts) {
  const o = opts || {};
  // A character class, so the symbols must be escaped INTO the class (not joined
  // with "|" — `(?:£€$)` is a literal three-character sequence, not an alternation,
  // and silently matches nothing). This exact bug made every `£1,234.56` in the
  // fixtures parse with `currency: null`.
  const symClass = `[${Object.keys(CURRENCY_SYMBOLS).map((s) => s.replace(/[\^\]\\-]/g, '\\$&')).join('')}]`;
  const codeAlt = CURRENCY_CODES.join('|');
  // Horizontal whitespace ONLY — a money token must never span a line break.
  // `\s` includes the newline, and the `\s?` sitting before the optional closing
  // paren was silently swallowing it: `1,204.50\n` matched, and because the match
  // ended *past* the line break, `lineContext()` then scored the NEXT line's
  // context words ("DEBIT", "AMOUNT DUE") against this amount. That is the exact
  // cross-line bleed `lineContext()` exists to prevent, reintroduced through the
  // regex. `[^\S\r\n]` is "whitespace that is not a line ending".
  const HS = '[^\\S\\r\\n]';
  // The currency marker is NOT consumed by this match, so it can also prefix the
  // *next* amount on the same line ("£1.00 £2.00") — the leader is checked
  // explicitly below instead, which a greedy optional prefix cannot do.
  const re = new RegExp(
    `(?:(?:\\(${HS}*)?`                                 // optional opening paren (accounting negative)
    + `(-)?${HS}?(\\d{1,3}(?:[.,'\\u00a0 ]\\d{3})+(?:[.,]\\d{1,2})?|\\d+(?:[.,]\\d{1,2})?)` // the number
    + `${HS}?(\\))?`                                    // optional closing paren
    + `(?:${HS}?(?:-|\\bDR\\b|\\bdb\\b))?)`,             // trailing negative marker
    'gi',
  );

  const out = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    const whole = m[0];
    const [full, leadMinus, digits, closeParen] = m;

    // Leader: the currency symbol (or code) immediately before the number.
    const lead = text.slice(Math.max(0, m.index - 4), m.index);
    const leadSym = (lead.match(new RegExp(`${symClass}\\s?$`)) || [null])[0];
    const leadCode = (lead.match(new RegExp(`\\b(${codeAlt})\\b\\s*$`, 'i')) || [null])[1];

    // Trailer: currency after the number, e.g. "123.45 EUR".
    const after = text.slice(m.index + whole.length, m.index + whole.length + 6);
    const trailSym = (after.match(new RegExp(`^\\s?(${symClass})`)) || [null])[1];
    const trailCode = (after.match(new RegExp(`^\\s?\\b(${codeAlt})\\b`, 'i')) || [null])[1];

    const symbol = CURRENCY_SYMBOLS[(leadSym || trailSym || '').trim()] || null;
    const code = (leadCode || trailCode || '').toUpperCase() || null;
    const currency = symbol || code || null;

    const parsed = parseAmountToken(digits);
    if (parsed.value === null) continue;

    // A bare integer with no currency marker and no context is not money —
    // it is a page number, a quantity, a year. Requiring *some* money evidence
    // is what stops the summary reading out "2".
    // Context is LINE-SCOPED: see lineContext() for why.
    const ctx = lineContext(text, m.index, m.index + whole.length, 60, 60);
    const window = `${ctx.before} ${whole} ${ctx.after}`;
    let contextWeight = 0;
    const contextLabels = [];
    for (const c of MONEY_CONTEXT) {
      if (c.re.test(window)) { contextWeight += c.weight; contextLabels.push(c.label); }
    }

    const trailingMinus = /-|\bDR\b|\bdb\b/.test(whole.slice(digits.length ? whole.indexOf(digits) + digits.length : 0));
    const negative = !!leadMinus || !!closeParen || trailingMinus;

    let confidence = 0.45;                    // a clean parse of a plausible number
    if (currency) confidence += 0.2;
    if (parsed.decimalSep) confidence += 0.08;
    if (contextWeight) confidence += contextWeight;
    if (!currency && !contextWeight) confidence -= 0.25;   // bare number, no evidence
    if (parsed.ambiguous) confidence -= 0.12;
    if (digits.replace(/\D/g, '').length > 12) confidence -= 0.3; // not a real amount

    // Rank on the *unclamped* evidence, otherwise several amounts all sit at
    // confidence 1 and the winner is decided purely by document order.
    const evidence = contextWeight + (currency ? 0.5 : 0) + (parsed.decimalSep ? 0.2 : 0);
    out.push({
      kind: 'money',
      value: negative ? -parsed.value : parsed.value,
      amount: parsed.amount,
      currency,
      negative,
      raw: squeezeWhitespace(whole),
      line: ctx.line,
      index: m.index,
      confidence: round4(clamp01(confidence)),
      context: contextLabels,
      contextWeight: round4(contextWeight),
      ambiguous: parsed.ambiguous,
      valid: parsed.value <= 1e9,
      reason: parsed.reason,
      // Prefer stronger currency+context evidence; then earlier in the document.
      rank: evidence * 1000 - m.index / 1000,
    });
  }

  out.sort((a, b) => b.rank - a.rank);
  return out;
}

// ---------------------------------------------------------------------------
// DATES — §6.4 "ISO, DD/MM/YYYY, D Mon YYYY, and relative forms — normalise to
// ISO, and disambiguate day/month by validating both readings against a
// plausible-date rule; if both are valid, mark the field low-confidence rather
// than guessing"
// ---------------------------------------------------------------------------

const MONTHS = {
  jan: 1, january: 1, feb: 2, february: 2, mar: 3, march: 3, apr: 4, april: 4,
  may: 5, jun: 6, june: 6, jul: 7, july: 7, aug: 8, august: 8,
  sep: 9, sept: 9, september: 9, oct: 10, october: 10, nov: 11, november: 11, dec: 12, december: 12,
};

const RELATIVE_DAYS = { today: 0, yesterday: -1, tomorrow: 1 };

/**
 * Resolve a relative expression against a reference date.
 * @param {string} phrase
 * @param {Date} now
 * @returns {{iso:string, offsetDays:number, phrase:string}|null}
 */
export function parseRelativeDate(phrase, now) {
  const ref = now || new Date();
  const p = squeezeWhitespace(phrase).toLowerCase();
  if (Object.prototype.hasOwnProperty.call(RELATIVE_DAYS, p)) {
    const d = addDays(ref, RELATIVE_DAYS[p]);
    return { iso: toIsoDate(d.getFullYear(), d.getMonth() + 1, d.getDate()), offsetDays: RELATIVE_DAYS[p], phrase: p };
  }
  let m = /^in\s+(\d{1,3})\s+(day|days|week|weeks|month|months)$/.exec(p);
  if (m) {
    const n = Number(m[1]);
    const months = /month/.test(m[2]);
    const days = /week/.test(m[2]) ? n * 7 : months ? 0 : n;
    const d = months ? addMonths(ref, n) : addDays(ref, days);
    return { iso: toIsoDate(d.getFullYear(), d.getMonth() + 1, d.getDate()), offsetDays: days, phrase: p };
  }
  m = /^(\d{1,3})\s+(day|days|week|weeks)\s+ago$/.exec(p);
  if (m) {
    const n = Number(m[1]);
    const days = -(/week/.test(m[2]) ? n * 7 : n);
    const d = addDays(ref, days);
    return { iso: toIsoDate(d.getFullYear(), d.getMonth() + 1, d.getDate()), offsetDays: days, phrase: p };
  }
  m = /^next\s+(week|month|year)$/.exec(p);
  if (m) {
    const d = m[1] === 'week' ? addDays(ref, 7) : m[1] === 'month' ? addMonths(ref, 1) : addMonths(ref, 12);
    return { iso: toIsoDate(d.getFullYear(), d.getMonth() + 1, d.getDate()), offsetDays: 7, phrase: p };
  }
  m = /^last\s+(week|month|year)$/.exec(p);
  if (m) {
    const d = m[1] === 'week' ? addDays(ref, -7) : m[1] === 'month' ? addMonths(ref, -1) : addMonths(ref, -12);
    return { iso: toIsoDate(d.getFullYear(), d.getMonth() + 1, d.getDate()), offsetDays: -7, phrase: p };
  }
  return null;
}

function addDays(date, n) {
  const d = new Date(date.getTime());
  d.setDate(d.getDate() + n);
  return d;
}

function addMonths(date, n) {
  const d = new Date(date.getTime());
  const day = d.getDate();
  d.setDate(1);
  d.setMonth(d.getMonth() + n);
  const last = new Date(d.getFullYear(), d.getMonth() + 1, 0).getDate();
  d.setDate(Math.min(day, last));
  return d;
}

/**
 * Two-digit years: 00-49 -> 2000s, 50-99 -> 1900s. Documented and fixed rather
 * than sliding with the clock, so a test is reproducible.
 */
export function expandYear(y) {
  const n = Number(y);
  if (String(y).length === 4) return n;
  return n <= 49 ? 2000 + n : 1900 + n;
}

/**
 * Extract dates. Every candidate carries the evidence for its confidence, and a
 * day/month-ambiguous numeric date is *forced* low so the gate omits it.
 * @param {string} text
 * @param {{now?:Date}} [opts]
 */
export function extractDates(text, opts) {
  const o = opts || {};
  const now = o.now || new Date();
  const out = [];

  // --- ISO 2024-03-15 --------------------------------------------------------
  for (const m of text.matchAll(/\b(\d{4})-(\d{2})-(\d{2})\b/g)) {
    const [full, y, mo, d] = m;
    const year = Number(y); const month = Number(mo); const day = Number(d);
    const valid = isRealDate(year, month, day);
    out.push({
      kind: 'date', format: 'iso', raw: full, iso: valid ? toIsoDate(year, month, day) : null,
      index: m.index, valid, ambiguous: false,
      // ISO is self-describing: no ambiguity is possible, so it scores high.
      confidence: valid ? 0.96 : 0.0,
      reason: valid ? 'ISO 8601' : `not a real date: ${full}`,
    });
  }

  // --- 15 Mar 2024 / 15 March 2024 ------------------------------------------
  // "15 March 2024" and the legal-deed form "15th day of March, 2024".
  for (const m of text.matchAll(/\b(\d{1,2})\s*(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?([A-Za-z]{3,9})\.?\s*,?\s*(\d{4}|\d{2})\b/g)) {
    const [full, d, monName, y] = m;
    const month = MONTHS[monName.toLowerCase()];
    const day = Number(d);
    const year = expandYear(y);
    if (!month) continue;
    const valid = isRealDate(year, month, day);
    out.push({
      kind: 'date', format: 'd-mon-yyyy', raw: full, iso: valid ? toIsoDate(year, month, day) : null,
      index: m.index, valid, ambiguous: false,
      confidence: valid ? 0.94 : 0.0,
      reason: valid ? 'named month, unambiguous' : `not a real date: ${full}`,
    });
  }

  // --- Mar 15, 2024 (the US ordering of the same, still unambiguous) ---------
  for (const m of text.matchAll(/\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4}|\d{2})\b/g)) {
    const [full, monName, d, y] = m;
    const month = MONTHS[monName.toLowerCase()];
    if (!month) continue;
    const day = Number(d);
    const year = expandYear(y);
    const valid = isRealDate(year, month, day);
    // Skip if this is the tail of a "15 Mar 2024" match already captured above.
    if (out.some((f) => f.format === 'd-mon-yyyy' && f.index <= m.index && f.index + f.raw.length >= m.index + full.length)) continue;
    out.push({
      kind: 'date', format: 'mon-d-yyyy', raw: full, iso: valid ? toIsoDate(year, month, day) : null,
      index: m.index, valid, ambiguous: false,
      confidence: valid ? 0.92 : 0.0,
      reason: valid ? 'named month, unambiguous' : `not a real date: ${full}`,
    });
  }

  // --- numeric DD/MM/YYYY vs MM/DD/YYYY — THE AMBIGUOUS CASE ----------------
  for (const m of text.matchAll(/\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})\b/g)) {
    const [full, a, b, y] = m;
    const year = expandYear(y);
    // Skip anything already claimed by the ISO pass.
    if (out.some((f) => f.index === m.index)) continue;

    const asDayFirst = { day: Number(a), month: Number(b) };   // DD/MM
    const asMonthFirst = { day: Number(b), month: Number(a) }; // MM/DD
    const okDayFirst = isRealDate(year, asDayFirst.month, asDayFirst.day);
    const okMonthFirst = isRealDate(year, asMonthFirst.month, asMonthFirst.day);

    let iso = null;
    let ambiguous = false;
    let valid = false;
    let confidence = 0;
    let reason = '';

    if (okDayFirst && okMonthFirst) {
      const isoA = toIsoDate(year, asDayFirst.month, asDayFirst.day);
      const isoB = toIsoDate(year, asMonthFirst.month, asMonthFirst.day);
      if (isoA === isoB) {
        // "03/03/2024" — both readings agree, so there is nothing to guess.
        iso = isoA; valid = true; ambiguous = false; confidence = 0.9;
        reason = 'numeric date, both readings agree';
      } else {
        // §6.4: "if both are valid, mark the field low-confidence rather than
        // guessing". This is the rule the G2 test asserts: 03/04/2024 must NOT
        // be spoken. Confidence is forced to 0.35, far below CONFIDENCE_MIN (0.75)
        // and below the 0.45 "bare number" floor, so no amount of OCR optimism
        // can push it over the gate.
        iso = null; valid = false; ambiguous = true; confidence = 0.35;
        reason = `ambiguous day/month: could be ${isoA} (DD/MM) or ${isoB} (MM/DD)`;
      }
    } else if (okDayFirst) {
      iso = toIsoDate(year, asDayFirst.month, asDayFirst.day);
      valid = true; confidence = 0.9; reason = 'first field > 12, so DD/MM is the only valid reading';
    } else if (okMonthFirst) {
      iso = toIsoDate(year, asMonthFirst.month, asMonthFirst.day);
      valid = true; confidence = 0.9; reason = 'second field > 12, so MM/DD is the only valid reading';
    } else {
      confidence = 0.0; reason = `not a real date in either reading: ${full}`;
    }

    out.push({
      kind: 'date', format: 'numeric', raw: full, iso, index: m.index,
      valid, ambiguous, confidence, reason,
      readings: {
        ddmm: okDayFirst ? toIsoDate(year, asDayFirst.month, asDayFirst.day) : null,
        mmdd: okMonthFirst ? toIsoDate(year, asMonthFirst.month, asMonthFirst.day) : null,
      },
    });
  }

  // --- relative forms --------------------------------------------------------
  const relRe = /\b(today|yesterday|tomorrow|in\s+\d{1,3}\s+(?:days?|weeks?|months?)|next\s+(?:week|month|year)|last\s+(?:week|month|year)|\d{1,3}\s+(?:days?|weeks?)\s+ago)\b/gi;
  for (const m of text.matchAll(relRe)) {
    const resolved = parseRelativeDate(m[0], now);
    if (!resolved) continue;
    out.push({
      kind: 'date', format: 'relative', raw: m[0], iso: resolved.iso, index: m.index,
      valid: true, ambiguous: false,
      // Resolved against today's date, so it is only as good as the clock —
      // scored below an explicit calendar date on purpose.
      confidence: 0.8,
      reason: `relative to today: ${resolved.phrase} -> ${resolved.iso}`,
    });
  }

  out.sort((a, b) => b.confidence * 1000 - a.confidence * 1000 - (a.index - b.index) / 1000);
  // Candidates that are provably not dates (confidence exactly 0) are moved to a
  // `rejected` list rather than dropped: the transcript stays honest about what
  // the scanner looked at, and the gate never has to consider them. Keeping them
  // in the main list would also let one bogus match occupy the "best date" slot.
  const accepted = out.filter((d) => d.confidence > 0);
  accepted.rejected = out.filter((d) => d.confidence === 0);
  return accepted;
}

// ---------------------------------------------------------------------------
// DOSAGE — §6.4 "number + unit (mg, mcg, ml, g, IU) + frequency phrases
// ('once daily', 'three times a day', 'every 8 hours', 'PRN') + duration
// ('for 7 days')"
// ---------------------------------------------------------------------------

const DOSE_UNITS = [
  { re: /^(?:mg|milligram|milligrams)$/i, unit: 'mg', kind: 'mass' },
  { re: /^(?:mcg|ug|\u00b5g|microgram|micrograms)$/i, unit: 'mcg', kind: 'mass' },
  { re: /^(?:g|gram|grams|gm)$/i, unit: 'g', kind: 'mass' },
  { re: /^(?:ml|mL|millilitre|milliliter|millilitres|milliliters)$/i, unit: 'ml', kind: 'volume' },
  { re: /^(?:l|litre|liter|litres|liters)$/i, unit: 'l', kind: 'volume' },
  { re: /^(?:iu|i\.u\.)$/i, unit: 'IU', kind: 'activity' },
  { re: /^(?:units?|u)$/i, unit: 'units', kind: 'activity' },
  { re: /^(?:tablets?|tabs?)$/i, unit: 'tablet', kind: 'form' },
  { re: /^(?:capsules?|caps?)$/i, unit: 'capsule', kind: 'form' },
  { re: /^(?:drops?)$/i, unit: 'drop', kind: 'form' },
  { re: /^(?:puffs?)$/i, unit: 'puff', kind: 'form' },
  { re: /^(?:sprays?)$/i, unit: 'spray', kind: 'form' },
  { re: /^(?:patches?)$/i, unit: 'patch', kind: 'form' },
  { re: /^(?:suppositor(?:y|ies))$/i, unit: 'suppository', kind: 'form' },
  { re: /^(?:%)$/i, unit: '%', kind: 'concentration' },
];

const FREQUENCY_PHRASES = [
  { re: /\b(?:once|one time|1\s*x?)\s*(?:a|per|each|every)?\s*(?:day|daily|24\s*hours?)\b/i, phrase: 'once daily', confidence: 0.92 },
  { re: /\b(?:twice|two times|2\s*x?)\s*(?:a|per|each|every)?\s*(?:day|daily|24\s*hours?)\b/i, phrase: 'twice daily', confidence: 0.92 },
  { re: /\b(?:three times|thrice|3\s*x?)\s*(?:a|per|each)?\s*(?:day|daily|24\s*hours?)\b/i, phrase: 'three times a day', confidence: 0.92 },
  { re: /\b(?:four times|4\s*x?)\s*(?:a|per|each)?\s*(?:day|daily|24\s*hours?)\b/i, phrase: 'four times a day', confidence: 0.9 },
  { re: /\bevery\s+(\d{1,2})\s*hours?\b/i, phrase: 'every N hours', confidence: 0.9, capture: 1 },
  { re: /\bevery\s+(\d{1,2})\s*days?\b/i, phrase: 'every N days', confidence: 0.88, capture: 1 },
  { re: /\bq(\d{1,2})h\b/i, phrase: 'every N hours (qNh)', confidence: 0.85, capture: 1 },
  { re: /\b(\d{1,2})\s*(?:times|x)\s*(?:a|per)\s*(?:day|week|night|24\s*hours?)\b/i, phrase: 'N times a day', confidence: 0.88, capture: 1 },
  { re: /\b(?:prn|p\.r\.n\.)\b/i, phrase: 'as needed (PRN)', confidence: 0.9 },
  { re: /\bas\s+(?:needed|required)\b/i, phrase: 'as needed', confidence: 0.88 },
  { re: /\b(?:bd|b\.d\.|bds)\b/i, phrase: 'twice daily (bd)', confidence: 0.85 },
  { re: /\b(?:tds|t\.d\.s\.|tid)\b/i, phrase: 'three times a day (tds)', confidence: 0.85 },
  { re: /\b(?:qds|q\.d\.s\.|qid)\b/i, phrase: 'four times a day (qds)', confidence: 0.85 },
  { re: /\b(?:od|o\.d\.)\b/i, phrase: 'once daily (od)', confidence: 0.8 },
  { re: /\b(?:nocte|at night|nightly)\b/i, phrase: 'at night', confidence: 0.85 },
  { re: /\b(?:mane|in the morning|each morning)\b/i, phrase: 'in the morning', confidence: 0.85 },
  { re: /\b(?:weekly|once a week)\b/i, phrase: 'once a week', confidence: 0.88 },
];

// EVERY pattern here must CAPTURE the unit as group 2 -- the metadata says so and both call
// sites read `hit[2].toLowerCase()`. The two "day course"/"day supply" patterns shipped with a
// NON-capturing `(?:day|days)`, so a match had NO group 2, `hit[2]` was undefined, and the
// TypeError took down the whole scan: "Cannot read properties of undefined (reading
// 'toLowerCase')" on every prescription, the one document this product is for. The default
// markers also name the unit when the text is a course/supply rather than "for N days".
const DURATION_PHRASES = [
  { re: /\bfor\s+(\d{1,3})\s*(day|days|week|weeks|month|months)\b/i, capture: 2, unitIndex: 2 },
  { re: /\bx\s*(\d{1,3})\s*(day|days|week|weeks|month|months)\b/i, capture: 2, unitIndex: 2 },
  { re: /\b(\d{1,3})[\s-]*day\s+course\b/i, capture: 2, unitIndex: 2, unit: "day" },
  { re: /\b(\d{1,3})\s*(day|days)\s+supply\b/i, capture: 2, unitIndex: 2 },
];

/**
 * Extract dosage instructions. A dose is only useful with a frequency; a bare
 * "500 mg" tells a blind user nothing about how to take it, so a dose without a
 * frequency is scored well below the gate and the summary says it is unclear.
 */
export function extractDosage(text, opts) {
  const o = opts || {};
  const out = [];
  const punct = '.,;:()\\[\\]';
  const unitAlt = 'mg|milligrams?|mcg|ug|\u00b5g|micrograms?|gm|g|grams?|m[lL]|millilit(?:re|er)s?|l|lit(?:re|er)s?|iu|i\\.u\\.|units?|u|tablets?|tabs?|capsules?|caps?|drops?|puffs?|sprays?|patches?|suppositor(?:y|ies)';
  const re = new RegExp(`(?<![\\d.])(\\d{1,4}(?:\\.\\d{1,2})?)\\s*(${unitAlt})\\b`, 'gi');

  for (const m of text.matchAll(re)) {
    const [full, num, unitRaw] = m;
    const unitDef = DOSE_UNITS.find((u) => u.re.test(unitRaw.trim()));
    const amount = Number(num);
    if (!isFinite(amount)) continue;

    // The instruction usually trails the dose. Look ahead first, then behind —
    // but only on the SAME LINE: a frequency phrase on the next line belongs to
    // the next medication, not to this one.
    const ctx = lineContext(text, m.index, m.index + full.length, 70, 120);
    const after = ctx.after;
    const before = ctx.before;
    let frequency = null;
    let freqConfidence = 0;
    for (const f of FREQUENCY_PHRASES) {
      const hitAfter = f.re.exec(after);
      const hitBefore = hitAfter ? null : f.re.exec(before);
      const hit = hitAfter || hitBefore;
      if (hit) {
        let phrase = f.phrase;
        if (f.capture && hit[f.capture]) phrase = phrase.replace('N', hit[f.capture]);
        frequency = { phrase, raw: squeezeWhitespace(hit[0]), confidence: f.confidence };
        freqConfidence = f.confidence;
        break;
      }
    }

    let duration = null;
    for (const d of DURATION_PHRASES) {
      const hit = d.re.exec(after) || d.re.exec(before);
      if (hit && hit[1] != null && (hit[2] != null || d.unit)) {
        duration = { raw: squeezeWhitespace(hit[0]), count: Number(hit[1]),
                     unit: String(hit[2] != null ? hit[2] : d.unit).toLowerCase().replace(/s$/, '') };
        break;
      }
    }

    let confidence = 0.5;
    if (unitDef && unitDef.kind === 'mass' || (unitDef && unitDef.kind === 'volume')) confidence += 0.1;
    if (amount > 0 && amount < 20000) confidence += 0.05;
    else confidence -= 0.25;                       // 84000 mg is not a dose
    if (frequency) confidence += 0.28 * (freqConfidence / 0.92);
    else confidence -= 0.22;                       // §6.4: dose without frequency
    if (duration) confidence += 0.06;

    const unitClean = unitRaw.replace(new RegExp(`^[${punct}\\s]+|[${punct}\\s]+$`, 'g'), '');

    out.push({
      kind: 'dosage',
      amount,
      unit: (unitDef && unitDef.unit) || unitClean.toLowerCase(),
      unitRaw: unitClean,
      unitKind: unitDef ? unitDef.kind : 'unknown',
      frequency: frequency ? frequency.phrase : null,
      frequencyRaw: frequency ? frequency.raw : null,
      duration,
      // The instruction line, so the transcript and the tests can see exactly
      // what evidence the dose was read from.
      raw: ctx.line || squeezeWhitespace(full),
      index: m.index,
      confidence: round4(clamp01(confidence)),
      valid: !!unitDef && amount > 0,
      reason: frequency
        ? `dose with frequency (${frequency.raw})`
        : 'dose with no frequency phrase — not enough to act on',
    });
  }

  out.sort((a, b) => b.confidence - a.confidence || a.index - b.index);
  return out;
}

// ---------------------------------------------------------------------------
// ACCOUNT / REFERENCE NUMBERS — §6.4 "long digit runs, with all but the last 4
// digits masked when spoken"; gate: "Never speak a full account number"
// ---------------------------------------------------------------------------

/** Luhn check (ISO/IEC 7812). Catches most OCR digit slips in a card number. */
export function luhnCheck(digits) {
  const s = String(digits).replace(/\D/g, '');
  if (s.length < 12) return false;
  let sum = 0;
  let double = false;
  for (let i = s.length - 1; i >= 0; i--) {
    let d = s.charCodeAt(i) - 48;
    if (double) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
    double = !double;
  }
  return sum % 10 === 0;
}

/**
 * True when a digit run is obviously not an identifier.
 *
 * ONLY the modifications that are found in every plausible corpus are rejected
 * here — otherwise this becomes an OCR-confidence feature, which is wrong.
 *
 * An earlier version also rejected any run containing `1234567890` or
 * `0123456789`. That is far too broad: the fixture "Account number: 1234 5678
 * 9012" contains "456789" and was thrown away, so the app silently lost a real
 * account number. The strict ascending/descending test is what actually
 * identifies a fabricated number, and it survives that fixture.
 */
export function looksDegenerate(digits) {
  const s = String(digits).replace(/\D/g, '');
  if (s.length < 4) return true;
  if (/^(\d)\1+$/.test(s)) return true;                    // 000000000 / 88888888
  if (/^(?:0{4,}|9{4,})/.test(s) && /^(\d)\1+$/.test(s)) return true;
  const codes = s.split('').map((c) => c.charCodeAt(0));
  const asc = codes.every((c, i) => i === 0 || c === codes[i - 1] + 1);
  const desc = codes.every((c, i) => i === 0 || c === codes[i - 1] - 1);
  if (asc || desc) return true;                             // 12345678 / 98765432
  return false;
}

/**
 * Mask an identifier for speech: "ending 3456". The full number is never
 * returned by any speech-facing function in this app.
 */
export function maskAccountForSpeech(digits) {
  const s = String(digits).replace(/\D/g, '');
  if (s.length < 4) return 'ending unknown';
  return `ending ${s.slice(-4)}`;
}

/**
 * Mask for display: keep the last four, star out the rest, keep any separators
 * so the shape is still recognisable.
 */
export function maskAccountForDisplay(raw) {
  const s = String(raw);
  const digitPositions = [];
  for (let i = 0; i < s.length; i++) if (/\d/.test(s[i])) digitPositions.push(i);
  if (digitPositions.length < 4) return s;
  const keepFrom = digitPositions.length - 4;
  const chars = s.split('');
  digitPositions.slice(0, keepFrom).forEach((p) => { chars[p] = '\u2022'; });
  return chars.join('');
}

const ACCOUNT_CONTEXT = [
  { re: /\b(?:account|acct|a\/c)\s*(?:number|no\.?|#)?/i, weight: 0.25, label: 'account' },
  { re: /\b(?:card|credit\s+card|debit\s+card)\s*(?:number|no\.?|#)?/i, weight: 0.25, label: 'card' },
  { re: /\b(?:sort\s*code|routing|iban|swift|bic)\b/i, weight: 0.22, label: 'banking code' },
  { re: /\b(?:reference|ref|invoice|statement|customer|client|member|policy|claim|nhs|national\s+insurance|nino|ssn|vat|tax)\s*(?:number|no\.?|#|id)?/i, weight: 0.2, label: 'reference' },
  { re: /\b(?:patient|prescription|rx|casenote|file)\s*(?:number|no\.?|#|id)?/i, weight: 0.2, label: 'record id' },
  { re: /\b(?:your|our)\s+(?:number|no)\b/i, weight: 0.08, label: 'weak context' },
];

/**
 * Extract account / reference numbers.
 * Length floor is 8: a 6-digit "12-34-56" sort code alone is not an account
 * number, and 4-digit groups would match every year in the document.
 */
export function extractAccounts(text, opts) {
  const o = opts || {};
  const minDigits = o.minDigits || 8;
  const maxDigits = o.maxDigits || 34;
  const out = [];
  // Digit runs allowing single space / dash / dot separators inside the run.
  const re = /\b(\d[\d '.\-]{5,48}\d)\b/g;

  for (const m of text.matchAll(re)) {
    const raw = squeezeWhitespace(m[0]);
    const digits = raw.replace(/\D/g, '');
    if (digits.length < minDigits || digits.length > maxDigits) continue;
    if (looksDegenerate(digits)) continue;

    const ctx = lineContext(text, m.index, m.index + m[0].length, 45, 20);
    const window = `${ctx.before} ${raw} ${ctx.after}`;
    let contextWeight = 0;
    const labels = [];
    for (const c of ACCOUNT_CONTEXT) {
      if (c.re.test(window)) { contextWeight += c.weight; labels.push(c.label); }
    }

    const isCard = digits.length >= 15 && digits.length <= 19;
    const luhn = isCard ? luhnCheck(digits) : null;

    let confidence = 0.35;
    confidence += contextWeight;
    if (isCard) {
      confidence += luhn ? 0.35 : -0.1;   // a failing Luhn is evidence of a misread
    } else if (digits.length >= 9 && digits.length <= 18) {
      confidence += 0.15;                 // plausible account/reference length
    } else if (digits.length >= 24) {
      confidence -= 0.2;                  // probably concatenated noise
    }
    if (/(\d)\1{4,}/.test(digits)) confidence -= 0.2;      // 5555555 in the middle
    if (/^\d{8}$/.test(digits) && !contextWeight) confidence -= 0.15; // bare 8 digits (a date?)

    out.push({
      kind: 'account',
      digits,
      raw,
      masked: maskAccountForDisplay(raw),
      speakAs: maskAccountForSpeech(digits),
      length: digits.length,
      isCard,
      luhn,
      context: labels,
      index: m.index,
      confidence: round4(clamp01(confidence)),
      // Format/checksum sanity per §6.4. A card number must pass Luhn to be
      // considered valid; an account number only needs a plausible length.
      valid: isCard ? luhn === true : (digits.length >= minDigits && digits.length <= 22 && contextWeight > 0),
      reason: isCard
        ? (luhn ? 'card-length number passing the Luhn checksum' : 'card-length number FAILING the Luhn checksum')
        : (contextWeight > 0 ? 'identifier with account/reference context' : 'digit run with no account context'),
    });
  }

  out.sort((a, b) => b.confidence - a.confidence || a.index - b.index);
  return out;
}

// ---------------------------------------------------------------------------
// DOCUMENT TYPE — §6.4's weighted keyword scoring, narrowed this round to
// receipt / prescription plus an `unrecognised` fallback (docs/DECISIONS.md D34).
// ---------------------------------------------------------------------------

// THE TWO VOCABULARIES (this round's brief §3.1). `banking`, `legal` and the old
// catch-all `general` are gone: billing, letters and legal documents are out of
// scope and must come back `unrecognised` (docs/DECISIONS.md D34).
//
// The weights matter as much as the words. A bank statement contains the word
// "CARD" and the phrase "AMOUNT DUE"; neither is a receipt signal. So the receipt
// vocabulary leans on words a statement does not have — RECEIPT, TILL, VAT,
// "thank you for shopping" — and the classifier additionally refuses to name a
// type below an absolute score floor (see MIN_TYPE_SCORE).
const DOC_KEYWORDS = {
  receipt: [
    [/\b(?:receipt|till\s*(?:no|number|#)?|transaction\s*(?:no|number|id)|ref(?:erence)?\s*no)\b/i, 3],
    [/\b(?:sub-?total|total|amount\s+paid|balance\s+due|change\s+due)\b/i, 3],
    [/\b(?:vat|gst|tax|cash|card|change|contactless|chip\s*&?\s*pin)\b/i, 2],
    [/\b(?:visa|mastercard|master\s*card|amex|american\s+express)\b/i, 2],
    [/\b(?:supermarket|market|grocer|shop|store|restaurant|cafe|café|fuel|petrol|garage)\b/i, 2],
    [/\b(?:thank\s+you\s+for\s+shopping|please\s+come\s+again|items?|qty)\b/i, 2],
    [/\b(?:opening\s+hours|tel|phone)\b/i, 1],
  ],
  prescription: [
    [/\b(?:prescription|prescribed|repeat\s+prescription)\b/i, 4],
    [/\brx\b/i, 4],
    [/\b(?:tablet|capsule|dose|dosage|sig|take\s+(?:one|two|1|2))\b/i, 3],
    [/\b\d+\s*(?:mg|mcg|ml|iu|g)\b/i, 3],
    [/\b(?:once|twice|three\s+times|four\s+times)\s+(?:a|per|daily)\b|\bevery\s+\d+\s+hours\b|\b(?:prn|bd|tds|qds|as\s+needed)\b/i, 3],
    [/\b(?:patient|prescriber|dispensed|dispense|pharmacy|chemist|repeats?|refills?)\b/i, 3],
    [/\b(?:gp|doctor|dr\.?|physician|consultant|surgery|clinic|medical\s+cent(?:re|er))\b/i, 2],
    [/\b(?:date\s+of\s+birth|dob|nhs|medicare|quantity)\b/i, 2],
  ],
};

// The absolute score below which NO type is named. A single weak keyword must not
// be enough — that is exactly the mistake that would label a bank statement a
// receipt because it contains "CARD PAYMENT". Measured on the fixtures: the
// receipt scores in the high teens, the prescription in the low twenties, and the
// kept bank statement scores 2 (the single word "card"). 5 sits cleanly between.
const MIN_TYPE_SCORE = 5;

/**
 * Weighted keyword scoring for the document type (§6.4, narrowed this round).
 *
 * Returns `unrecognised` — never a guessed type — when the evidence is thin. The
 * `reason` string is what makes a misclassification debuggable on a real photo:
 * it names the winning score, the runner-up and the floor.
 *
 * @returns {{type:string, scores:object, ranked:string[], confidence:number,
 *            evidence:string[], reason:string}}
 */
export function classifyDocument(text) {
  const scores = {};
  const evidence = {};
  for (const type of DOC_TYPES) { scores[type] = 0; evidence[type] = []; }

  for (const type of DOC_TYPES) {
    for (const [re, weight] of DOC_KEYWORDS[type]) {
      const m = re.exec(text);
      if (m) {
        // Diminishing returns per keyword: three hits of one weak word must not
        // outweigh one decisive word such as "prescription".
        scores[type] += weight;
        if (evidence[type].length < 6) evidence[type].push(squeezeWhitespace(m[0]).slice(0, 40));
      }
    }
  }

  const ranked = DOC_TYPES.slice().sort((a, b) => scores[b] - scores[a]);
  const top = ranked[0];
  const runnerUp = ranked[1];
  const topScore = scores[top];
  const total = DOC_TYPES.reduce((a, t) => a + scores[t], 0);

  // THE FLOOR. Below it, name no type at all. This is the arm that keeps a bank
  // statement (which contains "CARD PAYMENT") out of the receipt bucket.
  if (topScore < MIN_TYPE_SCORE) {
    return {
      type: UNRECOGNISED_TYPE,
      scores,
      ranked,
      confidence: round4(clamp01(topScore / (2 * MIN_TYPE_SCORE))),
      evidence: [],
      reason: topScore === 0
        ? 'no receipt or prescription keywords matched'
        : `only ${topScore} point(s) of evidence for ${top} (needs ${MIN_TYPE_SCORE})`,
    };
  }

  // Confidence blends the margin over the runner-up with the share of all the
  // evidence. A page that is clearly one type and barely the other scores high;
  // a page with a little of both does not.
  const margin = (topScore - scores[runnerUp]) / topScore;
  const share = topScore / (total || 1);
  const confidence = clamp01(0.5 + 0.35 * margin + 0.15 * share);

  return {
    type: top,
    scores,
    ranked,
    confidence: round4(confidence),
    evidence: evidence[top],
    reason: `weighted score ${topScore} for ${top} vs ${scores[runnerUp]} for ${runnerUp}`,
  };
}

// ---------------------------------------------------------------------------
// THE FIELD LAYER — recognised text -> the labelled fields the results screen
// renders (this round's brief §3.3; the shape lives in js/taxonomy.js).
// ---------------------------------------------------------------------------
// WHY THIS LAYER EXISTS, SEPARATE FROM THE EXTRACTORS ABOVE
//   extractMoney/extractDates/extractDosage/extractAccounts answer "what values
//   are on this page?". This layer answers "which of the DOCUMENT'S fields did we
//   find?", which is a different question: a receipt's TOTAL and its SUBTOTAL are
//   both money, but only one of them is `total`. The mapping from a value to the
//   field it belongs to is the document model, so it is driven entirely by
//   js/taxonomy.js — every record here carries its {key, group, kind, label} from
//   that file, and nothing in this layer invents a field the taxonomy does not
//   declare.
//
//   A field that is not found is ABSENT from the map, never present-and-empty.
//   The results screen renders only what is here, and a group with no found field
//   does not render at all (js/ui.js). "Not found" and "found but blank" are the
//   same thing to a reader, and only one of them is honest.
//
//   The `confidence` on each record is what the speak-or-omit gate will read once
//   the grouped rows are spoken; it is deliberately conservative for a field whose
//   position on the page (rather than its own shape) is the only evidence.

/** Split recognised text into non-blank lines, keeping each line's character offset. */
function splitLines(text) {
  const out = [];
  let index = 0;
  for (const raw of String(text || '').split(/\r?\n/)) {
    const trimmed = raw.trim();
    if (trimmed) out.push({ text: trimmed, index });
    index += raw.length + 1;
  }
  return out;
}

/** The text of the line containing character `index` (lines are in offset order). */
function lineAt(lines, index) {
  let found = '';
  for (const l of lines) {
    if (l.index <= index) found = l.text;
    else break;
  }
  return found;
}

/** Build one field record in the shape js/taxonomy.js declares. */
function makeField(key, value, raw, confidence, line) {
  const def = FIELDS[key];
  return {
    key,
    type: def.type,
    group: def.group,
    kind: def.kind,
    label: def.label,
    value,
    raw: squeezeWhitespace(raw === undefined || raw === null ? String(value) : raw),
    line: line || null,
    confidence: round4(clamp01(confidence)),
    // `valid`/`ambiguous` are what gateField() reads. A field reaches this map
    // only because its pattern matched the page, so it is valid by construction
    // and `confidence` is what decides whether it may be spoken. Without these
    // two keys the gate treats every taxonomy field as invalid — `!undefined` is
    // true — so the results screen would omit every row it renders.
    valid: true,
    ambiguous: false,
  };
}

// Merchant -> spending category (this round's brief §3.3). A small gazetteer, and
// the ORDER matters: "GREENFIELD MARKET" is a supermarket, not retail, so the more
// specific words are tested first. Nothing is guessed — if no word matches, the
// `category` row is absent rather than "Unknown".
const MERCHANT_CATEGORY = [
  { re: /\b(?:supermarket|grocer|groceries|market|food\s*hall|hypermarket|convenience\s+store)\b/i, category: 'supermarket' },
  { re: /\b(?:pharmacy|chemist|drug\s*store|dispensary)\b/i, category: 'pharmacy' },
  { re: /\b(?:restaurant|cafe|café|coffee|bistro|diner|grill|pizzeria|pizza|burger|takeaway|take\s*away|kitchen|bakery)\b/i, category: 'restaurant' },
  { re: /\b(?:fuel|petrol|diesel|garage|service\s+station|filling\s+station)\b/i, category: 'fuel' },
  { re: /\b(?:transport|railway|rail|train|bus|coach|taxi|metro|airline|airport|parking|toll)\b/i, category: 'transport' },
  { re: /\b(?:online|\.com|\.co\.uk|mail\s+order|delivery)\b/i, category: 'online' },
  { re: /\b(?:retail|store|shop|boutique|outlet|clothing|fashion|electronics|hardware)\b/i, category: 'retail' },
];

// Dose FORM — the prescription's "type" row (§3.3). One canonical word per form.
const MEDICINE_FORMS = [
  { re: /\b(?:tablets?|tabs?)\b/i, form: 'tablet' },
  { re: /\b(?:capsules?|caps?)\b/i, form: 'capsule' },
  { re: /\b(?:inhalers?)\b/i, form: 'inhaler' },
  { re: /\b(?:syrups?|suspension|elixir|solution)\b/i, form: 'syrup' },
  { re: /\b(?:eye\s*drops?|ear\s*drops?|drops?)\b/i, form: 'drops' },
  { re: /\b(?:patches?)\b/i, form: 'patch' },
];

/** Canonical form word for a matched span ("CAPSULES" -> "capsule"). */
function normaliseForm(word) {
  const hit = MEDICINE_FORMS.find((f) => f.re.test(word));
  return hit ? hit.form : squeezeWhitespace(word).toLowerCase().replace(/s$/, '');
}

// Payment method. A brand is decisive; a generic "CARD" is not, which is why the
// brand rows come first and there is no bare `/\bcard\b/` row at all.
const PAYMENT_METHODS = [
  { re: /\bvisa\b/i, method: 'visa', confidence: 0.9 },
  { re: /\bmastercard\b|\bmaster\s+card\b/i, method: 'mastercard', confidence: 0.9 },
  { re: /\bamex\b|\bamerican\s+express\b/i, method: 'amex', confidence: 0.9 },
  { re: /\bcontactless\b/i, method: 'contactless', confidence: 0.85 },
  { re: /\bchip\s*(?:and|&)\s*pin\b/i, method: 'chip and pin', confidence: 0.85 },
  { re: /\bdebit\s+card\b/i, method: 'debit card', confidence: 0.8 },
  { re: /\bcredit\s+card\b/i, method: 'credit card', confidence: 0.8 },
  { re: /\bcash\b/i, method: 'cash', confidence: 0.8 },
];

/** The strongest money candidate whose line matches `re` (optionally not `exclude`). */
function moneyOn(money, re, exclude) {
  let found = null;
  for (const m of money || []) {
    const line = m.line || '';
    if (!re.test(line)) continue;
    if (exclude && exclude.test(line)) continue;
    if (!found || m.confidence > found.confidence) found = m;
  }
  return found;
}

/** The first date candidate that is valid AND sits on a line matching `re`. */
function dateOn(dates, lines, re) {
  return (dates || []).find((d) => d.iso && re.test(lineAt(lines, d.index))) || null;
}

// --- receipt ----------------------------------------------------------------

/** Lines that can never be the shop's name, because they are a labelled row. */
const RECEIPT_MERCHANT_SKIP = /\b(?:receipt|invoice|statement|order|till|vat|gst|tax|date|time|tel|phone|total|sub-?total|amount|cash|card|change|thank)\b/i;
const STREET_RE = /\b\d{1,4}[a-z]?\s+[A-Za-z0-9'’.\- ]*\b(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|way|close|court|crescent|place|square|highway|parade|terrace|boulevard|blvd)\b/i;
const PHONE_RE = /\+?\d[\d\s().\-]{7,17}\d/g;

function extractReceiptFields(ctx) {
  const { text, lines, money, dates } = ctx;
  const out = {};

  // --- the shop --------------------------------------------------------------
  const merchantLine = lines.find((l) => {
    if (RECEIPT_MERCHANT_SKIP.test(l.text)) return false;
    const alpha = (l.text.match(/[A-Za-z]/g) || []).length;
    const digits = (l.text.match(/\d/g) || []).length;
    return alpha >= 3 && alpha > digits;
  });
  if (merchantLine) {
    out.merchant = makeField('merchant', merchantLine.text, merchantLine.text, 0.86, merchantLine.text);
    const cat = MERCHANT_CATEGORY.find((c) => c.re.test(merchantLine.text));
    if (cat) {
      const hit = merchantLine.text.match(cat.re);
      out.category = makeField('category', cat.category, hit ? hit[0] : cat.category, 0.8, merchantLine.text);
    }
  }

  // The shop address: the first line after the letterhead that looks like one.
  const addressLine = lines.slice(1).find((l) => STREET_RE.test(l.text));
  if (addressLine) {
    // A letterhead often carries the phone on the same line; the address stops there.
    const cut = squeezeWhitespace(addressLine.text.split(/\b(?:tel|phone|telephone)\b/i)[0]);
    if (cut) out.shop_address = makeField('shop_address', cut, addressLine.text, 0.7, addressLine.text);
  }

  // The phone: a plausible-length digit run, strongest when the line says "tel".
  let phone = null;
  for (const l of lines) {
    for (const m of l.text.matchAll(PHONE_RE)) {
      const digits = m[0].replace(/\D/g, '');
      if (digits.length < 9 || digits.length > 15) continue;
      if (/^[12]\d{3}[01]\d[0-3]\d/.test(digits)) continue;              // yyyymmdd-shaped
      const confidence = /\b(?:tel|phone|telephone|mobile|call)\b/i.test(l.text) ? 0.9 : 0.78;
      if (!phone || confidence > phone.confidence) phone = makeField('shop_phone', digits, m[0], confidence, l.text);
    }
  }
  if (phone) out.shop_phone = phone;

  // --- the purchase ----------------------------------------------------------
  const DATE_LABEL = /\b(?:date|dated|purchase\s+date|transaction\s+date|txn\s+date)\b/i;
  const purchaseDate = dateOn(dates, lines, DATE_LABEL) || (dates || []).find((d) => d.iso) || null;
  if (purchaseDate) {
    out.purchase_date = makeField('purchase_date', purchaseDate.iso, purchaseDate.raw,
      purchaseDate.confidence, lineAt(lines, purchaseDate.index));
  }

  const TIME_RE = /\b([01]?\d|2[0-3]):([0-5]\d)\b/;
  for (const l of lines) {
    const m = l.text.match(TIME_RE);
    if (!m) continue;
    const hh = String(Number(m[1])).padStart(2, '0');
    const confidence = /\b(?:time|at)\b/i.test(l.text) ? 0.9 : 0.72;
    out.purchase_time = makeField('purchase_time', `${hh}:${m[2]}`, m[0], confidence, l.text);
    break;
  }

  // --- the totals ------------------------------------------------------------
  const SUBTOTAL_RE = /\bsub[\s-]*total\b/i;
  const TAX_RE = /\b(?:vat|gst|sales\s+tax|tax)\b/i;
  const TOTAL_RE = /\b(?:total|amount\s+(?:paid|due)|balance\s+due|grand\s+total|to\s+pay)\b/i;
  const sub = moneyOn(money, SUBTOTAL_RE);
  const tax = moneyOn(money, TAX_RE);
  const total = moneyOn(money, TOTAL_RE, SUBTOTAL_RE);
  if (sub) out.subtotal = makeField('subtotal', sub.value, sub.raw, sub.confidence, sub.line);
  if (tax) out.tax = makeField('tax', tax.value, tax.raw, tax.confidence, tax.line);
  if (total) out.total = makeField('total', total.value, total.raw, total.confidence, total.line);

  // --- payment ---------------------------------------------------------------
  for (const l of lines) {
    const pm = PAYMENT_METHODS.find((p) => p.re.test(l.text));
    if (!pm) continue;
    const hit = l.text.match(pm.re);
    out.payment_method = makeField('payment_method', pm.method, hit ? hit[0] : pm.method, pm.confidence, l.text);
    break;
  }

  // LAST FOUR DIGITS ONLY, EVER. The extractor never keeps a full PAN: it reads
  // the four digits the receipt itself already prints, and nothing else.
  const MASKED_RE = /(?:[*•xX]{2,}|ending\s+(?:in\s+)?)[\s-]*(\d{4})\b/;
  let last4 = null;
  for (const l of lines) {
    const m = l.text.match(MASKED_RE);
    if (m) { last4 = { v: m[1], line: l.text }; break; }
  }
  if (!last4) {
    for (const l of lines) {
      if (!/\b(?:visa|mastercard|master\s+card|amex|american\s+express)\b/i.test(l.text)) continue;
      const digits = l.text.replace(/\D/g, '');
      if (digits.length >= 4) { last4 = { v: digits.slice(-4), line: l.text }; break; }
    }
  }
  if (last4) out.card_last4 = makeField('card_last4', last4.v, last4.line, 0.9, last4.line);

  // --- reference -------------------------------------------------------------
  const REF_RE = /\b(receipt|invoice|till|order|sale)\s*(?:no|number|#|id)?\.?\s*[:.]?\s*([A-Z0-9][A-Z0-9\-\/]{2,})/i;
  const ref = text.match(REF_RE);
  if (ref) out.receipt_number = makeField('receipt_number', ref[2], ref[0], 0.85, lineAt(lines, ref.index));
  const TX_RE = /\b(?:transaction|trans|txn|auth(?:orisation|orization)?)\s*(?:id|no|number|#)?\.?\s*[:.]?\s*([A-Z0-9][A-Z0-9\-\/]{3,})/i;
  const tx = text.match(TX_RE);
  if (tx) out.transaction_id = makeField('transaction_id', tx[1], tx[0], 0.8, lineAt(lines, tx.index));

  return out;
}

// --- prescription -----------------------------------------------------------

const CLINIC_RE = /\b(?:medical\s+cent(?:re|er)|health\s+cent(?:re|er)|clinic|surgery|practice|pharmacy|chemist|hospital|nhs)\b/i;
const PRESCRIBER_RE = /^(?:dr|doctor|prescriber|prescribed\s+by)\b/i;
const STRENGTH_RE = /\b(\d{1,5}(?:\.\d{1,2})?)\s*(mg|mcg|µg|ug|g|ml|iu|units?)\b/i;
const DOSE_INSTRUCTION_RE = /\b(?:take|use|swallow|inhale)\s+(one|two|three|four|\d{1,2})\s+(tablets?|tabs?|capsules?|caps?|drops?|puffs?|sprays?|patches?|suppositor(?:y|ies)s?)\b/i;
const DOB_RE = /\b(?:d\.?o\.?b\.?|date\s+of\s+birth)\b/i;
const ISSUED_RE = /\b(?:issued|date\s+issued|dispensed|date\s+dispensed)\b/i;
const VALID_UNTIL_RE = /\b(?:valid\s+until|expir(?:es|y|ation)|use\s+by)\b/i;
const WARNING_RE = /\b(?:warning|caution|do\s+not|keep\s+out\s+of|may\s+cause|avoid|seek\s+(?:medical|advice))\b/i;
const ROUTE_RE = /\b(?:by\s+mouth|orally|oral|topically|apply\s+to|inhale[drs]?|nasal(?:ly)?|sublingual(?:ly)?|rectal(?:ly)?)\b/i;

function extractPrescriptionFields(ctx) {
  const { text, lines, money, dates } = ctx;
  const out = {};

  const clinicLine = lines.find((l) => CLINIC_RE.test(l.text));
  if (clinicLine) out.clinic = makeField('clinic', clinicLine.text, clinicLine.text, 0.85, clinicLine.text);

  const drLine = lines.find((l) => PRESCRIBER_RE.test(l.text));
  if (drLine) {
    out.prescriber = makeField('prescriber', drLine.text, drLine.text, 0.85, drLine.text);
  }

  const patient = text.match(/^\s*patient(?:\s+name)?\s*[:.]?\s*(.+?)\s*$/im);
  if (patient) {
    const name = squeezeWhitespace(patient[1]);
    if (name) out.patient_name = makeField('patient_name', name, name, 0.85, lineAt(lines, patient.index));
  }

  const dob = dateOn(dates, lines, DOB_RE);
  if (dob) out.date_of_birth = makeField('date_of_birth', dob.iso, dob.raw, dob.confidence, lineAt(lines, dob.index));

  // The medicine itself: the line that states a strength, minus its trailing form.
  const strengthLine = lines.find((l) => !/^\s*(?:take|use|sig|directions|dose)\b/i.test(l.text) && STRENGTH_RE.test(l.text));
  if (strengthLine) {
    const sm = strengthLine.text.match(STRENGTH_RE);
    const unit = sm[2].toLowerCase();
    const unitNorm = (unit === 'ug' || unit === 'µg') ? 'mcg' : unit;
    out.strength = makeField('strength', `${Number(sm[1])} ${unitNorm}`, sm[0], 0.85, strengthLine.text);

    const namePart = strengthLine.text.slice(0, sm.index).replace(/[^A-Za-z'’.\- ]+$/, '').trim();
    const firstWord = namePart.split(/\s+/).filter(Boolean)[0] || '';
    if (/[A-Za-z]{3}/.test(firstWord)) {
      out.medicine = makeField('medicine', firstWord.toUpperCase(), firstWord, 0.8, strengthLine.text);
    }
  }

  const formLine = strengthLine || lines.find((l) => MEDICINE_FORMS.some((f) => f.re.test(l.text)));
  if (formLine) {
    const formHit = MEDICINE_FORMS.find((f) => f.re.test(formLine.text));
    if (formHit) {
      const hit = formLine.text.match(formHit.re);
      out.form = makeField('form', formHit.form, hit ? hit[0] : formHit.form, 0.85, formLine.text);
    }
  }

  const dose = text.match(DOSE_INSTRUCTION_RE);
  if (dose) {
    const words = { one: 1, two: 2, three: 3, four: 4 };
    const count = words[dose[1].toLowerCase()] || Number(dose[1]);
    out.dose = makeField('dose', `${count} ${normaliseForm(dose[2])}`, dose[0], 0.85, lineAt(lines, dose.index));
  }

  const freq = FREQUENCY_PHRASES.map((f) => ({ f, m: f.re.exec(text) })).filter((x) => x.m)
    .sort((a, b) => a.m.index - b.m.index)[0];
  if (freq) {
    let phrase = freq.f.phrase;
    if (freq.f.capture && freq.m[freq.f.capture]) phrase = phrase.replace('N', freq.m[freq.f.capture]);
    out.frequency = makeField('frequency', phrase, freq.m[0], freq.f.confidence, lineAt(lines, freq.m.index));
  }

  const routeLine = lines.find((l) => ROUTE_RE.test(l.text));
  if (routeLine) {
    const hit = routeLine.text.match(ROUTE_RE);
    out.route = makeField('route', hit[0].toLowerCase(), hit[0], 0.7, routeLine.text);
  }

  // `x.m[1] != null && (x.m[2] != null || x.d.unit)` -- guarding only on the MATCH is what
  // let an undefined group through to .toLowerCase() and kill the scan. A pattern whose
  // groups did not fire is simply not a usable duration.
  const dur = DURATION_PHRASES.map((d) => ({ d, m: d.re.exec(text) }))
    .filter((x) => x.m && x.m[1] != null && (x.m[2] != null || x.d.unit))
    .sort((a, b) => a.m.index - b.m.index)[0];
  if (dur) {
    const count = Number(dur.m[1]);
    const unit = String(dur.m[2] != null ? dur.m[2] : dur.d.unit).toLowerCase().replace(/s$/, '');
    out.duration = makeField('duration', `${count} ${unit}${count === 1 ? '' : 's'}`, dur.m[0], 0.85, lineAt(lines, dur.m.index));
  }

  const qty = text.match(/\bquantity\s*[:.]?\s*(\d{1,5})\b/i);
  if (qty) out.quantity = makeField('quantity', qty[1], qty[0], 0.85, lineAt(lines, qty.index));
  const repeats = text.match(/\brepeat(?:s|ed)?\s*[:.]?\s*(\d{1,3})\b/i);
  if (repeats) out.repeats = makeField('repeats', repeats[1], repeats[0], 0.85, lineAt(lines, repeats.index));

  const issued = dateOn(dates, lines, ISSUED_RE);
  if (issued) out.issued = makeField('issued', issued.iso, issued.raw, issued.confidence, lineAt(lines, issued.index));
  const until = dateOn(dates, lines, VALID_UNTIL_RE);
  if (until) out.valid_until = makeField('valid_until', until.iso, until.raw, until.confidence, lineAt(lines, until.index));

  const warnLine = lines.find((l) => WARNING_RE.test(l.text));
  if (warnLine) out.warning = makeField('warning', warnLine.text, warnLine.text, 0.7, warnLine.text);

  return out;
}

/**
 * Map recognised text onto the labelled fields of its document type.
 *
 * Returns `{fieldKey: record}` — only the fields that were FOUND. An unrecognised
 * document has no field vocabulary, so the map is empty (its transcript is still
 * shown; the app simply claims no fields).
 */
export function extractFields(text, opts) {
  const o = opts || {};
  const src = String(text || '');
  const docType = o.docType || classifyDocument(src);
  const ctx = {
    text: src,
    lines: splitLines(src),
    money: o.money || extractMoney(src),
    dates: o.dates || extractDates(src, { now: o.now }),
  };
  if (docType.type === 'receipt') return extractReceiptFields(ctx);
  if (docType.type === 'prescription') return extractPrescriptionFields(ctx);
  return {};
}

/**
 * Arrange found fields into the document's section blocks, in importance order.
 * A group with no found field is dropped, so "Personal details" can never render
 * as an empty heading (rule 2 at the top of js/taxonomy.js).
 */
export function groupFields(fields, type) {
  const found = fields || {};
  return groupsForType(type)
    .map((g) => ({ id: g.id, label: g.label, fields: g.fields.map((k) => found[k]).filter(Boolean) }))
    .filter((g) => g.fields.length > 0);
}

// ---------------------------------------------------------------------------
// THE SPEAK-OR-OMIT GATE — §6.4
//   "a field is spoken only if confidence >= CONFIDENCE_MIN (default 0.75) AND it
//    validates (checksum/format/date sanity). Otherwise the summary says that
//    field was unclear."
// ---------------------------------------------------------------------------

/**
 * Decide, for one candidate, whether it may be spoken.
 * @returns {{speak:boolean, reason:string, confidence:number}}
 */
export function gateField(field, opts) {
  const o = opts || {};
  const min = o.confidenceMin !== undefined ? o.confidenceMin : CONFIG.confidenceMin;
  if (!field) return { speak: false, reason: 'no field', confidence: 0 };
  const conf = typeof field.confidence === 'number' ? field.confidence : 0;

  if (!field.valid) {
    return { speak: false, reason: `failed validation: ${field.reason || 'not valid'}`, confidence: conf };
  }
  if (field.ambiguous) {
    return { speak: false, reason: `ambiguous: ${field.reason || 'two readings are equally plausible'}`, confidence: conf };
  }
  if (conf < min) {
    return { speak: false, reason: `confidence ${conf} < CONFIDENCE_MIN ${min}`, confidence: conf };
  }
  return { speak: true, reason: `confidence ${conf} >= ${min} and the field validates`, confidence: conf };
}

/**
 * Run the gate over every candidate group.
 *
 * The important part is `spoken` vs `omitted`: `omitted` records *what kind of
 * thing* was unclear so the summary can say "the amount is unclear" without ever
 * quoting the value.
 */
export function applySpeakGate(fields, opts) {
  const o = opts || {};
  const min = o.confidenceMin !== undefined ? o.confidenceMin : CONFIG.confidenceMin;
  const groups = {
    money: fields.money || [],
    date: fields.dates || [],
    dosage: fields.dosage || [],
    account: fields.accounts || [],
  };

  const spoken = [];
  const omitted = [];
  for (const [kind, list] of Object.entries(groups)) {
    let spokeOne = false;
    list.forEach((field, i) => {
      const verdict = gateField(field, { confidenceMin: min });
      const entry = { kind, field, verdict, rank: i };
      // One confident value per kind: reading three amounts aloud is worse than
      // reading the one that matters (§6.5 caps the summary at 4 clauses).
      if (verdict.speak && !spokeOne) {
        spokeOne = true;
        spoken.push(entry);
      } else if (!verdict.speak) {
        omitted.push(entry);
      }
    });
    if (!spokeOne && list.length === 0) {
      omitted.push({ kind, field: null, verdict: { speak: false, reason: 'no candidate found', confidence: 0 }, rank: 0 });
    }
  }
  return { spoken, omitted, confidenceMin: min };
}

// ---------------------------------------------------------------------------
// §6.5 spoken summary — deterministic template, no LLM
// ---------------------------------------------------------------------------

const CURRENCY_NAMES = {
  USD: { major: 'dollars', minor: 'cents', singular: 'dollar' },
  GBP: { major: 'pounds', minor: 'pence', singular: 'pound' },
  EUR: { major: 'euros', minor: 'cents', singular: 'euro' },
  AUD: { major: 'Australian dollars', minor: 'cents', singular: 'Australian dollar' },
  CAD: { major: 'Canadian dollars', minor: 'cents', singular: 'Canadian dollar' },
  NZD: { major: 'New Zealand dollars', minor: 'cents', singular: 'New Zealand dollar' },
  JPY: { major: 'yen', minor: null, singular: 'yen' },
  INR: { major: 'rupees', minor: 'paise', singular: 'rupee' },
  CHF: { major: 'Swiss francs', minor: 'centimes', singular: 'Swiss franc' },
  SGD: { major: 'Singapore dollars', minor: 'cents', singular: 'Singapore dollar' },
  HKD: { major: 'Hong Kong dollars', minor: 'cents', singular: 'Hong Kong dollar' },
};

/**
 * Format an amount for speech.
 *
 * TTS engines read a bare "£1,234.56" inconsistently (some say "pound one two
 * three four point five six"), so the number is always spelled out with its
 * currency name. Where the currency is unknown the number is still spoken, but
 * without a name — the gate is what decides whether it gets that far.
 */
export function speakableMoney(field) {
  const sign = field.negative || field.value < 0 ? 'minus ' : '';
  const abs = Math.abs(field.value);
  const major = Math.floor(abs);
  const minor = Math.round((abs - major) * 100);
  const name = field.currency ? CURRENCY_NAMES[field.currency] : null;

  // No currency to name: read the digits. A whole amount drops the ".00" —
  // "minus 850" is what a person says; "minus 850 point zero zero" is not.
  if (!name) return `${sign}${Number.isInteger(abs) ? String(abs) : abs.toFixed(2)}`;
  const majorWord = major === 1 ? name.singular : name.major;
  if (!minor) return `${sign}${major} ${majorWord}`;
  const minorWord = name.minor ? ` and ${minor} ${name.minor}` : '';
  return `${sign}${major} ${majorWord}${minorWord}`;
}

/** Format a date for speech without ambiguity: "15 March 2024". */
export function speakableDate(field) {
  if (!field.iso) return field.raw;
  const [y, m, d] = field.iso.split('-').map(Number);
  const names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  return `${d} ${names[m - 1]} ${y}`;
}

/** Format a dosage for speech: "500 milligrams, twice daily, for 7 days". */
export function speakableDosage(field) {
  const parts = [`${field.amount}${field.unit === '%' ? ' percent' : ` ${spokenUnit(field.unit)}`}`];
  if (field.frequency) parts.push(field.frequency);
  if (field.duration) parts.push(`for ${field.duration.count} ${field.duration.unit}${field.duration.count === 1 ? '' : 's'}`);
  return parts.join(', ');
}

function spokenUnit(unit) {
  const map = {
    mg: 'milligrams', mcg: 'micrograms', g: 'grams', ml: 'millilitres', l: 'litres',
    IU: 'international units', units: 'units', tablet: 'tablets', capsule: 'capsules',
    drop: 'drops', puff: 'puffs', spray: 'sprays', patch: 'patches', suppository: 'suppositories',
  };
  return map[unit] || unit;
}

// Plain words, every sentence <= 15 words (docs/UX_ELDER.md §3). The fallback
// says it could not tell which — it never guesses a type (this round's brief §3.1).
const DOC_TYPE_SPOKEN = {
  receipt: 'a shop receipt',
  prescription: 'a prescription',
};

// The fallback needs its own sentence, not a noun phrase: "This looks like not a
// receipt or a prescription." is not English.
const UNRECOGNISED_SPOKEN = 'I could not tell if this is a receipt or a prescription.';

/**
 * Build the spoken summary (§6.5).
 *
 *   "This looks like a <type>. <fields, one clause each.> The full text is on screen."
 *
 * It is PHYSICALLY unable to speak an ungated value: it reads `gated.spoken`
 * only, and every clause is generated from a field that already passed
 * `gateField()`. The unit test for "never speak a number we are not confident
 * about" asserts exactly that property.
 *
 * @param {{docType:object, gated:object, ocr:object, preprocess?:object}} input
 *   `preprocess.contrastStretch` (the 2nd/98th percentile luma of the page) is the
 *   ONLY evidence that may justify naming light. See isDarkPage() below.
 */
export function buildSpokenSummary(input) {
  const cfg = CONFIG;
  const o = input || {};
  const docType = o.docType || { type: UNRECOGNISED_TYPE, confidence: 0, reason: '' };
  const gated = o.gated || { spoken: [], omitted: [] };
  const ocr = o.ocr || {};

  const lowQuality = typeof ocr.mean_confidence === 'number'
    && ocr.mean_confidence < cfg.summary.lowQualityMeanConfidence;
  const clauses = [];
  const byKind = { money: null, date: null, dosage: null, account: null };
  for (const entry of lowQuality ? [] : gated.spoken) {
    if (!byKind[entry.kind]) byKind[entry.kind] = entry;
  }

  // §6.5: "prefer amount, date, and the single most important field per type".
  const order = SUMMARY_ORDER[docType.type] || SUMMARY_ORDER.general;
  const omittedKinds = new Set(gated.omitted.map((e) => e.kind));

  for (const kind of order) {
    if (clauses.length >= cfg.summary.maxClauses) break;
    const entry = byKind[kind];
    if (!entry) continue;
    clauses.push(phraseFor(kind, entry.field));
  }

  // §6.4: "Otherwise the summary says that field was unclear." Name the kind, and
  // never the value — this is the sentence that replaces a confident wrong number.
  if (clauses.length < cfg.summary.maxClauses) {
    for (const kind of order) {
      if (clauses.length >= cfg.summary.maxClauses) break;
      if (!omittedKinds.has(kind)) continue;
      if (byKind[kind]) continue;
      clauses.push(`the ${FIELD_NOUN[kind]} is unclear`);
      omittedKinds.delete(kind);
    }
  }

  // Low-quality recognition never contributes a spoken value.
  const noText = !ocr.char_count;
  // Did the app actually MEASURE a dark page? Only then may the advice name
  // light; otherwise it must not, because nothing measured the light.
  const dark = isDarkPage(o.preprocess, cfg);

  const sentences = [docType.type === UNRECOGNISED_TYPE
    ? UNRECOGNISED_SPOKEN
    : `This looks like ${DOC_TYPE_SPOKEN[docType.type] || DOC_TYPE_SPOKEN.receipt}.`];
  if (noText) {
    sentences.push(dark ? cfg.spoken.noTextDark : cfg.spoken.noText);
  } else {
    if (lowQuality) sentences.push(dark ? cfg.spoken.hardToReadDark : cfg.spoken.hardToRead);
    if (clauses.length) sentences.push(`${clauses.map(capitaliseFirst).join('. ')}.`);
    sentences.push('The full text is on screen.');
  }

  return {
    text: sentences.join(' ').replace(/\s+/g, ' ').trim(),
    clauses,
    docType: docType.type,
    omittedKinds: [...new Set(gated.omitted.map((e) => e.kind))],
    lowQuality,
    noText,
    dark,
  };
}

/**
 * Did preprocessing MEASURE a dark page?
 *
 * `contrastStretch` records the 2nd and 98th percentile luma of the downscaled
 * page. A well-lit white page's 98th percentile sits near 230-255; a dim photo's
 * brightest parts are mid-grey. The advice may say "brighter light" only when
 * this returns true (docs/DECISIONS.md D29) — otherwise the app would be guessing
 * at a cause it never measured, which BRIEF.md §6.4 forbids.
 *
 * @param {object|null} preprocess  the preprocess record (or null)
 * @param {object} cfg
 * @returns {boolean}
 */
export function isDarkPage(preprocess, cfg) {
  const c = cfg || CONFIG;
  const cs = preprocess && preprocess.contrastStretch;
  if (!cs || typeof cs.high !== 'number' || !isFinite(cs.high)) return false;
  return cs.high < c.summary.darkBrightnessBelow;
}

const SUMMARY_ORDER = {
  receipt: ['money', 'date', 'account'],
  prescription: ['dosage', 'date', 'money', 'account'],
  [UNRECOGNISED_TYPE]: ['date', 'money', 'account'],
};

const FIELD_NOUN = {
  money: 'amount',
  date: 'date',
  dosage: 'dosage',
  account: 'account number',
};

function phraseFor(kind, field) {
  if (kind === 'money') return `the amount is ${speakableMoney(field)}`;
  if (kind === 'date') return `the date is ${speakableDate(field)}`;
  if (kind === 'dosage') return `the dosage is ${speakableDosage(field)}`;
  if (kind === 'account') return `the account number is ${field.speakAs}`;
  return '';
}

function capitaliseFirst(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

// ---------------------------------------------------------------------------
// Top-level convenience: text -> everything
// ---------------------------------------------------------------------------

/**
 * Extract every field type from recognised text and apply the gate.
 * @param {string} text
 * @param {{now?:Date, confidenceMin?:number}} [opts]
 */
export function extractAll(text, opts) {
  const o = opts || {};
  const src = String(text || '');
  const money = extractMoney(src);
  const dates = extractDates(src, { now: o.now });
  const dosage = extractDosage(src);
  const accounts = extractAccounts(src);
  const docType = classifyDocument(src);
  const fields = extractFields(src, { docType, money, dates });
  // Pattern validity does not establish recognition reliability. When a caller
  // supplies OCR evidence, cap every candidate before grouping or speech gating.
  // Unknown recognition confidence fails closed; text-only callers keep their
  // existing extraction contract (benchmarks can still measure recognition).
  if (o.ocr) {
    const measured = o.ocr.mean_confidence;
    const cap = Number.isFinite(measured) ? clamp01(measured / 100) : 0;
    for (const f of [...money, ...dates, ...dosage, ...accounts, ...Object.values(fields)]) {
      f.confidence = Math.min(f.confidence, cap);
      // Available word boxes provide stronger local evidence than a page mean.
      const source = String(f.line || f.raw || '').toLowerCase();
      for (const box of o.ocr.boxes || []) {
        const token = String(box.text || '').trim().toLowerCase();
        if (token && source.includes(token) && Number.isFinite(box.confidence)) {
          f.confidence = Math.min(f.confidence, clamp01(box.confidence / 100));
        }
      }
    }
  }
  const gated = applySpeakGate({ money, dates, dosage, accounts }, o);
  return {
    money, dates, dosage, accounts, docType, gated,
    // The document's own fields, and those arranged into their section blocks.
    // Both are empty for an `unrecognised` document — it has no vocabulary.
    fields,
    groups: groupFields(fields, docType.type),
    counts: {
      money: money.length, dates: dates.length, dosage: dosage.length,
      accounts: accounts.length,
      fields: Object.keys(fields).length,
    },
  };
}

export default {
  countAlnum,
  extractMoney,
  parseAmountToken,
  extractDates,
  parseRelativeDate,
  expandYear,
  isRealDate,
  extractDosage,
  extractAccounts,
  luhnCheck,
  looksDegenerate,
  maskAccountForSpeech,
  maskAccountForDisplay,
  classifyDocument,
  extractFields,
  groupFields,
  gateField,
  applySpeakGate,
  buildSpokenSummary,
  isDarkPage,
  speakableMoney,
  speakableDate,
  speakableDosage,
  extractAll,
  DOC_TYPES,
};
