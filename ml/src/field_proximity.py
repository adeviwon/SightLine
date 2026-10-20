"""
Lane H — keyword-proximity value extraction + PER-TYPE VALIDATION.

WHY THIS EXISTS
---------------
Published field-extraction systems do not transcribe-then-regex. They locate a
field by its LABEL (keyword proximity: direction + max distance), read the value
region beside it, and then VALIDATE BY FIELD TYPE. Validation is the half that
lets the product refuse instead of speaking a wrong dosage: our product rule is
"a value is only spoken if confidence >= 0.75 AND it validates".

This module is the validation half plus the proximity read.

WHAT IS HERE
------------
  validate(kind, value) -> (ok, reason)         every rule returns a REASON,
                                                because the reason is what the
                                                product surfaces INSTEAD of the
                                                value.
  value_after(line_text, span) -> str           text right of a label on the
                                                same line, leading separators
                                                trimmed.
  extract_fields(lines, spots) -> dict          label spots -> validated fields,
                                                with cross-line fallback and
                                                provenance.

KINDS come from the product, not invented here: evaluate.FIELDS
(500mg/400mg dosage, 40218877 account, 40-11-04 sort code, 2500 money,
2024-CV-00456 identifier) and the PATTERNS table in app/js/pipeline.js.

`spots` is Lane G's output and is taken as INPUT (never imported), so this
module can be tested with literal dicts:
    {"label": "Sort Code", "start": 0, "end": 9, "score": 0.9, "line_index": 1}

DESIGN RULES PINNED BY THE TESTS
--------------------------------
  * THE VALIDATOR RULE. A value the app can legitimately READ must never be
    refused for its SHAPE. Refusal is reserved for a value that cannot be real
    (an impossible calendar day, a 2-character case reference) or that
    contradicts evidence (the two extraction paths disagreeing). The app's own
    regex layer (pipeline.js PATTERNS) reads long-form dates — '15 January
    2024', '1 February 2024' — and any validator that turned one of those into
    a refusal would be a correctness regression hiding inside a safety feature.
  * Deterministic. Spots are sorted, ties break on (line_index, start, end,
    label), so a shuffled spot list gives the same dict.
  * One value region is claimed once. Two labels cannot both consume the same
    region.
  * A label with nothing usable beside it is reported, not dropped: ok False,
    provenance "none".
"""

from __future__ import annotations

import argparse
import datetime
import re

# ── field kinds ───────────────────────────────────────────────────────────────

UNITS = {"mg", "mcg", "g", "ml", "tablet", "tablets", "capsule", "capsules",
         "drop", "drops", "unit", "units"}

# Dosing frequency phrases (the product's Frequency pattern + the abbreviations
# a British prescription writes).
FREQ_RE = re.compile(
    r"(?:once|twice|three|four|five|six|\d+)\s*times?\s*(?:a\s*|per\s*)?"
    r"(?:day|daily|week|weekly)"
    r"|every\s*\d+(?:\s*(?:to|-)\s*\d+)?\s*(?:hours?|hrs?|h)\b"
    r"|\b(?:bd|b\.d\.|tds|t\.d\.s\.|qds|q\.d\.s\.|prn|p\.r\.n\.|"
    r"nocte|mane|od|qid|tid|bid)\b"
    r"|at\s+bedtime|as\s+(?:required|needed)|in\s+the\s+morning",
    re.I,
)

_CURRENCY = r"(?:[£$€]|GBP|USD|EUR)"
_THOUSANDS_OK = re.compile(r"\d{1,3}(?:,\d{3})+$")

# Separator characters trimmed to the LEFT of a value: punctuation joining a
# label to its value, never a digit and never a real minus sign (see _trim_index).
_SEP = " \t:#=|.,·—"


def _first_token(s: str) -> str:
    parts = s.split()
    return parts[0] if parts else ""


def _has_digit(s: str) -> bool:
    return bool(re.search(r"\d", s))


# ── validators, one per kind ──────────────────────────────────────────────────

def _validate_sort_code(v: str):
    """Exactly 6 digits: '40-11-04', '40 11 04', '401104'.

    Takes the LEADING run of digits/spaces/hyphens, not the first token: for a
    space-separated sort code the space IS the separator, so '40 11 04' would
    otherwise be read as the single digit-run '40'.
    """
    s = v.strip()
    cut = len(s)
    for i, c in enumerate(s):
        if not (c.isdigit() or c in " -"):
            cut = i
            break
    cleaned = re.sub(r"[-\s]", "", s[:cut].strip(" -"))
    if re.fullmatch(r"\d{6}", cleaned):
        return True, "ok"
    return False, "6 digits required, got %d" % len(re.sub(r"\D", "", cleaned))


def _validate_account(v: str):
    """4-12 digits. A value masked to its last 4 digits is ACCEPTED and flagged."""
    s = v.strip()
    tokens = s.split() or [""]
    cand = next((t for t in tokens if _has_digit(t)), "")
    digits = re.sub(r"\D", "", cand)
    masked = (len(digits) == 4 or bool(re.search(r"[*xX#]", s))
              or "ending" in s.lower())
    if not 4 <= len(digits) <= 12:
        return False, "4-12 digits required, got %d" % len(digits)
    if masked:
        return True, "masked, last 4 digits only"
    return True, "ok"


_DMY = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})$")
_YMD = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")

# Long-form month-name dates: '15 January 2024', '3 Mar 2026', '15 Jan 24',
# '15th January, 2024'. ASCII month names only (no locale handling). The day may
# carry an English ordinal suffix because pipeline.js's own legal Date pattern
# emits '\d{1,2}(?:st|nd|rd|th)?'. Covers full and abbreviated month names, an
# optional comma before the year, and a 2- or 4-digit year, case-insensitively.
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_DMONTH = re.compile(
    r"^(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?\s*,?\s*(\d{2}|\d{4})$",
    re.I,
)


def _validate_date(v: str):
    """A real calendar date in DD/MM/YYYY, DD/MM/YY, YYYY-MM-DD, DD-MM-YYYY or
    a long-form month-name date ('15 January 2024', '3 Mar 2026', '15 Jan 24')."""
    s = v.strip()
    tok = _first_token(s)
    m = _DMY.match(tok)
    if m:
        d, mo, y = (int(x) for x in m.groups())
    else:
        m = _YMD.match(tok)
        if m:
            y, mo, d = (int(x) for x in m.groups())
        else:
            # Long-form month-name dates: the whole value, not the first token,
            # because the value is multi-token. The app's own Date pattern reads
            # these, so refusing them would refuse a value it reads correctly.
            m = _DMONTH.match(s)
            if m:
                mo = _MONTHS.get(m.group(2).lower())
                if mo is None:                       # e.g. '15 Foo 2024'
                    return False, "not a calendar date"
                d, y = int(m.group(1)), int(m.group(3))
            elif not re.fullmatch(r"\d{8}", tok):
                # SEPARATOR-FREE DATES, and this is not a nicety: the
                # recogniser's output alphabet is 0-9 A-Z and space — it CANNOT
                # emit '/' or '-'. Every date this product reads arrives as 8
                # digits (04021979 for 04/02/1979), so refusing them refuses
                # every date the model is physically able to produce.
                # Reconstruct, then validate the same way.
                return False, "not a calendar date"
            else:
                d, mo, y = int(tok[:2]), int(tok[2:4]), int(tok[4:])
                if d > 31 or mo > 12:                # DDMMYYYY first (HK/UK), then
                    d, mo, y = int(tok[6:]), int(tok[4:6]), int(tok[:4])  # YYYYMMDD
    if y < 100:
        y += 2000
    try:
        datetime.date(y, mo, d)
    except ValueError:
        return False, "not a calendar date"
    return True, "ok"


def _validate_money(v: str):
    """Optional currency symbol/pair, thousands separators, optional sign, and
    exactly two decimals when a decimal point is present."""
    s = v.strip()
    prev = None
    while s != prev:                       # currency may sit either side
        prev = s
        s = re.sub(r"^\s*%s\s*" % _CURRENCY, "", s, flags=re.I)
        s = re.sub(r"\s*%s\s*$" % _CURRENCY, "", s, flags=re.I)
    m = re.fullmatch(r"([+-]?)\s*([\d,]+(?:\.\d+)?)", s)
    if not m:
        return False, "not a monetary amount: %s" % v.strip()
    num = m.group(2)
    if "." in num:
        dec = num.split(".")[1]
        if len(dec) != 2:
            return False, "money needs two decimals, got %d" % len(dec)
    if "," in num and not _THOUSANDS_OK.match(num.split(".")[0]):
        return False, "bad thousands grouping: %s" % v.strip()
    return True, "ok"


def _validate_dosage(v: str):
    """A number followed by a KNOWN unit, optionally a frequency."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)", v)
    if not m:
        return False, "no dosage in value: %s" % v.strip()
    unit = m.group(2).lower()
    if unit not in UNITS:
        return False, "unknown unit: %s" % m.group(2)
    rest = v[m.end():].strip(" .,;")
    if rest and rest.lower() not in UNITS and not FREQ_RE.search(rest):
        return False, "unknown frequency: %s" % rest
    return True, "ok"


def _validate_identifier(v: str):
    """Case references like 2024-CV-00456: letters, digits, hyphens, 4-20 chars."""
    tok = _first_token(v.strip())
    bad = re.search(r"[^A-Za-z0-9-]", tok)
    if bad:
        return False, "invalid character in identifier: %s" % bad.group(0)
    if not 4 <= len(tok) <= 20:
        return False, "identifier must be 4-20 chars, got %d" % len(tok)
    if not re.search(r"[A-Za-z0-9]", tok):
        return False, "identifier needs a letter or digit"
    return True, "ok"


def _validate_general(v: str):
    """Any non-empty string under 40 chars with at least one alphanumeric."""
    s = v.strip()
    if not s:
        return False, "empty value"
    if len(s) >= 40:
        return False, "too long: %d chars" % len(s)
    if not re.search(r"[A-Za-z0-9]", s):
        return False, "no alphanumeric"
    return True, "ok"


VALIDATORS = {
    "sort_code": _validate_sort_code,
    "account": _validate_account,
    "date": _validate_date,
    "money": _validate_money,
    "dosage": _validate_dosage,
    "identifier": _validate_identifier,
    "general": _validate_general,
}


def validate(kind: str, value: str):
    """(ok, reason) for `value` under `kind`. Never raises on bad input."""
    fn = VALIDATORS.get(str(kind))
    if fn is None:
        return False, "unknown kind: %s" % kind
    if value is None:
        return False, "empty value"
    return fn(str(value))


# ── proximity read ────────────────────────────────────────────────────────────

def _trim_index(s: str, i: int, limit=None) -> int:
    """First index >= i that is not a leading separator.

    A '-' is a separator only when it does NOT start a digit: 'Sort Code: 40-11-04'
    trims the colon and space, while a value written '-1,100.00' keeps its sign.
    """
    n = len(s) if limit is None else min(limit, len(s))
    while i < n:
        c = s[i]
        if c in _SEP:
            i += 1
        elif c == "-" and not (i + 1 < n and s[i + 1].isdigit()):
            i += 1
        else:
            break
    return i


def value_after(line_text: str, span):
    """Text to the RIGHT of a label span on the same line, leading separators
    trimmed ('Sort Code: 40-11-04' + span of 'Sort Code' -> '40-11-04')."""
    text = "" if line_text is None else str(line_text)
    end = int(span[1])
    return text[_trim_index(text, end):].rstrip()


# ── label -> (field name, kind) ───────────────────────────────────────────────
#
# Ordered most-specific-first; the first needle that matches as a whole word
# wins. Derived from pipeline.js PATTERNS and capture.py's own document labels.

FIELD_LABELS = [
    ("sort code", "sort_code", "sort_code"),
    ("account number", "account_number", "account"),
    ("account no", "account_number", "account"),
    ("card ending", "card_ending", "account"),
    ("ending in", "card_ending", "account"),
    ("ending", "card_ending", "account"),
    ("account holder", "account_holder", "general"),
    ("opening balance", "opening_balance", "money"),
    ("closing balance", "closing_balance", "money"),
    ("available funds", "available_funds", "money"),
    ("balance", "balance", "money"),
    ("amount", "amount", "money"),
    ("date of birth", "date_of_birth", "date"),
    ("dob", "date_of_birth", "date"),
    ("date", "date", "date"),
    ("case reference", "case_number", "identifier"),
    ("case number", "case_number", "identifier"),
    ("case ref", "case_number", "identifier"),
    ("case no", "case_number", "identifier"),
    ("matter", "case_number", "identifier"),
    ("iban", "iban", "general"),
    ("dosage", "dosage", "dosage"),
    ("dose", "dosage", "dosage"),
    ("prescription for", "medication", "general"),
    ("medication", "medication", "general"),
    ("rx", "medication", "general"),
    ("frequency", "frequency", "general"),
    ("patient", "patient", "general"),
    ("name", "patient", "general"),
    ("warning", "warning", "general"),
    ("caution", "warning", "general"),
    ("clause", "clause", "general"),
    ("section", "clause", "general"),
    ("phone", "phone", "general"),
    ("email", "email", "general"),
    ("time", "time", "general"),
    ("period", "period", "general"),
    ("property", "property", "general"),
    ("term", "term", "general"),
]

_LABEL_RES = [(re.compile(r"(?<![a-z])%s(?![a-z])" % re.escape(n)), name, kind)
              for n, name, kind in FIELD_LABELS]


def resolve_label(label: str):
    """(field_name, kind) for a spotted label; unknown labels read as general."""
    low = str(label or "").strip().lower()
    for rx, name, kind in _LABEL_RES:
        if rx.search(low):
            return name, kind
    return re.sub(r"[^a-z0-9]+", "_", low).strip("_") or "field", "general"


# ── extraction ────────────────────────────────────────────────────────────────

def _next_line_text(lines, li, box):
    """The value line just below a label's line, or None.

    "just below" = smallest box.y0 at or under the label line's y1, tie-broken by
    x0 then index, so two stacked lines never flip between runs.
    """
    if box is not None:
        y1 = box[3]
        cands = []
        for j, ln in enumerate(lines):
            if j == li:
                continue
            b = ln.get("box")
            if b and b[1] >= y1 and str(ln.get("text", "")).strip():
                cands.append((b[1], b[0], j))
        if cands:
            j = min(cands)[2]
            return j, str(lines[j]["text"])
        return None
    j = li + 1
    if 0 <= j < len(lines) and str(lines[j].get("text", "")).strip():
        return j, str(lines[j]["text"])
    return None


def extract_fields(lines, spots):
    """{field_name: {value, ok, kind, reason, provenance}} from label spots."""
    lines = list(lines or [])
    ordered = sorted(
        (s for s in (spots or []) if isinstance(s, dict)),
        key=lambda s: (int(s.get("line_index", 0)), int(s.get("start", 0)),
                       int(s.get("end", 0)), str(s.get("label", ""))),
    )

    out = {}
    claimed = set()          # (line_index, value_start) already spoken for
    for i, sp in enumerate(ordered):
        li = int(sp.get("line_index", 0))
        if not 0 <= li < len(lines):
            continue
        name, kind = resolve_label(sp.get("label", ""))
        if name in out:                       # dict is keyed by name: first wins
            continue

        text = str(lines[li].get("text", ""))
        end = int(sp.get("end", 0))
        nxt = len(text)
        for sp2 in ordered[i + 1:]:
            if int(sp2.get("line_index", 0)) == li and int(sp2.get("start", 0)) > end:
                nxt = int(sp2.get("start", 0))
                break

        j = _trim_index(text, end, nxt)
        value = text[j:nxt].rstrip()
        provenance = "same-line-right"
        region = (li, j)

        if not value:
            found = _next_line_text(lines, li, lines[li].get("box"))
            if found:
                nli, ntext = found
                k = _trim_index(ntext, 0)
                value, provenance, region = ntext[k:].rstrip(), "next-line", (nli, k)
            else:
                out[name] = {"value": "", "ok": False, "kind": kind,
                             "reason": "no value found after label",
                             "provenance": "none"}
                continue

        if region in claimed:                 # one value, one field. no dupes.
            continue
        claimed.add(region)

        ok, reason = validate(kind, value)
        out[name] = {"value": value, "ok": ok, "kind": kind, "reason": reason,
                     "provenance": provenance}
    return out


# ── self-test ─────────────────────────────────────────────────────────────────

def _case(name, rows, labels):
    """rows: [(text, x0, y0)]; labels: [(line_index, label)]."""
    lines = [{"text": t, "box": (x0, y, x0 + 300, y + 20)} for t, x0, y in rows]
    spots = []
    for li, lab in labels:
        t = rows[li][0]
        i = t.find(lab)
        spots.append({"label": lab, "start": i, "end": i + len(lab), "score": 0.9,
                      "line_index": li})
    return name, lines, spots


def _one(name, text, label):
    return _case(name, [(text, 40, 20)], [(0, label)])


CASES = [
    _one("sort_code accept 40-11-04", "Sort Code: 40-11-04", "Sort Code"),
    _one("sort_code accept 401104", "Sort Code: 401104", "Sort Code"),
    _one("sort_code reject 5 digits", "Sort Code: 40-11-0", "Sort Code"),
    _one("account accept 40218877", "Account Number: 40218877", "Account Number"),
    _one("account accept masked", "Account Number: **** 4521", "Account Number"),
    _one("account reject 3 digits", "Account Number: 123", "Account Number"),
    _one("date accept 01/10/2026", "Date: 01/10/2026", "Date"),
    _one("date reject 31/02/2026", "Date: 31/02/2026", "Date"),
    _one("money accept GBP 1,204.33", "Opening Balance        GBP 1,204.33",
         "Opening Balance"),
    _one("money reject one decimal", "Closing Balance: GBP 12.4", "Closing Balance"),
    _one("dosage accept 500mg", "Dosage: 500mg", "Dosage"),
    _one("dosage reject unknown unit", "Dosage: 500xyz", "Dosage"),
    _one("identifier accept case ref", "Case Reference: 2024-CV-00456",
         "Case Reference"),
    _one("identifier reject too short", "Case Reference: 20", "Case Reference"),
    _one("general accept warning", "Warning: may cause drowsiness", "Warning"),
    _one("no value after label", "Sort Code:", "Sort Code"),
    _case("next-line value",
          [("Account Number:", 40, 20), ("40218877", 40, 50)],
          [(0, "Account Number")]),
    _case("two labels, one line",
          [("Account Number: 40218877   Sort Code: 40-11-04", 40, 20)],
          [(0, "Account Number"), (0, "Sort Code")]),
]


def _selftest():
    print("%-32s %-12s %-22s %-5s %-32s %s"
          % ("case", "field", "value", "ok", "reason", "provenance"))
    print("-" * 132)
    rows = 0
    for name, lines, spots in CASES:
        got = extract_fields(lines, spots)
        for field, r in got.items():
            rows += 1
            print("%-32s %-12s %-22s %-5s %-32s %s"
                  % (name, field, r["value"][:22] or "(none)", str(r["ok"]),
                     r["reason"][:32], r["provenance"]))
    assert len(CASES) >= 12, "need >= 12 self-test cases, have %d" % len(CASES)
    assert rows >= len(CASES), "every case must produce a field"
    print("-" * 132)
    print("field_proximity self-test ok: %d cases, %d fields" % (len(CASES), rows))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="keyword-proximity field extraction with per-type validation")
    ap.add_argument("--selftest", action="store_true",
                    help="print extracted value + ok + reason + provenance")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    ap.print_usage()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
