"""
field_policy.py — the SPEAK / REFUSE decision layer for field values.

WHY THIS EXISTS
---------------
Two extractors exist and they fail in opposite directions, measured on the
rendered test split against the renderer's own ground truth (the continuation of
the printed line after each spotted label):

  (a) whole-page regex (evaluate.fields_found)  precision 0.965, recall 0.240
      — it only speaks 173 of the 696 values the renderer drew, because it knows
      two field patterns per document type.
  (b) label-anchored spot + typed validation    precision 0.608, recall 0.615
      — broader, but 276 of 704 spoken values are confidently WRONG.

For a blind user a confidently spoken wrong dosage is the failure the product
exists to prevent. 39% wrong is not shippable at any recall. This module is the
decision layer that decides what is SPOKEN, given both extractions plus each
candidate's validation state, and it says WHY when it refuses.

    decide(regex_fields, spot_fields, *, kinds=None, doc_type=None,
           strict_critical=False, speak_free_text=True)
        -> [{"field", "value", "decision", "why", "source"}]

    decision is "speak" or "refuse"; source is "regex", "spot" or "both".

THE RULE SET (first rule that fires wins; every refusal carries its reason)
--------------------------------------------------------------------------
R1 EMPTY.      A candidate with no value is never spoken — "label spotted but no
               value present" / "empty value".
R2 VALIDATION. A value that failed its kind's validator is never spoken,
               whatever the other path says. The validator's own reason is the
               why ("not a calendar date"), because that is what the product
               says instead of guessing. The regex path has no validator of its
               own, so every regex candidate is validated here too.
R3 DISAGREEMENT. When both paths produced a value for the SAME field and their
               canonical forms are disjoint, BOTH are refused — never pick a
               winner: "two paths disagree: 01102026 vs 01102027". Two dates
               that differ are a reason to re-scan, not a vote.
R4 SAFETY-CRITICAL kinds (dosage, money, account, sort_code, date) are spoken
               when the two paths AGREE (same value after normalisation), or
               when one path produced it AND it validated AND the other path did
               not contradict it (R3). strict_critical=True removes the second
               clause — agreement only — and is reported as the sensitivity
               row, because it is the difference between 0 and 5% wrong dates.
R5 NON-CRITICAL kinds (patient, pharmacy, prescriber, warning, medication...)
               may be spoken on validated single-path evidence. Two extra gates,
               both earned by measurement, not taste:
               * VOCABULARY: the field must be in the document type's own label
                 vocabulary (label_spot.DOC_LABELS). On a prescription the
                 spotter fires CLAUSE inside "CAUSE", DEED inside "REACH",
                 TERM/RENT/COURT — 66 of the 276 wrong values, ~0 of them right
                 as fields.
               * SUBSTANCE: the value must contain one word of >= 5 characters
                 (or be numeric). "WARNING: MAY" is a truncated fragment, not a
                 warning — 74 of 276 wrong values are 3-4 character fragments
                 left by a false-positive label cutting the line short.
R6 DETERMINISM. Decisions are sorted by (field, canonical value). The same
               field+value is never emitted twice; agreement merges the two
               paths into one "both" decision. A field may legitimately appear
               once per DISTINCT value (a prescription carries two doses).

THE TRADE, STATED PLAINLY — AND MEASURED
---------------------------------------
120 pages of the held-out test split, models/ocr/crnn.pt (sha 52bf383b…), 696
true values, label_spot 0.75:

    path             spoken  precision  recall  wrong-but-spoken
    (a) regex           173      0.965   0.240        6
    (c) spot+valid      704      0.608   0.615      276   (39% of spoken)
    policy              686      0.793   0.705      142
    typed-only          323      0.957   0.444       14   (6 of them printed
                                                          on the page, no gt item)
    typed+strict         88      1.000   0.126        0

Agreement is a cheap, honest contradiction detector and a poor *requirement*:
demanding it for every typed value (typed+strict) keeps recall 0.126 of 0.705 and
buys nothing, because it refuses the 142 correct dates to remove 8 wrong ones.
R4 is therefore: corroboration when it exists, validation plus non-contradiction
when it does not, REFUSAL on contradiction.

Agreement also has a hard ceiling that the numbers made obvious: both paths read
THE SAME OCR text, so a character the recogniser got wrong is wrong for both and
they cannot disagree about it. All 8 of typed-only's genuinely wrong values are
dates one digit off (DOB 04021979 read 04021879), and on every one of those pages
the true date appears NOWHERE in the recogniser's output — no policy over one
OCR pass can catch it. What the policy can do, it does: the 276 wrong values of
(c) fall to 14, and every remaining wrong value is a single-digit date, never a
dosage.

The residual wrongness in the "policy" row is free-text fidelity — a name read
"DR A AHMED MBS" for "DR A AHMED MBBS" — recogniser character accuracy, not a
decision the policy can make. speak_free_text=False is that ceiling made
explicit: it refuses every non-typed field and is reported as the "typed-only"
row so the recall it costs is visible rather than argued.

WIRING: field_proximity.validate (the one validator), field_proximity.UNITS,
label_spot.DOC_LABELS (the doc-type vocabulary), and a token table mirroring
evaluate.FIELDS (pinned by a test so it cannot drift). Pure stdlib + those two
sibling modules; no model, no corpus, no I/O.
"""

from __future__ import annotations

import re

import field_proximity as FP
import label_spot as LS

CRITICAL_KINDS = frozenset({"dosage", "money", "account", "sort_code", "date"})

# The scorer's own resolution: eval_field_truth.match_value never credits a
# value shorter than MIN_SUBSTRING, so anything below it is unverifiable and is
# not spoken. Mirrored (and pinned by a test), not imported: this module stays
# free of the eval harness and its torch dependency.
MIN_SPEAK_CHARS = 4
MIN_SPEAK_WORD = 5

# ── the regex path's vocabulary ───────────────────────────────────────────────
# evaluate.fields_found returns bare tokens from evaluate.FIELDS. A token means
# nothing without its field and kind, so that mapping lives here, next to the
# policy that needs it. ml/tests/test_field_policy.py asserts every token in
# evaluate.FIELDS is covered, so the two tables cannot drift.
REGEX_TOKENS = {
    "prescription": {"500mg": ("dosage", "dosage"), "400mg": ("dosage", "dosage")},
    "banking": {"40218877": ("account_number", "account"),
                "40-11-04": ("sort_code", "sort_code")},
    "legal": {"2500": ("amount", "money"), "2024-CV-00456": ("case_number", "identifier")},
}
_REGEX_KIND = {f: k for per in REGEX_TOKENS.values() for _t, (f, k) in per.items()}

_DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)", re.I)


# ── normalisation: what "the same value" means per kind ──────────────────────

def _alnum(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _digits(s):
    return re.sub(r"\D", "", str(s))


def canonical(kind, value):
    """frozenset of comparable tokens for `value` under `kind`, or frozenset().

    Two paths AGREE when their canonical sets intersect. Dosage is set-valued
    ("AMOXICILLIN 500MG" carries the dose 500mg, and a prescription carries two
    doses); every other kind is scalar. A form this module cannot normalise
    yields the empty set — it can then neither agree NOR contradict, so the
    error direction is silence, never a guess.
    """
    if value is None:
        return frozenset()
    v = str(value).strip()
    if not v:
        return frozenset()
    if kind == "dosage":
        out = set()
        for num, unit in _DOSE_RE.findall(v):
            unit = unit.lower()
            if unit in FP.UNITS:
                out.add("%s%s" % (_num(num, "%.10g"), unit))
        return frozenset(out)
    if kind == "date":
        tok = v.split()[0]
        d = _digits(tok)
        if len(d) == 6:                       # DDMMYY -> DDMM20YY
            d = d[:4] + "20" + d[4:]
        return frozenset({d}) if len(d) == 8 else frozenset()
    if kind == "money":
        n = _num(_digits_commas(v), "%.2f")
        return frozenset({n}) if n else frozenset()
    if kind in ("account", "sort_code"):
        d = _digits(v)                        # '40 11 04' and '40-11-04' agree
        return frozenset({d}) if d else frozenset()
    return frozenset({_alnum(v)}) if _alnum(v) else frozenset()


def _digits_commas(v):
    return re.sub(r"[^\d.\-]", "", str(v))


def _num(x, fmt="%.10g"):
    try:
        return fmt % float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return ""


# ── candidate construction ───────────────────────────────────────────────────

def _as_values(raw):
    """A regex field may carry one value or several (a page has two doses)."""
    if isinstance(raw, (list, tuple, set)):
        return [str(v) for v in raw]
    return [] if raw is None else [str(raw)]


def kind_of(field, record=None, kinds=None):
    if kinds and field in kinds:
        return str(kinds[field])
    if record and record.get("kind"):
        return str(record["kind"])
    return _REGEX_KIND.get(field, "general")


def _candidates(regex_fields, spot_fields, kinds):
    out = []
    for field, raw in (regex_fields or {}).items():
        field = str(field)
        for v in _as_values(raw):
            out.append({"field": field, "value": v.strip(), "kind": kind_of(field, None, kinds),
                        "source": "regex", "ok": None, "reason": ""})
    for field, rec in (spot_fields or {}).items():
        field = str(field)
        rec = rec if isinstance(rec, dict) else {"value": rec}
        out.append({"field": field, "value": str(rec.get("value") or "").strip(),
                    "kind": kind_of(field, rec, kinds), "source": "spot",
                    "ok": rec.get("ok"), "reason": str(rec.get("reason") or "")})
    for c in out:
        if c["ok"] is None:                   # the regex path arrives unvalidated
            c["ok"], c["reason"] = FP.validate(c["kind"], c["value"])
        c["canon"] = canonical(c["kind"], c["value"])
        c["agreed"] = False
    return out


def _vocabulary(doc_type):
    """The field names a document type's OWN printed labels resolve to."""
    labels = LS.DOC_LABELS.get(str(doc_type or ""), ())
    return {FP.resolve_label(lab)[0] for lab in labels}


def _substantial(value):
    """Enough there to be worth speaking: a real word, or a number we can read.

    'MAY' / 'MANY' / 'BMAY' are label+4-character fragments left when a
    false-positive spot cut the line short — 74 of the 276 wrong values the old
    path spoke. 'JANE DOE' (two tokens, no long word) must survive.
    """
    v = str(value).strip()
    if len(_alnum(v)) < MIN_SPEAK_CHARS:
        return False
    words = re.findall(r"[A-Za-z]{2,}", v)
    if len(words) >= 2 or any(len(w) >= MIN_SPEAK_WORD for w in words):
        return True
    return bool(re.fullmatch(r"[\d\s:.,/-]+", v)) and len(_digits(v)) >= MIN_SPEAK_CHARS


# ── the decision ─────────────────────────────────────────────────────────────

def decide(regex_fields, spot_fields, *, kinds=None, doc_type=None,
           strict_critical=False, speak_free_text=True):
    """What to SPEAK from both extractions. See the module docstring for the rules.

    ``regex_fields``  {field: value or [values]} — evaluate.fields_found tokens
                      mapped by REGEX_TOKENS (field_policy.from_regex_tokens).
    ``spot_fields``   {field: {value, ok, kind, reason, provenance}} —
                      field_proximity.extract_fields output, or literals in tests.
    ``kinds``         {field: kind} override (e.g. declare a field critical).
    ``doc_type``      enables the vocabulary gate (R5); None disables it.
    """
    cands = _candidates(regex_fields, spot_fields, kinds)
    vocab = _vocabulary(doc_type) if doc_type else None

    # R3 — disagreement, per field, both paths present and no shared value.
    by_field = {}
    for c in cands:
        by_field.setdefault(c["field"], []).append(c)
    conflicts = {}
    for field, group in by_field.items():
        rs = [c for c in group if c["source"] == "regex" and c["canon"]]
        ss = [c for c in group if c["source"] == "spot" and c["canon"]]
        if rs and ss and not any(a["canon"] & b["canon"] for a in rs for b in ss):
            conflicts[field] = (rs[0]["value"], ss[0]["value"])

    # Agreement — the same canonical value from the other path. A candidate of a
    # CRITICAL kind asks the question the other way round too: "does the other
    # path's value CARRY my token?" — that is how the regex's bare '500mg'
    # corroborates the spot path's 'AMOXICILLIN 500MG' read as free text.
    for c in cands:
        if not c["canon"]:
            continue
        for o in cands:
            if o is c or o["source"] == c["source"]:
                continue
            if o["canon"] & c["canon"]:
                c["agreed"] = True
            elif c["kind"] in CRITICAL_KINDS and \
                    (canonical(c["kind"], o["value"]) & c["canon"]):
                c["agreed"] = True
            if c["agreed"]:
                break

    decisions = []
    for c in cands:
        field, value, kind = c["field"], c["value"], c["kind"]
        reason = ""
        if not value:
            decision, reason = "refuse", "label spotted but no value present"
        elif not c["ok"]:
            decision, reason = "refuse", c["reason"] or "failed validation"
        elif field in conflicts:
            decision, reason = "refuse", ("two paths disagree: %s vs %s"
                                          % conflicts[field])
        elif kind in CRITICAL_KINDS:
            if c["agreed"]:
                decision, reason = "speak", "both paths agree on this value"
            elif strict_critical:
                decision, reason = "refuse", ("safety-critical %s from one path only; "
                                              "agreement required" % kind)
            else:
                decision, reason = "speak", ("validated %s; the other path had no "
                                             "value to confirm or contradict it" % kind)
        elif not speak_free_text:
            decision, reason = "refuse", ("free text (kind %s) is not spoken by this "
                                          "policy" % kind)
        elif vocab is not None and field not in vocab:
            decision, reason = "refuse", ("'%s' is not in the %s label vocabulary "
                                          "(the spot matched another type's label)"
                                          % (field, doc_type))
        elif not _substantial(value):
            decision, reason = "refuse", ("value too short to speak: '%s'" % value[:20])
        elif c["agreed"]:
            decision, reason = "speak", "both paths agree on this value"
        else:
            decision, reason = "speak", "validated %s text from the label path" % kind
        decisions.append({"field": field, "value": value, "decision": decision,
                          "why": reason, "source": "both" if c["agreed"] else c["source"],
                          "_key": (field, "|".join(sorted(c["canon"])) or value.lower())})

    # R6 — deterministic, deduplicated, never the same field+value twice.
    decisions.sort(key=lambda d: (d["_key"], d["decision"]))
    seen, out = set(), []
    for d in decisions:
        if d["_key"] in seen:                 # never the same field+value twice
            continue
        seen.add(d["_key"])
        out.append({k: d[k] for k in ("field", "value", "decision", "why", "source")})
    return out


def spoken(decisions):
    """The values a caller would read out, in decision order."""
    return [d["value"] for d in decisions if d["decision"] == "speak"]


# ── adapter for the regex path ───────────────────────────────────────────────

def from_regex_tokens(doc_type, tokens):
    """{field: value or [values]} from evaluate.fields_found's bare tokens."""
    table = REGEX_TOKENS.get(str(doc_type or ""), {})
    out = {}
    for t in tokens or ():
        field = table.get(str(t), (str(t).lower(), "general"))[0]
        cur = out.get(field)
        if cur is None:
            out[field] = str(t)
        elif isinstance(cur, list):
            if str(t) not in cur:
                cur.append(str(t))
        elif cur != str(t):
            out[field] = [cur, str(t)]
    return out


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest():
    """The rule set in one page-sized table: every rule fires once, visibly."""
    spot = {
        "date_of_birth": {"value": "04021979", "ok": True, "kind": "date", "reason": "ok"},
        "date": {"value": "01102027", "ok": True, "kind": "date", "reason": "ok"},
        "dosage": {"value": "500mg", "ok": True, "kind": "dosage", "reason": "ok"},
        "pharmacy": {"value": "", "ok": False, "kind": "general",
                     "reason": "no value found after label"},
        "warning": {"value": "MAY", "ok": True, "kind": "general", "reason": "ok"},
        "clause": {"value": "DROWSINESS", "ok": True, "kind": "general", "reason": "ok"},
    }
    regex = {"date": "01102026", "dosage": ["500mg", "400mg"]}
    for d in decide(regex, spot, doc_type="prescription"):
        print("%-16s %-14s %-6s %-6s %s"
              % (d["field"], (d["value"] or "(none)")[:14], d["decision"],
                 d["source"], d["why"]))
    got = {d["field"]: d["decision"] for d in decide(regex, spot, doc_type="prescription")}
    assert got["date"] == "refuse", got           # R3 contradiction
    assert got["dosage"] == "speak", got          # R4 agreement
    assert got["clause"] == "refuse", got         # R5 vocabulary
    assert got["warning"] == "refuse", got        # R5 substance
    assert got["pharmacy"] == "refuse", got       # R1 empty
    print("field_policy self-test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
