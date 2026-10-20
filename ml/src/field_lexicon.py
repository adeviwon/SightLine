"""
Field lexicon + field grammar — Tesseract's second structural advantage.

Tesseract reads clean text better than our CRNN for two reasons that are not a
better neural net: its output is WORD-level and constrained by an English
lexicon, so a wrong character gets pruned by a dictionary. Ours is not, so one
wrong character kills a whole token and the field with it.

This module builds the equivalent dictionary FROM OUR OWN DOMAIN and pins the
field grammar the product actually extracts, so a word-level read can be scored
against the vocabulary that matters.

NO SYSTEM WORD LIST. This box has no /usr/share/dict/words (only
README.select-wordlist), so the lexicon is MINED FROM DATA ON DISK: the domain
documents in capture.DOCS, the rendered-corpus labels, and the handheld receipt
transcripts. `report["system"]` is therefore empty and is reported as such
rather than pretending a system list contributed.

Field grammar is copied from the product, not invented: evaluate.FIELDS and the
PATTERNS table in app/js/pipeline.js.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]

# The model's emit alphabet. A token containing anything else cannot be produced
# by the recogniser, so it is not a candidate lexicon entry.
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_VALID = set(CHARSET)
_TOKEN_RE = re.compile(r"^[0-9A-Z]+$")

# ── field grammar (product classes) ───────────────────────────────────────────
#
# money   — currency symbol or ISO code, thousands separators, leading sign
# date    — the four formats the app writes/reads
# dosage  — number + mass/volume/form unit
# freq    — dosing frequency phrases
# acct    — account/card numbers masked to the last four
# sort    — UK sort code NN-NN-NN
# case    — case/reference number YYYY-XX-NNNNN
# iban    — GB IBAN
PATTERNS = {
    "money": [
        r"[£$€]\s?-?\s?\d[\d,]*(?:\.\d{2})?",
        r"\b(?:GBP|USD|EUR)\s?-?\s?\d[\d,]*(?:\.\d{2})?",
        r"\b\d[\d,]*\.\d{2}\b",
        r"-\s?[£$€]?\s?\d[\d,]*(?:\.\d{2})?",
    ],
    "date": [
        r"\b(?:0?[1-9]|[12]\d|3[01])/(?:0?[1-9]|1[0-2])/(?:\d{2}|\d{4})\b",
        r"\b(?:0?[1-9]|[12]\d|3[01])-(?:0?[1-9]|1[0-2])-(?:\d{2}|\d{4})\b",
        r"\b(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{4}\b",
        r"\b\d{4}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])\b",
    ],
    "dosage": [
        r"\b\d{1,4}(?:\.\d+)?\s?(?:MG|MCG|ML|G|TABLETS?|CAPSULES?)\b",
    ],
    "frequency": [
        r"\bEVERY\s+\d+\s*(?:HOURS?|HRS?)\b",
        r"\b\d+\s*TIMES?\s*(?:A\s*)?DAY\b",
        r"\bONCE DAILY\b|\bTWICE DAILY\b|\bTHREE TIMES DAILY\b|\bAT BEDTIME\b",
    ],
    "account_masked": [
        r"\bENDING\s*(?:IN\s*)?\d{4}\b",
        r"\b[*X]{2,}\s*\d{4}\b",
    ],
    "sort_code": [
        r"\b\d{2}-\d{2}-\d{2}\b",
        r"\b\d{2}\s\d{2}\s\d{2}\b",
    ],
    "case_ref": [
        r"\b\d{4}-[A-Z]{2}-\d{3,8}\b",
    ],
    "iban": [
        r"\bGB\d{2}\s?[A-Z]{4}\s?[\dA-Z]{4}\s?\d{4}[\dA-Z]?\s?\d{4}\b",
    ],
}

_COMPILED = {c: [re.compile(p, re.I) for p in pats] for c, pats in PATTERNS.items()}


def valid_token(tok):
    """True when `tok` is a non-empty run of the model's emit alphabet."""
    return bool(tok) and bool(_TOKEN_RE.match(str(tok).strip().upper()))


def classify(token):
    """Field-grammar classes a string belongs to, in PATTERNS order."""
    s = str(token).strip().upper()
    return [c for c in PATTERNS if any(rx.search(s) for rx in _COMPILED[c])]


def _domain_lines():
    """The domain text on disk: rendered documents, corpus labels, transcripts."""
    lines = []
    try:
        import capture
        for doc_lines, _ in capture.DOCS:
            lines.extend(doc_lines)
    except Exception as e:                       # pragma: no cover - env dependent
        print("  (capture.DOCS unavailable: %s)" % e)
    try:
        import render_corpus as RC
        d = RC.load()
        lines.extend(str(x) for x in d["labels"])
    except Exception as e:                       # pragma: no cover - env dependent
        print("  (render corpus labels unavailable: %s)" % e)
    try:
        import realdata as R
        for _, _, tlines in R.handheld_pages():
            lines.extend(tlines)
    except Exception as e:                       # pragma: no cover - env dependent
        print("  (handheld transcripts unavailable: %s)" % e)
    return lines


def system_tokens():
    """Tokens from a system word list, if one exists on this box."""
    for p in ("/usr/share/dict/words", "/usr/share/dict/american-english"):
        f = Path(p)
        if f.exists():
            toks = set()
            for line in f.read_text(errors="ignore").splitlines():
                t = line.strip().upper()
                if valid_token(t):
                    toks.add(t)
            return sorted(toks)
    return []


def build(out_path=None, corpus_dir=None):
    """Mine the lexicon, write it, and return a report dict."""
    lines = _domain_lines()

    domain = set()
    classes = {c: [] for c in PATTERNS}
    for line in lines:
        for raw in str(line).split():
            t = raw.strip().upper()
            if valid_token(t):
                domain.add(t)
        for c in PATTERNS:
            for rx in _COMPILED[c]:
                for m in rx.finditer(str(line)):
                    v = (m.group(0) or "").strip().upper()
                    if v:
                        classes[c].append(v)

    system = set(system_tokens())
    domain -= system
    tokens = sorted(domain | system)

    classes = {c: sorted(set(v)) for c, v in classes.items()}
    class_counts = {c: len(v) for c, v in classes.items()}

    out_path = Path(out_path or (ROOT / "data" / "lexicon.txt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(tokens) + ("\n" if tokens else ""))

    return {
        "total": len(tokens),
        "domain": sorted(domain),
        "system": sorted(system),
        "domain_count": len(domain),
        "system_count": len(system),
        "classes": classes,
        "class_counts": class_counts,
        "out_path": str(out_path),
    }


def _selfcheck():
    assert classify("40-11-04") == ["sort_code"], classify("40-11-04")
    assert "dosage" in classify("500mg")
    assert "date" in classify("01/10/2026")
    assert "date" not in classify("40-11-04")
    assert not valid_token("JANE-DOE") and valid_token("ACCOUNT")
    r = build(out_path=Path("/tmp/_fl_selfcheck.txt"))
    print("field_lexicon self-check ok: %d tokens, classes %s"
          % (r["total"], r["class_counts"]))


if __name__ == "__main__":
    _selfcheck()
