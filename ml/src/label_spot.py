"""
Closed-vocabulary label spotting — the blind-reader path, not transcription.

A sighted reader transcribes a page. A blind user holding a phone over a
prescription needs ONE thing: the value next to a label they already know
("ACCOUNT NUMBER", "SORT CODE", "TERM"). Spotting one of ~15 known labels is
APPROXIMATE STRING MATCHING against a CLOSED list, so it survives a recogniser
that is only ~70% character-accurate — which is where this repo is (the CRNN
reads its own renders at 4.4% exact). Published: a lexicon-constrained decoder
cuts word error 35-41.5% (Weinman et al., PMC3021989), and keyword-proximity
field extraction is the standard product pattern.

WHY IT WORKS HERE. Our recogniser's characteristic failure is DIGIT-FOR-LETTER:
it learned one typeface family and digitises letters on anything else —
PRESCRIPTION -> PRE5CR1P710N, DATE01102026 -> 0A7E01102026 (S->5, T->7, O->0,
I->1, G->6, B->8; capture.py documents the cause). Folding those classes on BOTH
sides turns those misses back into exact hits at zero edit cost, and the small
edit budget absorbs the rest of the noise.

WHAT IT WILL NOT DO. It does not transcribe, does not know values, does not
touch a model, the network, or a corpus. Pure stdlib over text already in hand.

Contract (frozen — other lanes code against it):
    spot_labels(text, threshold=0.75) -> [{"label", "start", "end", "score"}]
        start/end are character offsets into `text`, end exclusive, of the
        MATCHED REGION (not of the canonical label).
    spot_in_lines(lines) -> same dicts + "line_index"
        lines are {"text": str, "box": (x0, y0, x1, y1)}.

    bash run.sh py ml/src/label_spot.py --selftest   # hand-built cases, table
    bash run.sh py ml/src/label_spot.py --measure    # real OCR text, P/R
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ── the vocabulary ────────────────────────────────────────────────────────────
#
# Mined from this repo, not imagined: the label keywords the three document
# types in capture.DOCS actually print, plus the field classes field_lexicon
# names and the keywords evaluate.classify_doc matches on (COURT/DEED appear
# there and not in the renders). `provenance()` re-derives that mapping from
# disk, and ml/tests/test_label_spot.py fails if a label has no repo source.
#
# Kept to LABEL phrases, not field_lexicon's 2,878 mined tokens: the tokens are
# values and prose, and firing on those would bury the ~15 headings that matter
# under false positives. A closed list is the point.
DOC_LABELS = {
    "prescription": (
        "PRESCRIPTION", "PATIENT", "DOB", "DATE", "PRESCRIBER", "PHARMACY",
        "RX", "TAKE", "WARNING", "DOSAGE", "MG", "TABLETS", "CAPSULES",
    ),
    "banking": (
        "HSBC UK BANK PLC", "STATEMENT", "ACCOUNT HOLDER", "ACCOUNT NUMBER",
        "SORT CODE", "PERIOD", "OPENING BALANCE", "SALARY CREDIT", "RENT",
        "TESCO", "BRITISH GAS", "ATM WITHDRAWAL", "CASHBACK",
    ),
    "legal": (
        "LEASE AGREEMENT", "LEASE", "AGREEMENT", "CLAUSE", "TENANT",
        "LANDLORD", "PROPERTY", "TERM", "NOTICE", "COURT", "DEED",
    ),
}

VOCAB = tuple(sorted({lab for labs in DOC_LABELS.values() for lab in labs}))

# ── normalisation ─────────────────────────────────────────────────────────────
#
# OCR's digit-for-letter confusions folded to one canonical class each, plus
# case and punctuation. Folding is free (no edit budget spent) and it is the
# single highest-value rule here.
_CONFUSE = {"0": "O", "1": "I", "5": "S", "7": "T", "6": "G", "8": "B",
            "L": "I"}          # L/I is a glyph confusion, not a digit one
_ALNUM = re.compile(r"[0-9A-Z]")


def _norm(text):
    """(folded string, index map into `text`) — punctuation and case dropped."""
    out, idx = [], []
    for i, ch in enumerate(str(text).upper()):
        if not _ALNUM.match(ch):
            continue
        out.append(_CONFUSE.get(ch, ch))
        idx.append(i)
    return "".join(out), idx


_NORM_LABELS = {lab: _norm(lab)[0] for lab in VOCAB}


def _lev(a, b, cap):
    """Levenshtein distance, abandoned early once it exceeds `cap` (-> cap+1)."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def spot_labels(text, threshold=0.75):
    """Find known labels in `text`; return spotted regions, longest first.

    A candidate is a same-length window of the folded text whose edit distance
    to a folded label is within budget, PROVIDED the window starts and ends on a
    token boundary — that is what stops "TERM" firing inside "TERMINAL"/"TERMS".

    The budget is the score gate: accepted iff 1 - dist/len >= threshold, so the
    default 0.75 means <=1.5 edits per 6 characters, and the conservative
    1-edit-per-6 rule the brief asks for is threshold=0.833. A separate hard
    cap was rejected: it binds tighter than any threshold below 0.833 and would
    have made the knob one-directional.

    ponytail: windows are len(label) or len(label)-1 folded chars, so a
    mid-label INSERTION in the OCR shifts the alignment and can miss while
    dropped characters, split/spaced text and substitutions are caught. Take a
    longer window too if recall matters more than the inside-a-longer-token
    guarantee (it is what would make TERM fire on "TERMS").
    """
    text = text if isinstance(text, str) else ""
    norm, idx = _norm(text)
    if not norm:
        return []

    candidates = []
    for label in VOCAB:
        lab = _NORM_LABELS[label]
        L = len(lab)
        if L > len(norm):
            continue
        cap = int((1.0 - threshold) * L + 1e-9)
        # Window = the folded label's length, or one shorter: OCR DROPS characters
        # (PATIENT -> PATENT) far more often than it invents them, and a shorter
        # window is the only way to align that. One LONGER window is deliberately
        # not tried — it is what would let TERM fire inside "TERMS".
        for wl in ((L, L - 1) if L > 2 else (L,)):
            if wl > len(norm):
                continue
            for i in range(len(norm) - wl + 1):
                prv = idx[i] - 1
                if prv >= 0 and text[prv].isalnum():
                    continue                    # starts inside a longer token
                j = i + wl
                nxt = idx[j - 1] + 1
                if nxt < len(text) and text[nxt].isalnum():
                    continue                    # ends inside a longer token
                d = _lev(lab, norm[i:j], cap)
                if d > cap:
                    continue
                score = 1.0 - d / L
                if score < threshold:
                    continue
                candidates.append((-L, idx[i], -score, label, idx[i], nxt, score))

    # Longest match first, then earliest start, then best score: two spots never
    # cover the same characters, and the result does not depend on dict order.
    candidates.sort(key=lambda c: c[:4])
    taken, kept = [], []
    for _L, _s, _sc, label, s, e, score in candidates:
        if any(s < te and ts < e for ts, te in taken):
            continue
        taken.append((s, e))
        kept.append({"label": label, "start": s, "end": e,
                     "score": round(score, 4)})
    kept.sort(key=lambda d: (d["start"], d["end"], d["label"]))
    return kept


def spot_in_lines(lines):
    """Spot labels line by line, tagging each spot with its `line_index`."""
    out = []
    for i, line in enumerate(lines or []):
        for spot in spot_labels((line or {}).get("text", "")):
            spot["line_index"] = i
            out.append(spot)
    return out


# ── provenance: the vocabulary came off disk ──────────────────────────────────

def provenance():
    """{label: source} where source names where the label was found in the repo.

    Sources are capture.DOCS (the phrase is printed by one of the three
    documents), evaluate.classify_doc (a keyword that classifier matches on) or
    field_lexicon.PATTERNS (a field class, e.g. DOSAGE). None = invented.
    """
    try:
        import capture
        printed = " ".join(line for lines, _ in capture.DOCS for line in lines).upper()
    except Exception:                                          # pragma: no cover
        printed = ""
    try:
        import inspect
        import evaluate
        kws = set(re.findall(r"[A-Za-z]{2,}", inspect.getsource(evaluate.classify_doc)))
        keywords = " ".join(sorted(kws)).upper()
    except Exception:                                          # pragma: no cover
        keywords = ""
    try:
        import field_lexicon
        classes = " ".join(field_lexicon.PATTERNS).upper()
    except Exception:                                          # pragma: no cover
        classes = ""

    src = {}
    for lab in VOCAB:
        if lab in printed:
            src[lab] = "capture.DOCS"
        elif lab in keywords:
            src[lab] = "evaluate.classify_doc"
        elif lab in classes:
            src[lab] = "field_lexicon.PATTERNS"
        else:
            src[lab] = None
    return src


# ── self-test: the hand-built cases, printed as a table ───────────────────────

# (line, labels that must be spotted). Digit-for-letter cases are marked.
CASES = [
    ("Account Number: 40218877", ["ACCOUNT NUMBER"]),
    ("ACC0UNT N0MBER 40218877", ["ACCOUNT NUMBER"]),          # 0 for O
    ("SORT C0DE 40-11-04", ["SORT CODE"]),                    # 0 for O
    ("PRESCRIPTION", ["PRESCRIPTION"]),
    ("PATENT: JANE DOE   DOB: 04/02/1979", ["PATIENT", "DOB"]),
    ("DATE 01102026", ["DATE"]),
    ("PRESCRIGER DTH AMED", ["PRESCRIBER"]),                   # B -> G
    ("Rx  Amoxicillin 500mg capsules", ["RX", "CAPSULES"]),
    # MG is a unit glued to its number in every real render ("500mg"), so the
    # inside-a-longer-token rule suppresses it there BY DESIGN; the value path
    # (field_lexicon's dosage pattern) is what reads that. Spaced, it spots:
    ("Dosage: 500 MG per day", ["DOSAGE", "MG"]),
    ("K 1BUDICTEN 400M0 1ABLETS", ["TABLETS"]),                # 1 for T, 1 for L
    ("HSBC UK BANK PLC", ["HSBC UK BANK PLC"]),
    ("01/03 SALARY CREDIT   GBP 2,900.00", ["SALARY CREDIT"]),
    ("14/03 BRITISH GA5 DD", ["BRITISH GAS"]),                 # 5 for S
    ("LEASE AGREEMENT", ["LEASE AGREEMENT"]),                  # longer label wins
    ("TERM: 24 months from 1 February 2024", ["TERM"]),
    ("TERMINAL 4 IS CLOSED", []),                              # TERM must NOT fire
    ("TERMS AND CONDITIONS APPLY", []),                        # nor here
    ("MAKNING MAY AUSE IONSNETE", []),                         # damaged beyond budget
    ("the quick brown fox jumps over", []),                    # nothing at all
    ("", []),                                                   # no crash on empty
]


def _selftest():
    print("label_spot self-test — %d cases, vocabulary %d labels "
          "(%s)" % (len(CASES), len(VOCAB),
                    ", ".join("%s %d" % (k, len(v)) for k, v in DOC_LABELS.items())))
    print("%-42s %-18s %6s  %s" % ("line", "spotted label", "score", "offsets"))
    print("-" * 84)
    fails = 0
    for text, want in CASES:
        got = spot_labels(text)
        labels = [s["label"] for s in got]
        show = text if text else "(empty)"
        if not got:
            print("%-42s %-18s %6s  %s" % (show[:42], "-", "-", "-"))
        for s in got:
            print("%-42s %-18s %6.3f  (%d,%d) %r"
                  % (show[:42], s["label"], s["score"], s["start"], s["end"],
                     text[s["start"]:s["end"]]))
            show = ""
        if labels != want:
            fails += 1
            print("!! expected %s, got %s" % (want, labels))

    # spot_in_lines keeps the shape and adds line_index
    lines = [{"text": t, "box": (0, 0, 10, 10)} for t in ("HSBC UK BANK PLC",
                                                          "nothing",
                                                          "SORT C0DE 40-11-04")]
    sl = spot_in_lines(lines)
    assert [x["label"] for x in sl] == ["HSBC UK BANK PLC", "SORT CODE"], sl
    assert sl[0]["line_index"] == 0 and sl[1]["line_index"] == 2, sl

    # determinism, and every offset actually lands on what was matched
    for text, _want in CASES:
        assert spot_labels(text) == spot_labels(text), text
        for s in spot_labels(text):
            assert _norm(text[s["start"]:s["end"]])[0] == _norm(s["label"])[0] \
                or s["score"] < 1.0, s

    prov = provenance()
    missing = sorted(l for l, s in prov.items() if s is None)
    print("-" * 84)
    print("provenance: %d/%d labels trace to disk; sources %s"
          % (len(VOCAB) - len(missing), len(VOCAB),
             ", ".join(sorted({s for s in prov.values() if s}))))
    if missing:
        print("!! no repo source: %s" % missing)
    assert not missing, missing
    assert fails == 0, "%d case(s) failed" % fails
    print("self-test ok")
    return 0


# ── measurement: what the spotter does on REAL OCR text ───────────────────────

def measure(path=None, thresholds=(0.75, 0.833)):
    """Precision/recall of label spotting over the pipeline's real OCR text.

    Each entry of artifacts/pipeline_raw.json is one page: its `ocr_text` is the
    recogniser's raw output and its `expected_text` is the clean document, so a
    page's expected label set is exact ground truth (every vocabulary label the
    clean text prints) with no hand-labelling. Recall is over (page, label)
    pairs — a heading printed twice only has to be found once — which is what a
    reader needs: a label and a value to read beside it.
    """
    path = Path(path or (ROOT / "artifacts" / "pipeline_raw.json"))
    pages = json.loads(path.read_text())
    by_doc = {}
    for page in pages:
        clean = (page.get("expected_text") or "").upper()
        exp = {lab for lab in DOC_LABELS.get(page["doc"], ()) if lab in clean}
        lines = (page.get("ocr_text") or "").splitlines()
        by_doc[page["doc"] + "/" + page["profile"]] = (exp, lines)
    print("%s: %d pages, %d OCR lines, vocabulary %d labels"
          % (path.name, len(pages),
             sum(len((p.get("ocr_text") or "").splitlines()) for p in pages),
             len(VOCAB)))

    for th in thresholds:
        tp = fp = 0
        found, expected = set(), set()
        recs = []
        print("\n== threshold %.3f ==" % th)
        print("%-30s %4s %4s  %s" % ("page", "TP", "FP", "missed labels"))
        for name, (exp, lines) in by_doc.items():
            doc = name.split("/")[0]
            # spot_in_lines() has no threshold parameter (frozen signature), so
            # the sweep calls spot_labels() per line — same thing at 0.75.
            spots = [dict(s, line_index=i) for i, line in enumerate(lines)
                     for s in spot_labels(line, th)]
            hits = [s for s in spots if s["label"] in exp]
            noise = [s for s in spots if s["label"] not in exp]
            recs.append(len({s["label"] for s in hits}) / len(exp) if exp else 0.0)
            tp += len(hits)
            fp += len(noise)
            found |= {(doc, s["label"]) for s in hits}
            expected |= {(doc, lab) for lab in exp}
            print("%-30s %4d %4d  %s"
                  % (name, len(hits), len(noise),
                     " ".join(sorted(exp - {s["label"] for s in hits})) or "-"))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = len(found & expected) / len(expected) if expected else 0.0
        print("  precision %.3f (%d/%d spots), recall %.3f (%d/%d page-labels)"
              % (prec, tp, tp + fp, rec, len(found & expected), len(expected)))
        print("  per-page recall mean %.3f, pages with nothing spotted %d/%d"
              % (sum(recs) / len(recs) if recs else 0.0,
                 sum(1 for r in recs if r == 0.0), len(recs)))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    if "--measure" in sys.argv:
        i = sys.argv.index("--measure")
        measure(sys.argv[i + 1] if len(sys.argv) > i + 1 else None)
        sys.exit(0)
    print("usage: label_spot.py --selftest | --measure [pipeline_raw.json]")
    sys.exit(1)
