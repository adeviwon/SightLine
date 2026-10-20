"""
Field lexicon + field-grammar vocabulary — Tesseract's second structural trick.

Tesseract reads clean text better than our CRNN because its word output is
constrained by an English lexicon; its errors are pruned by a dictionary and
ours are not. This module builds that dictionary FROM OUR OWN DOMAIN (no system
word list exists on this box, so tokens are mined from data on disk) and pins
the field grammar we actually extract, so a word-level read can be scored
against the vocabulary the product cares about.
"""

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import field_lexicon as FL  # noqa: E402
import realdata as R        # noqa: E402


def test_every_pattern_compiles():
    for cls, pats in FL.PATTERNS.items():
        assert pats, f"empty pattern list for {cls}"
        for p in pats:
            try:
                re.compile(p)
            except re.error as e:  # pragma: no cover - failure path
                raise AssertionError(f"{cls}: bad regex {p!r}: {e}")


def test_known_recovered_values_classify():
    # Values the product actually extracts (evaluate.FIELDS, pipeline.js).
    assert "sort_code" in FL.classify("40-11-04")
    assert "dosage" in FL.classify("500mg")
    assert "date" in FL.classify("01/10/2026")
    assert "money" in FL.classify("2,500.00")
    assert "money" in FL.classify("GBP -1,100.00")
    assert "case_ref" in FL.classify("2024-CV-00456")
    assert "account_masked" in FL.classify("Card ending 4521")
    assert "dosage" in FL.classify("400mg")
    # A sort code must not be classified as a date or an account number.
    cls = FL.classify("40-11-04")
    assert "date" not in cls


def test_four_date_formats_are_covered():
    for d in ("01/10/2026", "01-10-2026", "1 October 2026", "2026-10-01"):
        assert "date" in FL.classify(d), d


def test_token_outside_charset_rejected():
    # valid_token is about the MODEL CHARSET (A-Z0-9), so - / : . £ are rejected.
    for bad in ("JANE-DOE", "café", "40/11/04", "£2,500", "O'BRIEN", "ACCOUNT#"):
        assert not FL.valid_token(bad), bad
    for good in ("ACCOUNT", "40218877", "500MG", "SORT"):
        assert FL.valid_token(good), good


def test_build_writes_deduped_sorted_uppercase(tmp_path):
    out = tmp_path / "lexicon.txt"
    report = FL.build(out_path=out)
    assert out.exists()
    toks = [t for t in out.read_text().splitlines() if t]
    assert toks, "lexicon must not be empty"
    assert toks == sorted(toks), "lexicon must be sorted"
    assert toks == sorted(set(toks)), "lexicon must be deduped"
    assert all(t == t.upper() for t in toks), "lexicon must be uppercase"
    assert all(FL.valid_token(t) for t in toks), "lexicon must be charset-clean"


def test_build_report_has_domain_and_class_counts(tmp_path):
    report = FL.build(out_path=tmp_path / "lexicon.txt")
    assert report["total"] > 0
    assert "domain" in report and "system" in report
    assert isinstance(report["class_counts"], dict)
    # The domain vocabulary must contain the product's own word tokens.
    for t in ("500MG", "40218877", "AMOXICILLIN"):
        assert t in report["domain"], t
    # And the field grammar must have caught the structured values.
    assert report["class_counts"].get("sort_code", 0) >= 1
    assert "40-11-04" in report["classes"]["sort_code"]


def test_lexicon_includes_expected_field_tokens():
    # The field vocabulary the product reads must be in the mined lexicon.
    toks = set(FL.build(out_path=Path("/tmp/_fl_test_lex.txt"))["domain"])
    for t in ("SORT", "CODE", "ACCOUNT", "NUMBER", "ENDING", "BALANCE",
              "DATE", "DOB", "PHARMACY", "PRESCRIPTION"):
        assert t in toks, t


def test_domain_lines_reads_the_train_split_only(monkeypatch):
    """The lexicon must not learn from val or TEST labels.

    Measured leak: _domain_lines() called RC.load() with no split, so the domain
    vocabulary and the field grammar were mined from all three splits. A field
    spotter given the test strings it will be scored on reports a recall it did
    not earn. The shipped vocabulary is a fixed product list, so train-only costs
    nothing.
    """
    import field_lexicon as FL
    import render_corpus as RC
    seen = []

    def fake_load(*args, **kwargs):
        seen.append(kwargs.get("split"))
        return {"labels": ["ACCOUNT NUMBER 40218877"]}

    monkeypatch.setattr(RC, "load", fake_load)
    FL._domain_lines()
    assert seen, "the corpus was never read — the test is not exercising the call"
    assert all(s == "train" for s in seen), f"splits read: {seen}"
