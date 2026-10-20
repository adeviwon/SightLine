"""
Lane H tests — keyword-proximity value extraction + per-type validation.

The half of field extraction that lets the product REFUSE instead of guess: a
value is only spoken if it validates, and the reason string is what the product
surfaces instead.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import field_proximity as FP  # noqa: E402


# ── validate(): one accept + one reject per kind, with the reason ──────────────

@pytest.mark.parametrize("kind,value,reason", [
    ("sort_code", "40-11-04", "ok"),
    ("sort_code", "40 11 04", "ok"),
    ("sort_code", "401104", "ok"),
    ("account", "40218877", "ok"),
    ("date", "01/10/2026", "ok"),
    ("money", "2,500.00", "ok"),
    ("dosage", "500mg", "ok"),
    ("identifier", "2024-CV-00456", "ok"),
    ("general", "Northgate Pharmacy", "ok"),
])
def test_kinds_accept(kind, value, reason):
    ok, why = FP.validate(kind, value)
    assert ok, f"{kind} {value!r} rejected: {why}"
    assert why == reason


@pytest.mark.parametrize("kind,value,reason", [
    ("sort_code", "40-11-0", "6 digits required, got 5"),
    ("account", "123", "4-12 digits required, got 3"),
    ("date", "31/02/2026", "not a calendar date"),
    ("money", "GBP 12.4", "money needs two decimals, got 1"),
    ("dosage", "500xyz", "unknown unit: xyz"),
    ("identifier", "20", "identifier must be 4-20 chars, got 2"),
    ("general", "", "empty value"),
])
def test_kinds_reject_with_reason(kind, value, reason):
    ok, why = FP.validate(kind, value)
    assert not ok
    assert why == reason


def test_sort_code_accepts_three_separators():
    for v in ("40-11-04", "40 11 04", "401104"):
        assert FP.validate("sort_code", v) == (True, "ok"), v


def test_sort_code_rejects_wrong_length_and_nondigits():
    assert not FP.validate("sort_code", "4011")[0]
    assert not FP.validate("sort_code", "4011045")[0]
    ok, why = FP.validate("sort_code", "40-11-0")
    assert not ok and why == "6 digits required, got 5"


def test_account_masked_is_accepted_and_flagged():
    ok, why = FP.validate("account", "4521")
    assert ok and why == "masked, last 4 digits only"
    ok, why = FP.validate("account", "**** 4521")
    assert ok and why == "masked, last 4 digits only"
    assert FP.validate("account", "40218877") == (True, "ok")


def test_account_rejects_out_of_range():
    assert not FP.validate("account", "123")[0]
    assert not FP.validate("account", "1234567890123")[0]


@pytest.mark.parametrize("bad", ["31/02/2026", "00/13/2026", "00/00/2026"])
def test_date_rejects_impossible_calendar_dates(bad):
    ok, why = FP.validate("date", bad)
    assert not ok, bad
    assert why == "not a calendar date"


def test_date_accepts_all_four_declared_formats():
    for v in ("01/10/2026", "01/10/26", "2026-10-01", "01-10-2026"):
        assert FP.validate("date", v) == (True, "ok"), v


def test_date_rejects_a_non_date():
    assert not FP.validate("date", "notadate")[0]
    assert not FP.validate("date", "40-11-04")[0]


def test_money_accepts_signs_symbols_and_iso_pairs():
    for v in ("2,500.00", "£2,500.00", "GBP -1,100.00", "GBP +12.40", "-200.00",
              "2500", "$1,204.33", "EUR 99"):
        assert FP.validate("money", v) == (True, "ok"), v


def test_money_rejects_bad_grouping_and_decimals():
    ok, why = FP.validate("money", "2,68.55")
    assert not ok and why.startswith("bad thousands grouping")
    ok, why = FP.validate("money", "12.4")
    assert not ok and why == "money needs two decimals, got 1"
    ok, why = FP.validate("money", "abc")
    assert not ok and why.startswith("not a monetary amount")


def test_dosage_unit_is_required_and_must_be_known():
    assert FP.validate("dosage", "500 mg") == (True, "ok")
    assert FP.validate("dosage", "2.5ml") == (True, "ok")
    ok, why = FP.validate("dosage", "500xyz")
    assert not ok and why == "unknown unit: xyz"
    ok, why = FP.validate("dosage", "500")
    assert not ok and why.startswith("no dosage in value")


def test_dosage_optional_frequency():
    assert FP.validate("dosage", "500mg capsules") == (True, "ok")
    assert FP.validate("dosage", "500mg three times daily") == (True, "ok")
    assert FP.validate("dosage", "500mg every 6 hours") == (True, "ok")
    ok, why = FP.validate("dosage", "500mg xyz")
    assert not ok and why == "unknown frequency: xyz"


def test_identifier_charset_and_length():
    assert FP.validate("identifier", "2024-CV-00456") == (True, "ok")
    ok, why = FP.validate("identifier", "2024_CV_00456")
    assert not ok and why == "invalid character in identifier: _"
    assert not FP.validate("identifier", "2024-CV-004561234567890")[0]


def test_general_rejects_long_and_alnum_free():
    ok, why = FP.validate("general", "x" * 40)
    assert not ok and why == "too long: 40 chars"
    ok, why = FP.validate("general", "...")
    assert not ok and why == "no alphanumeric"


def test_unknown_kind_is_refused():
    ok, why = FP.validate("nonsense", "x")
    assert not ok and why == "unknown kind: nonsense"


# ── value_after ───────────────────────────────────────────────────────────────

def test_value_after_trims_leading_separators():
    line = "Sort Code: 40-11-04"
    assert FP.value_after(line, (0, len("Sort Code"))) == "40-11-04"
    assert FP.value_after("Account Number: 40218877", (0, 14)) == "40218877"
    assert FP.value_after("Balance # 1,204.33", (0, 7)) == "1,204.33"


def test_value_after_keeps_a_negative_sign():
    line = "Balance -1,100.00"
    assert FP.value_after(line, (0, 7)) == "-1,100.00"


def test_value_after_empty_when_nothing_to_the_right():
    assert FP.value_after("Sort Code:", (0, 9)) == ""


# ── extract_fields ────────────────────────────────────────────────────────────

def _lines(*texts):
    return [{"text": t, "box": (40, 20 + 30 * i, 340, 40 + 30 * i)}
            for i, t in enumerate(texts)]


def _spot(label, line_index, text, score=0.9):
    i = text.find(label)
    return {"label": label, "start": i, "end": i + len(label), "score": score,
            "line_index": line_index}


def test_label_with_no_value_is_reported_not_dropped():
    lines = _lines("Account Number:")
    out = FP.extract_fields(lines, [_spot("Account Number", 0, lines[0]["text"])])
    assert out["account_number"]["ok"] is False
    assert out["account_number"]["provenance"] == "none"
    assert out["account_number"]["value"] == ""
    assert out["account_number"]["reason"]


def test_cross_line_fallback_uses_next_line_and_records_provenance():
    lines = _lines("Sort Code:", "40-11-04")
    out = FP.extract_fields(lines, [_spot("Sort Code", 0, lines[0]["text"])])
    r = out["sort_code"]
    assert (r["value"], r["ok"], r["provenance"]) == ("40-11-04", True, "next-line")


def test_same_line_wins_over_the_next_line():
    lines = _lines("Sort Code: 40-11-04", "99-99-99")
    out = FP.extract_fields(lines, [_spot("Sort Code", 0, lines[0]["text"])])
    assert out["sort_code"]["value"] == "40-11-04"
    assert out["sort_code"]["provenance"] == "same-line-right"


def test_two_labels_on_one_line_each_get_their_own_value():
    text = "Account Number: 40218877   Sort Code: 40-11-04"
    lines = _lines(text)
    spots = [_spot("Account Number", 0, text), _spot("Sort Code", 0, text)]
    out = FP.extract_fields(lines, spots)
    assert out["account_number"]["value"] == "40218877"
    assert out["account_number"]["ok"] is True
    assert out["sort_code"]["value"] == "40-11-04"
    assert out["sort_code"]["ok"] is True


def test_one_value_region_is_never_claimed_twice():
    # Two labels whose value regions start at the same index: the second is
    # dropped rather than reported twice. Deterministic: the earlier label name
    # ('Amount') wins the region.
    text = "Balance: 2,500.00"
    lines = _lines(text)
    a = {"label": "Balance", "start": 0, "end": 7, "score": 0.9, "line_index": 0}
    b = {"label": "Amount", "start": 0, "end": 7, "score": 0.4, "line_index": 0}
    out = FP.extract_fields(lines, [a, b])
    assert len(out) == 1
    assert out["amount"]["value"] == "2,500.00"
    assert out["amount"]["ok"] is True


def test_a_repeated_label_yields_one_field():
    lines = _lines("Sort Code: 40-11-04")
    s = _spot("Sort Code", 0, lines[0]["text"])
    out = FP.extract_fields(lines, [s, dict(s, score=0.5)])
    assert list(out) == ["sort_code"]


def test_extraction_is_deterministic_under_spot_reordering():
    text = "Account Number: 40218877   Sort Code: 40-11-04"
    lines = _lines(text)
    spots = [_spot("Account Number", 0, text), _spot("Sort Code", 0, text)]
    assert FP.extract_fields(lines, spots) == FP.extract_fields(lines, spots[::-1])


def test_result_shape_is_the_frozen_contract():
    lines = _lines("Dosage: 500mg")
    out = FP.extract_fields(lines, [_spot("Dosage", 0, lines[0]["text"])])
    r = out["dosage"]
    assert set(r) == {"value", "ok", "kind", "reason", "provenance"}
    assert r == {"value": "500mg", "ok": True, "kind": "dosage", "reason": "ok",
                 "provenance": "same-line-right"}


def test_out_of_range_spot_is_ignored():
    lines = _lines("Sort Code: 40-11-04")
    s = _spot("Sort Code", 7, lines[0]["text"])
    assert FP.extract_fields(lines, [s]) == {}


# ── product document, end to end ──────────────────────────────────────────────

def test_bank_statement_lines_extract_validated_fields():
    doc = ["Account Number: 40218877", "Sort Code: 40-11-04",
           "Closing Balance        GBP 2,682.55", "Card ending 4521"]
    lines = _lines(*doc)
    spots = [_spot("Account Number", 0, doc[0]), _spot("Sort Code", 1, doc[1]),
             _spot("Closing Balance", 2, doc[2]), _spot("ending", 3, doc[3])]
    out = FP.extract_fields(lines, spots)
    assert out["account_number"]["value"] == "40218877"
    assert out["sort_code"]["value"] == "40-11-04"
    assert out["closing_balance"]["value"] == "GBP 2,682.55"
    assert out["closing_balance"]["kind"] == "money"
    assert out["card_ending"]["ok"] is True
    assert out["card_ending"]["reason"] == "masked, last 4 digits only"
    assert all(r["ok"] for r in out.values()), out


def test_lease_case_reference_and_date_extract():
    doc = ["Date: 01/10/2026", "Case Reference: 2024-CV-00456"]
    lines = _lines(*doc)
    out = FP.extract_fields(lines, [_spot("Date", 0, doc[0]),
                                    _spot("Case Reference", 1, doc[1])])
    assert out["date"] == {"value": "01/10/2026", "ok": True, "kind": "date",
                           "reason": "ok", "provenance": "same-line-right"}
    assert out["case_number"]["value"] == "2024-CV-00456"
    assert out["case_number"]["kind"] == "identifier"


def test_self_check_runs_and_covers_at_least_twelve_cases(capsys):
    assert FP._selftest() == 0
    printed = capsys.readouterr().out
    assert "self-test ok" in printed
    assert printed.count("same-line-right") >= 8


def test_dates_without_separators_are_accepted_because_the_model_cannot_emit_them():
    """The recogniser's alphabet is 0-9 A-Z and space: '/' cannot be produced.

    Every date this product reads therefore arrives as 8 digits. Refusing them
    refused every date the model is able to emit — measured on a perfect read of
    a rendered prescription, where DATE 01102026 was rejected as 'not a calendar
    date' while the page had been transcribed character-for-character.
    """
    import field_proximity as FP
    assert FP.validate("date", "04021979") == (True, "ok")      # DDMMYYYY
    assert FP.validate("date", "01102026") == (True, "ok")
    assert FP.validate("date", "20261001") == (True, "ok")      # YYYYMMDD
    # and an impossible day is still refused, separator-free or not
    assert FP.validate("date", "31022026")[0] is False
    assert FP.validate("date", "32012026")[0] is False          # day 32
    assert FP.validate("date", "01992026")[0] is False          # month 99 -> invalid either way
