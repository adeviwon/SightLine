"""
Lane N tests — the SPEAK/REFUSE policy.

The policy exists because a confidently spoken wrong dosage is worse than a
refusal. Each test here pins one rule from the module docstring, and the last
block pins the two things that would silently rot: the regex token table must
cover evaluate.FIELDS, and the policy's "too short to speak" floor must stay the
scorer's own resolution.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import field_policy as P  # noqa: E402


# ── helpers: a spot record, and a decision lookup ─────────────────────────────

def spot(value, kind="general", ok=True, reason="ok"):
    return {"value": value, "ok": ok, "kind": kind, "reason": reason,
            "provenance": "same-line-right"}


def by_field(decisions):
    return {d["field"]: d for d in decisions}


# ── R4: agreement speaks ──────────────────────────────────────────────────────

def test_agreement_between_paths_speaks():
    """Same value, differently written, from each path -> one spoken decision."""
    got = P.decide({"dosage": "500mg"}, {"dosage": spot("500MG", "dosage")})
    assert len(got) == 1, got                        # agreement MERGES the two
    assert got[0]["decision"] == "speak"
    assert got[0]["source"] == "both"
    assert "agree" in got[0]["why"]


def test_agreement_survives_formatting_for_typed_kinds():
    """'40 11 04' and '40-11-04' are the same sort code."""
    got = P.decide({"sort_code": "40-11-04"}, {"sort_code": spot("40 11 04", "sort_code")})
    assert got[0]["decision"] == "speak" and got[0]["source"] == "both", got


def test_dose_inside_a_medication_value_counts_as_agreement():
    """The spot path reads 'AMOXICILLIN 500MG' where the regex read '500mg'.

    Different field names, same dose: that is corroboration, not silence.
    """
    got = by_field(P.decide({"dosage": ["500mg", "400mg"]},
                            {"medication": spot("AMOXICILLIN 500MG")}))
    assert got["dosage"]["decision"] == "speak"
    assert got["dosage"]["source"] == "both"


# ── R3: contradiction refuses, never picks a winner ──────────────────────────

def test_contradiction_refuses_both_sides_with_the_strings():
    got = P.decide({"date": "01102026"}, {"date": spot("01102027", "date")})
    assert len(got) == 2
    assert {d["decision"] for d in got} == {"refuse"}
    why = got[0]["why"]
    assert "01102026" in why and "01102027" in why, why
    assert "disagree" in why


def test_contradiction_refuses_even_when_each_side_validates():
    """Both dates are real calendar dates and both validate — still refused."""
    got = P.decide({"date": "01/10/2026"}, {"date": spot("02102026", "date")})
    assert {d["decision"] for d in got} == {"refuse"}


# ── R4: validated single-path evidence speaks (critical) ─────────────────────

def test_validated_single_path_critical_speaks():
    got = P.decide({}, {"date_of_birth": spot("04021979", "date")})
    assert got[0]["decision"] == "speak" and got[0]["source"] == "spot"
    assert "other path had no value" in got[0]["why"]


def test_strict_critical_requires_agreement():
    got = P.decide({}, {"date_of_birth": spot("04021979", "date")},
                   strict_critical=True)
    assert got[0]["decision"] == "refuse"
    assert "agreement required" in got[0]["why"]


# ── R2: failed validation is never spoken, whatever the other path says ──────

def test_failed_validation_refuses_even_when_both_paths_agree():
    got = P.decide({"date": "31/02/2026"}, {"date": spot("31022026", "date", ok=False,
                                                          reason="not a calendar date")})
    assert {d["decision"] for d in got} == {"refuse"}
    assert all("calendar date" in d["why"] for d in got), got


def test_regex_path_is_validated_even_though_it_arrives_unvalidated():
    """The regex path has no validator; the policy holds it to the kind's rule."""
    got = P.decide({"account_number": "123"}, {})
    assert got[0]["decision"] == "refuse"
    assert "digits" in got[0]["why"], got


# ── R5: non-critical kinds speak on one path ─────────────────────────────────

def test_non_critical_speaks_on_validated_single_path():
    got = P.decide({}, {"patient": spot("JANE DOE")})
    assert got[0]["decision"] == "speak"
    assert "label path" in got[0]["why"]


def test_vocabulary_gate_refuses_another_types_label():
    """CLAUSE is a legal label; on a prescription it is a false positive."""
    got = P.decide({}, {"clause": spot("DROWSINESS")}, doc_type="prescription")
    assert got[0]["decision"] == "refuse"
    assert "vocabulary" in got[0]["why"]


def test_vocabulary_gate_is_off_without_a_doc_type():
    """No doc type -> no claim about vocabulary; the field is not refused for it."""
    got = P.decide({}, {"clause": spot("DROWSINESS")})
    assert got[0]["decision"] == "speak"


def test_substance_floor_refuses_a_truncated_fragment():
    got = P.decide({}, {"warning": spot("MAY")}, doc_type="prescription")
    assert got[0]["decision"] == "refuse"
    assert "too short" in got[0]["why"]


def test_free_text_can_be_refused_outright():
    got = P.decide({}, {"patient": spot("JANE DOE")}, speak_free_text=False)
    assert got[0]["decision"] == "refuse" and "free text" in got[0]["why"]


# ── R1: an empty value is never spoken ──────────────────────────────────────

def test_label_without_a_value_is_refused_with_that_reason():
    got = P.decide({}, {"pharmacy": spot("", ok=False,
                                         reason="no value found after label")})
    assert got[0]["decision"] == "refuse"
    assert "no value present" in got[0]["why"]


# ── R6: determinism, no duplicates, and the degenerate page ─────────────────

def test_deterministic_order_independent_of_input_order():
    regex = {"dosage": ["500mg", "400mg"], "date": "01102026"}
    spots = {"date_of_birth": spot("04021979", "date"),
             "medication": spot("AMOXICILLIN 500MG"),
             "pharmacy": spot("NORTHGATE")}
    a = P.decide(regex, spots, doc_type="prescription")
    b = P.decide(dict(reversed(list(regex.items()))),
                 dict(reversed(list(spots.items()))), doc_type="prescription")
    assert a == b
    assert a == P.decide(regex, spots, doc_type="prescription")   # stable


def test_no_field_and_value_is_spoken_twice():
    got = P.decide({"dosage": ["500mg", "500MG"]},
                   {"dosage": spot("500 mg", "dosage")})
    spoken = [d["value"] for d in got if d["decision"] == "speak"]
    assert len(spoken) == 1, got
    keys = [(d["field"], d["value"].lower().replace(" ", "")) for d in got]
    assert len(keys) == len(set(keys))


def test_two_distinct_doses_are_both_spoken():
    got = P.decide({"dosage": ["500mg", "400mg"]}, {})
    assert len([d for d in got if d["decision"] == "speak"]) == 2


def test_both_paths_empty_speaks_nothing_and_does_not_crash():
    assert P.decide({}, {}) == []
    assert P.decide(None, None) == []
    assert P.spoken(P.decide({}, {})) == []


def test_never_speaks_an_empty_string():
    got = P.decide({"dosage": ""}, {"patient": spot("   ")})
    assert got and all(d["decision"] == "refuse" for d in got)


# ── the adapters and the limits that must not drift ─────────────────────────

def test_from_regex_tokens_groups_two_doses_under_one_field():
    assert P.from_regex_tokens("prescription", ["500mg", "400mg"]) == \
        {"dosage": ["500mg", "400mg"]}
    assert P.from_regex_tokens("banking", ["40-11-04"]) == {"sort_code": "40-11-04"}


def test_regex_token_table_covers_every_token_evaluate_ships():
    """A new FIELDS token without a kind here would be spoken untyped."""
    import evaluate as E
    for doc_type, tokens in E.FIELDS.items():
        for tok in tokens:
            assert tok in P.REGEX_TOKENS.get(doc_type, {}), (doc_type, tok)


def test_speak_floor_matches_the_scorers_resolution():
    import eval_field_truth as ET
    assert P.MIN_SPEAK_CHARS == ET.MIN_SUBSTRING


@pytest.mark.parametrize("kind,value,expect", [
    ("dosage", "AMOXICILLIN 500MG", {"500mg"}),
    ("date", "01/10/2026", {"01102026"}),
    ("date", "01102026", {"01102026"}),
    ("money", "GBP 1,204.33", {"1204.33"}),
    ("sort_code", "40-11-04", {"401104"}),
    ("account", "40218877", {"40218877"}),
    ("date", "NOT A DATE", set()),
    ("dosage", "DR A AHMED", set()),
])
def test_canonical_forms(kind, value, expect):
    assert P.canonical(kind, value) == expect
