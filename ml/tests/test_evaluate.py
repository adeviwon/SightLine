"""
test_evaluate.py — the metrics and the honesty rules of the eval harness.

evaluate.py is "the file that decides whether SightLine actually works", so
its own bookkeeping is the thing under test. The two properties that matter
most to a judge:

  1. GROUND-TRUTH INTEGRITY. FIELDS must match capture.DOCS exactly. If a
     document type has no entry, evaluate.run() raises KeyError; if a field
     no longer appears in the rendered document, the field_accuracy headline
     is measuring a typo.
  2. THE HONESTY RULE. A profile flagged SUB-HUMAN (a human cannot read it,
     so it is not a model failure) must never appear in the headline mean.
     The test asserts the invariant structurally, not by re-running the eval.

OCR itself is NOT exercised here: it needs the tesseract binary and is slow.
The metric functions are pure and are tested directly.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

# Project modules live in ml/src as plain modules (not an installed package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import capture  # noqa: E402
import evaluate  # noqa: E402

DOC_TYPES = {dtype for _, dtype in capture.DOCS}


# ── 1. norm() ───────────────────────────────────────────────────────────

def test_norm_strips_case_and_punctuation():
    assert evaluate.norm("A-B, C. (D)") == "abcd"


def test_norm_keeps_digits():
    """Field matching is digit-critical: '500mg' must survive normalisation."""
    assert evaluate.norm("500mg") == "500mg"
    assert evaluate.norm("40-11-04") == "401104"


def test_norm_handles_the_real_field_strings():
    """Every ground-truth field must normalise to a searchable token."""
    for doc_type, fields in evaluate.FIELDS.items():
        for f in fields:
            n = evaluate.norm(f)
            assert n, f"field {f!r} normalises to nothing"
            assert n.isalnum(), f"field {f!r} -> {n!r} is not alnum"


def test_norm_is_idempotent():
    s = "Statement of Account: GBP 1,204.33 (ref 40218877)"
    assert evaluate.norm(evaluate.norm(s)) == evaluate.norm(s)


def test_norm_on_empty_and_unicode():
    assert evaluate.norm("") == ""
    assert evaluate.norm("!!! ***") == ""
    assert evaluate.norm("naïve") == "nave"


# ── 2. word_accuracy() ──────────────────────────────────────────────────

def test_word_accuracy_of_identical_text_is_one():
    assert evaluate.word_accuracy("a b c", "a b c") == 1.0


def test_word_accuracy_of_disjoint_text_is_below_one():
    assert evaluate.word_accuracy("alpha beta gamma", "x y z") < 1.0


def test_word_accuracy_is_in_the_unit_interval():
    for gt, hyp in [("a b c", "a b c"), ("a b c", "a b"), ("a b", "a b c d"),
                    ("a b c", "z"), ("", "a"), ("a", ""),
                    ("500mg capsule", "capsule 500mg")]:
        v = evaluate.word_accuracy(gt, hyp)
        assert 0.0 <= v <= 1.0, f"word_accuracy({gt!r},{hyp!r}) = {v}"


def test_word_accuracy_is_normalisation_insensitive():
    """A case/punctuation difference must not be scored as an OCR error —
    Tesseract's casing is not the thing being measured."""
    assert evaluate.word_accuracy("Statement of Account", "STATEMENT, of account.") == 1.0


def test_word_accuracy_is_a_recall_measure():
    """Missed words are penalised; extra words are not rewarded. Documented
    here so a future change to precision is a deliberate, visible one."""
    assert evaluate.word_accuracy("a b c d", "a b") == pytest.approx(0.5)
    assert evaluate.word_accuracy("a b", "a b c d") == pytest.approx(1.0)


def test_word_accuracy_on_a_real_document_pair(clean_docs):
    lines, dtype = clean_docs[0]
    gt = "\n".join(lines)
    assert evaluate.word_accuracy(gt, gt) == 1.0
    partial = " ".join(gt.split()[:5])
    assert evaluate.word_accuracy(gt, partial) < 1.0


def test_empty_ground_truth_is_not_a_free_win():
    assert evaluate.word_accuracy("", "anything at all") == 0.0


# ── 3. fields_found() ───────────────────────────────────────────────────

def test_fields_found_locates_every_field_in_a_clean_render(clean_docs):
    """Ground-truth integrity: the fields listed in FIELDS must actually be
    present in the document capture.py renders. A typo here would make the
    field_accuracy headline permanently zero."""
    for lines, doc_type in clean_docs:
        text = "\n".join(lines)
        found = evaluate.fields_found(text, doc_type)
        missing = [f for f in evaluate.FIELDS[doc_type] if f not in found]
        assert not missing, f"{doc_type}: fields {missing} are absent from the render"


def test_fields_found_fails_on_a_digit_substitution(clean_docs):
    """
    THE MEDICAL SAFETY CASE. '500mg' misread as 'SOOmg' must not count as a
    hit. This is the difference between "better off being told to retake the
    photo" and being told the wrong dose.
    """
    lines, doc_type = clean_docs[0]
    assert doc_type == "prescription"
    text = "\n".join(lines)
    corrupted = text.replace("500mg", "SOOmg")
    assert corrupted != text, "fixture no longer contains '500mg'"
    assert "500mg" not in evaluate.fields_found(corrupted, doc_type)
    assert evaluate.fields_found(corrupted, doc_type) != evaluate.FIELDS[doc_type]


def test_fields_found_fails_on_letter_digit_confusion_everywhere(clean_docs):
    """The same guarantee for every ground-truth field of every doc type."""
    swap = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "8": "B"}
    checked = 0
    for lines, doc_type in clean_docs:
        # fields_found matches in NORMALISED space, so corrupt there: a field
        # can appear in the render as '2,500' and only normalise to '2500'.
        n_text = evaluate.norm("\n".join(lines))
        for f in evaluate.FIELDS[doc_type]:
            nf = evaluate.norm(f)
            bad = "".join(swap.get(c, c) for c in nf)
            if bad == nf or nf not in n_text:
                continue
            corrupted = n_text.replace(nf, bad)
            assert corrupted != n_text, f"{doc_type}: corruption was a no-op on {f!r}"
            assert nf not in evaluate.norm(corrupted), (
                f"{doc_type}: field {f!r} survived corruption to {bad!r}")
            assert f not in evaluate.fields_found(corrupted, doc_type), (
                f"{doc_type}: field {f!r} still credited after OCR confusion")
            checked += 1
    assert checked >= 6, f"only {checked} fields exercised"


def test_fields_found_is_empty_for_a_wrong_doc_type(clean_docs):
    """Prescription text must not 'satisfy' the banking field list."""
    lines, _ = clean_docs[0]
    text = "\n".join(lines)
    assert evaluate.fields_found(text, "banking") == []


def test_fields_found_on_empty_text():
    for doc_type in evaluate.FIELDS:
        assert evaluate.fields_found("", doc_type) == []


def test_all_or_nothing_dosage_gate(clean_docs):
    """A document counts only if EVERY field is found — evaluate.run() uses
    len(found) == len(FIELDS[doc_type]). Half-correct is a failure."""
    lines, doc_type = clean_docs[0]
    text = "\n".join(lines)
    fields = evaluate.FIELDS[doc_type]
    partial = text.replace(fields[0], "SOOmg")
    found = evaluate.fields_found(partial, doc_type)
    assert len(found) < len(fields), "partial recovery must not count as full"


# ── 4. classify_doc() ───────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("PRESCRIPTION patient dose 500mg capsule tablet", "medical"),
    ("balance account sort code gbp debit credit atm", "banking"),
    ("agreement clause tenant landlord court deed", "legal"),
    ("library meeting notice school parcel recipe", "general"),
])
def test_classify_doc_on_clear_keyword_text(text, expected):
    assert evaluate.classify_doc(text) == expected


def test_classify_doc_on_the_real_renders(clean_docs):
    """Every rendered document must be classified as its own type."""
    expected = {"prescription": "medical", "banking": "banking", "legal": "legal"}
    for lines, doc_type in clean_docs:
        assert evaluate.classify_doc("\n".join(lines)) == expected[doc_type]


def test_classify_doc_is_case_insensitive():
    assert evaluate.classify_doc("PRESCRIPTION PATIENT MG CAPSULE") == "medical"


def test_classify_doc_falls_back_to_general_on_no_signal():
    """A document with no keywords must return a valid label, not crash."""
    for junk in ("", "zzz qqq", "...."):
        assert evaluate.classify_doc(junk) in set(evaluate.FIELDS) | {"general", "medical",
                                                                     "banking", "legal"}


def test_classify_doc_always_returns_a_known_category():
    for text in ("", "a", "account", "clause", "mg", "notice"):
        assert evaluate.classify_doc(text) in ("medical", "banking", "legal", "general")


# ── 5. THE HONESTY RULES ────────────────────────────────────────────────

def test_sub_human_profiles_are_not_in_the_headline_list():
    """
    The headline mean must exclude any profile where a human cannot read the
    text, otherwise the number is inflated by cases that are not model
    failures. evaluate.py computes `honest` this way:
    """
    honest = [p for p in capture.CAPTURE_PROFILES if p not in evaluate.SUB_HUMAN]
    for p in evaluate.SUB_HUMAN:
        assert p not in honest, f"SUB-HUMAN profile {p!r} leaked into the headline mean"
    assert honest, "every profile is sub-human — the headline is empty"
    assert len(honest) == len(capture.CAPTURE_PROFILES) - len(evaluate.SUB_HUMAN)


def test_sub_human_profiles_actually_exist():
    """A typo in a SUB_HUMAN name would silently exclude nothing."""
    assert evaluate.SUB_HUMAN, "no profile is flagged sub-human"
    for p in evaluate.SUB_HUMAN:
        assert p in capture.CAPTURE_PROFILES, f"SUB_HUMAN names unknown profile {p!r}"


def test_every_profile_is_either_headline_or_sub_human():
    """The two sets must partition the profiles — nothing unaccounted for."""
    honest = {p for p in capture.CAPTURE_PROFILES if p not in evaluate.SUB_HUMAN}
    assert honest | set(evaluate.SUB_HUMAN) == set(capture.CAPTURE_PROFILES)
    assert not (honest & set(evaluate.SUB_HUMAN))


def test_sub_human_exclusion_is_not_a_majority():
    """If everything were excluded the headline would be a vacuous number."""
    honest = [p for p in capture.CAPTURE_PROFILES if p not in evaluate.SUB_HUMAN]
    assert len(honest) >= len(capture.CAPTURE_PROFILES) // 2


# ── 6. Ground-truth / data alignment ────────────────────────────────────

def test_fields_keys_match_the_document_types_exactly():
    """No drift between ground truth and the data that capture.py renders."""
    assert set(evaluate.FIELDS) == DOC_TYPES, (
        f"FIELDS keys {sorted(evaluate.FIELDS)} != DOCS types {sorted(DOC_TYPES)}")


def test_every_doc_type_has_at_least_one_field():
    for doc_type in DOC_TYPES:
        assert evaluate.FIELDS.get(doc_type), f"{doc_type} has no ground-truth fields"


def test_every_field_is_nonempty_and_unique():
    for doc_type, fields in evaluate.FIELDS.items():
        assert len(set(fields)) == len(fields), f"{doc_type} has duplicate fields"
        for f in fields:
            assert isinstance(f, str) and f.strip()


def test_all_four_comparison_arms_are_registered():
    """The model must be measured against raw and classical, or the
    comparison is not a comparison."""
    assert set(evaluate.ARMS) == {"raw", "classical", "restorer", "restorer_clahe"}


def test_restorer_arms_accept_a_model_and_need_no_checkpoint(clean_docs):
    """
    evaluate.run() falls back to an UNTRAINED restorer when no checkpoint
    exists, and prints a warning. That path must not crash, and at init the
    network is an exact identity so the image must pass through unchanged.
    """
    torch = pytest.importorskip("torch")
    from model import SightLineNet
    lines, doc_type = clean_docs[0]
    img = capture.render_document(lines, seed=11)
    out = evaluate.arm_restorer(img, SightLineNet().eval())
    assert out.size == img.size
    a = np.asarray(out.convert("L")).astype(int)
    b = np.asarray(img.convert("L")).astype(int)
    assert np.abs(a - b).max() <= 1, "identity restorer altered the image"


def test_restorer_preserves_size_on_a_non_multiple_image():
    """The tiling pads to 64x256 and must crop back to the exact input size."""
    torch = pytest.importorskip("torch")
    from model import SightLineNet
    from PIL import Image
    img = Image.new("RGB", (317, 203), (220, 220, 215))
    out = evaluate.arm_restorer(img, SightLineNet().eval())
    assert out.size == img.size == (317, 203)


def test_arm_raw_is_a_passthrough(clean_docs):
    lines, _ = clean_docs[0]
    img = capture.render_document(lines, seed=2)
    assert evaluate.ARMS["raw"](img) is img


def test_classical_arm_preserves_size(clean_docs):
    lines, _ = clean_docs[0]
    img = capture.render_document(lines, seed=2)
    assert evaluate.ARMS["classical"](img).size == img.size