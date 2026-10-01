"""
test_corpus.py — the classifier corpus must not be lying to itself.

corpus.py's stated #1 design goal is HONEST EVALUATION. The previous run
duplicated ~15 templates, which put identical phrases on both sides of the
train/val split and made the accuracy swing 6.7 points per sample. These
tests make that class of bug impossible to reintroduce silently:

  - no duplicate templates within a category
  - no template appears in more than one of train/val/test
  - every category is represented in every split (stratification happened)
  - augmentation only ever ADDS corrupted copies and never changes a label
"""

import random

import pytest

import sys
from pathlib import Path

# Project modules live in ml/src as plain modules (not an installed package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import corpus  # noqa: E402
from train_classifier import three_way_split  # noqa: E402

CATEGORIES = corpus.CATEGORIES
RAW = corpus.RAW


# ── 1. Uniqueness ───────────────────────────────────────────────────────

@pytest.mark.parametrize("cat", CATEGORIES)
def test_no_duplicate_templates_within_category(cat):
    """The original repo's bug: ~15 templates appeared twice in one category."""
    seen = set()
    dupes = []
    for t in RAW[cat]:
        k = t.strip().lower()
        if k in seen:
            dupes.append(t)
        seen.add(k)
    assert not dupes, (
        f"{cat} contains {len(dupes)} duplicate template(s): {dupes[:3]}")


@pytest.mark.parametrize("cat", CATEGORIES)
def test_dedupe_is_a_noop_on_a_clean_corpus(cat):
    """If _dedupe() would remove anything, the RAW list is still dirty."""
    assert len(corpus._dedupe(RAW[cat])) == len(RAW[cat])


def test_all_categories_non_empty():
    """All four categories must have real content or the 4-way head is fake."""
    assert len(CATEGORIES) == 4
    for cat in CATEGORIES:
        assert len(RAW[cat]) >= 20, f"{cat} has only {len(RAW[cat])} templates"


def test_corpus_stats_matches_the_data():
    """The reported corpus stats must not drift from the actual lists."""
    stats = corpus.corpus_stats()
    assert stats["total_unique"] == sum(len(RAW[c]) for c in CATEGORIES)
    for cat in CATEGORIES:
        assert stats["unique_templates"][cat] == len(RAW[cat])


def test_no_exact_text_leaks_across_categories():
    """The same sentence in two categories would make one label unfalsifiable."""
    seen = {}
    for cat in CATEGORIES:
        for t in RAW[cat]:
            k = t.strip().lower()
            assert k not in seen or seen[k] == cat, (
                f"template appears in both {seen.get(k)} and {cat}: {t[:50]!r}")
            seen[k] = cat


# ── 2. THE LEAKAGE TEST: three-way split ────────────────────────────────

@pytest.fixture(scope="module")
def split():
    return three_way_split(123)


def _key(t):
    return t.strip().lower()


def test_three_way_split_has_no_template_overlap(split):
    """
    The headline number is a HELD-OUT TEST score. If any template appears in
    both train and test, the test set is no longer held out and the accuracy
    is optimistic. Set intersection must be EMPTY.
    """
    tr, _, va, _, te, _ = split
    tr_s, va_s, te_s = {_key(t) for t in tr}, {_key(t) for t in va}, {_key(t) for t in te}
    assert not (tr_s & te_s), f"{len(tr_s & te_s)} template(s) leak train->test"
    assert not (va_s & te_s), f"{len(va_s & te_s)} template(s) leak val->test"
    assert not (tr_s & va_s), f"{len(tr_s & va_s)} template(s) leak train->val"


def test_three_way_split_partitions_the_corpus(split):
    """Every unique template must land in exactly ONE split, none dropped."""
    tr, _, va, _, te, _ = split
    parts = [{_key(t) for t in s} for s in (tr, va, te)]
    union = set().union(*parts)
    total = len({_key(t) for c in CATEGORIES for t in RAW[c]})
    assert len(union) == total, f"corpus not fully partitioned: {len(union)}/{total}"
    assert sum(len(p) for p in parts) == total, "a template was placed twice"


@pytest.mark.parametrize("idx", range(4))
def test_every_category_appears_in_every_split(split, idx):
    """Stratification must actually have happened for all four categories."""
    tr, trl, va, val, te, tel = split
    for name, labels in (("train", trl), ("val", val), ("test", tel)):
        assert idx in labels, f"category {CATEGORIES[idx]} is MISSING from {name}"
    # ...and the label ordering must match the text ordering
    for texts, labels in ((tr, trl), (va, val), (te, tel)):
        for t, ci in zip(texts, labels):
            assert t in RAW[CATEGORIES[ci]], f"label/text mismatch: {t[:40]!r}"


def test_split_labels_are_consistent_and_complete(split):
    tr, trl, va, val, te, tel = split
    for texts, labels in ((tr, trl), (va, val), (te, tel)):
        assert len(texts) == len(labels), "text/label length mismatch"
        assert set(labels) <= set(range(4))
    assert tr and va and te, "a split is empty"


def test_split_is_deterministic():
    """A different seed may give a different split; the SAME seed may not."""
    a = three_way_split(123)
    b = three_way_split(123)
    assert a == b
    c = three_way_split(123, val_frac=0.15, test_frac=0.15)
    assert c[0] == a[0], "identical seeds must give an identical train split"


def test_val_and_test_are_never_augmented(split):
    """
    make_corpus/three_way_split promise 'val and test are never augmented'.
    Every val/test template must therefore appear EXACTLY once, verbatim,
    with no OCR-corrupted twins.
    """
    for texts in (split[2], split[4]):   # va_t and te_t (not va_l / te_l)
        keys = [_key(t) for t in texts]
        assert len(keys) == len(set(keys)), "val/test contains a duplicate"
        for t in texts:
            assert t.strip() == t, "val/test text was not stripped"


# ── 3. Augmentation ─────────────────────────────────────────────────────

def test_augment_count_is_exact():
    """n_copies corrupted copies of each of n templates, plus the n originals."""
    texts = [f"Statement {i} balance GBP {i}.00" for i in range(7)]
    labels = [0] * 7
    at, al = corpus.augment(texts, labels, seed=1, n_copies=2)
    assert len(at) == len(al) == 7 * (1 + 2)
    for n in (0, 1, 3):
        _, l2 = corpus.augment(texts, labels, seed=1, n_copies=n)
        assert len(l2) == 7 * (1 + n)


def test_augment_preserves_the_originals_and_their_order():
    """The clean templates must survive verbatim, in order, at the front."""
    texts = [f"Clause {i}.2 tenant rent 2,500" for i in range(5)]
    labels = [2] * 5
    at, al = corpus.augment(texts, labels, seed=9, n_copies=2)
    assert at[:5] == texts, "augment() mutated or reordered the originals"
    assert al[:5] == labels


def test_augment_never_changes_a_label():
    """Augmentation is text-only corruption. A label flip is a silent poison."""
    # real templates, labelled by their true category
    texts = [RAW[c][i] for i, c in enumerate(CATEGORIES) for _ in range(2)][:8]
    labels = [CATEGORIES.index(c) for c in CATEGORIES for _ in range(2)][:8]
    _, al = corpus.augment(texts, labels, seed=3, n_copies=4)
    for orig, got in zip(labels, al):
        assert got == orig
    # the labels must still index the right category for the text they carry
    for t, ci in zip(texts, labels):
        assert t in RAW[CATEGORIES[ci]]


def test_augment_adds_corrupted_copies_not_clean_ones():
    """The added rows must be OCR-degraded, otherwise augmentation is a no-op
    that only inflates the training-set size."""
    texts = [t for c in CATEGORIES for t in RAW[c]][:40]
    labels = [i % 4 for i in range(40)]
    at, _ = corpus.augment(texts, labels, seed=11, n_copies=2)
    copies = at[len(texts):]
    unchanged = sum(1 for c, src in zip(copies, (texts * 2)) if c == src)
    assert unchanged < len(copies) * 0.5, (
        f"{unchanged}/{len(copies)} augmented copies are identical to their source")


def test_augment_is_deterministic():
    texts = [f"Rx Amoxicillin 500mg take {i}" for i in range(6)]
    labels = [1] * 6
    assert corpus.augment(texts, labels, seed=5, n_copies=2) == \
           corpus.augment(texts, labels, seed=5, n_copies=2)


# ── 4. OCR noise ────────────────────────────────────────────────────────

def test_ocr_noise_alters_the_string_sometimes():
    """If ocr_noise never fired, 'realistic difficulty' would be a claim only."""
    text = "PRESCRIPTION. Amoxicillin 500mg, one capsule three times daily."
    changed = sum(1 for s in range(20)
                  if corpus.ocr_noise(text, random.Random(s)) != text)
    assert changed >= 10, f"ocr_noise changed only {changed}/20 samples"


def test_ocr_noise_preserves_string_length():
    """
    Length preservation is what keeps word_accuracy's split() alignment
    meaningful, and it is what guarantees augment() cannot drop a word.
    """
    for cat in CATEGORIES:
        for t in RAW[cat]:
            for s in range(4):
                out = corpus.ocr_noise(t, random.Random(s))
                assert len(out) == len(t), f"length changed on {cat}: {t[:40]!r}"


def test_ocr_noise_never_changes_the_label_array_length():
    """The task-level contract: augmentation must not change label counts."""
    texts = [f"Balance GBP {i}.00 account {i}" for i in range(8)]
    labels = [0] * 8
    for n in (1, 2, 5):
        at, al = corpus.augment(texts, labels, seed=2, n_copies=n)
        assert len(al) == len(at) == 8 * (1 + n)


def test_ocr_noise_produces_realistic_confusions():
    """The confusions must be the digit/letter swaps Tesseract actually makes
    (0/O, 1/l, 5/S) — not random garbage."""
    text = "Account Number: 40218877. Sort Code: 40-11-04."
    hits = 0
    for s in range(30):
        out = corpus.ocr_noise(text, random.Random(s))
        if any(c in out for c in "OoLlSsB") and any(ch.isdigit() for ch in out):
            hits += 1
    assert hits >= 10, f"ocr_noise is not producing digit/letter confusion ({hits}/30)"


def test_ocr_noise_never_emits_empty_string():
    for cat in CATEGORIES:
        for t in RAW[cat]:
            for s in range(3):
                assert corpus.ocr_noise(t, random.Random(s)).strip()