"""
Tests for the line detector and its exact-ground-truth metrics.

TDD on SYNTHETIC images with KNOWN line boxes first: if the detector misses a
line the test put there, nothing downstream of it can be trusted. The real-page
behaviour (and the neutralisation of the ink-saturated collapse) is exercised by
line_detect.py itself, which needs the corpus and the handheld photos.

    env -u PYTHONPATH -u VIRTUAL_ENV ./.venv/bin/python -m pytest \
        ml/tests/test_line_detect.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import line_detect as LD  # noqa: E402


# ── synthetic pages with KNOWN boxes ─────────────────────────────────────────

def blank_page(h=320, w=260, bg=245):
    return np.full((h, w), bg, np.uint8)


def with_bars(spec, x0=20, x1=240, ink=30, bg=245):
    """spec = [(y, height), ...] -> (gray, [(y0, y1), ...] the TRUE ink boxes)."""
    a = blank_page(spec[-1][0] + spec[-1][1] + 20 if spec else 40,
                   x1 + 20)
    a[:] = bg
    truth = []
    for y, h in spec:
        a[y:y + h, x0:x1] = ink
        truth.append((y, y + h))
    return a, truth


def pr(truth, pred, iou=0.5):
    """(recall, precision) at IoU >= iou, greedy one-to-one."""
    g2p, p2g, hit = LD.match_boxes(list(truth), list(pred), iou)
    rec = hit / len(truth) if truth else 1.0
    pre = hit / len(pred) if pred else 1.0
    return rec, pre, g2p


# ── the properties the detector must have ────────────────────────────────────

def test_n_lines_are_all_found_with_no_false_positives():
    """DRAW N lines -> recall AND precision must both be 1.0."""
    for n in (1, 2, 3, 6):
        spec = [(30 + i * 40, 14) for i in range(n)]
        gray, truth = with_bars(spec)
        pred = [b for b in LD.variant_bands(gray, set())]
        rec, pre, _ = pr(truth, pred)
        assert rec == 1.0, f"n={n}: missed a drawn line (recall {rec})"
        assert pre == 1.0, f"n={n}: invented a box (precision {pre})"
        assert len(pred) == n


def test_blank_page_yields_zero_detections_and_no_crash():
    gray = blank_page()
    assert LD.shipped_bands(gray) == []
    assert LD.variant_bands(gray, set()) == []
    assert LD.variant_bands(gray, {"cc_group"}) == []


def test_noise_only_page_yields_zero_detections():
    """High-frequency texture that is NOT ink (all pixels stay near paper)."""
    rng = np.random.default_rng(0)
    gray = rng.integers(236, 256, size=(320, 260), dtype=np.uint8)
    assert LD.variant_bands(gray, set()) == []
    assert LD.variant_bands(gray, {"ink_floor"}) == []


def test_single_line_page_yields_exactly_one_box():
    gray, truth = with_bars([(40, 14)])
    pred = LD.variant_bands(gray, set())
    assert len(pred) == 1, pred
    rec, pre, _ = pr(truth, pred)
    assert (rec, pre) == (1.0, 1.0)


def test_two_lines_two_pixels_apart_merge_is_documented_behaviour():
    """Either both lines are found, or the merge is REPORTED (one box spanning
    both) — never a silent split/ghost."""
    gray, truth = with_bars([(40, 14), (56, 14)])   # 2px gap
    pred = LD.variant_bands(gray, set())
    assert len(pred) in (1, 2), pred
    if len(pred) == 1:
        y0, y1 = pred[0]
        assert y0 <= truth[0][0] and y1 >= truth[1][1], \
            "merged band must span both lines it merged"
    else:
        rec, pre, _ = pr(truth, pred)
        assert rec == 1.0 and pre == 1.0


def test_shipped_banding_core_equals_variant_base():
    """The port must not drift: variant_bands with no flags IS the shipped
    bander on the same trimmed profile."""
    gray, _ = with_bars([(30, 14), (70, 12), (110, 16)])
    assert LD.variant_bands(gray, set()) == \
        [b for b in LD.shipped_bands(gray, use_locate=False)]


# ── the metric primitives ────────────────────────────────────────────────────

def test_iou_y_bounds_and_matching():
    assert LD.iou_y((0, 10), (0, 10)) == 1.0
    assert LD.iou_y((0, 10), (20, 30)) == 0.0
    assert LD.iou_y((0, 10), (5, 20)) == pytest.approx(5 / 20)
    g2p, p2g, hit = LD.match_boxes([(0, 10), (20, 30)], [(1, 11), (100, 110)])
    assert hit == 1 and g2p == {0: 0} and p2g == {0: 0}


def test_score_page_counts_mismatch_when_counts_differ():
    truth = [{"line": 0, "box": (0, 10), "text": "A"},
             {"line": 1, "box": (20, 30), "text": "B"}]
    s = LD.score_page(truth, [(0, 10)])
    assert s["n_gt"] == 2 and s["n_pred"] == 1 and s["hit"] == 1
    assert s["mismatch"] is True


# ── the paired verdict is compare_runs' ──────────────────────────────────────

def _pg(doc, prof="studio_clean", dt="prescription"):
    return {"doc": doc, "profile": prof, "doc_type": dt}


def test_paired_verdict_reads_improved_and_inconclusive():
    pages = [_pg("p%d" % i) for i in range(40)]
    # old: only even pages right; new: all right -> IMPROVED
    old = [LD._rec(p, [0], [] if i % 2 else [0]) for i, p in enumerate(pages)]
    new = [LD._rec(p, [0], [0]) for p in pages]
    v = LD.paired_verdict(old, new)
    assert v["verdict"].startswith("IMPROVED"), v
    assert LD.paired_verdict(old, old)["verdict"].startswith("INCONCLUSIVE")


def test_count_records_round_trip_via_compare_runs():
    counts = [("a", 3, 3), ("b", 2, 5)]
    recs = LD._count_records(counts)
    _so, _sn, rows = LD.CR.compare(LD._artifact(recs), LD._artifact(recs))
    row = next(r for r in rows if r["metric"] == "field_recovery@pipeline")
    assert row["n"] == 2 and row["k_old"] == 1