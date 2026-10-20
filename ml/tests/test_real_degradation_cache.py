"""
test_real_degradation_cache.py — the REAL-photo crop cache must be honest.

`ml/src/real_degradation_cache.py` turns the 193 hand-photographed express-
expense receipts into labelled (crop, label) pairs the trainer can consume, and
three properties decide whether it is a fix or a poison:

  1. LABELS ARE ALIGNED BY GEOMETRY. Transcript line i pairs with detected box i
     in reading order. A page whose counts do not match is DROPPED, never
     guessed and never labelled from the model's own read (that would be
     circular -- training the model to reproduce its own mistakes).
  2. THE SCHEMA IS THE TRAINER'S. images [N,32,maxw] uint8 with width <= 320,
     packed by `build_dataset.crop_to_array`, plus widths/labels/variants/
     splits/page -- exactly `render_corpus.write_cache`.
  3. SPLITS ARE BY DOCUMENT. No page id in both train and test; the held-out
     real pages stay out of the training split.

Nothing here loads the real photos or a checkpoint: the detector and the pages
are injected, so the suite stays fast and deterministic.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import real_degradation_cache as RDC  # noqa: E402
import realdata as R                  # noqa: E402

CROP_H = R.CROP_H
MAX_W = R.MAX_W


# ── 1. geometry-only label alignment ─────────────────────────────────────────

def test_align_labels_pairs_in_reading_order():
    boxes = [(0, 10, 0, 50), (20, 30, 0, 50), (40, 50, 0, 50)]
    lines = ["FIRST", "SECOND", "THIRD"]
    pairs, reason = RDC.align_labels(boxes, lines)
    assert reason is None
    assert [p[1] for p in pairs] == lines       # order preserved, top to bottom
    assert [p[0] for p in pairs] == boxes


def test_align_labels_drops_on_count_mismatch_never_guesses():
    boxes = [(0, 10, 0, 50), (20, 30, 0, 50)]
    lines = ["A", "B", "C", "D", "E"]
    pairs, reason = RDC.align_labels(boxes, lines)
    assert pairs is None
    assert "mismatch" in reason
    assert "2" in reason and "5" in reason
    # The other direction (a spurious band) is a mismatch too.
    assert RDC.align_labels(boxes * 3, lines)[0] is None
    # An empty transcript is a drop, not a page of unlabelled crops.
    assert RDC.align_labels(boxes, [])[0] is None


# ── 2. the stats helper on arrays with known contrast / sharpness ────────────

def test_crop_stats_reads_known_contrast_and_sharpness():
    # Half paper-bright, half ink-dark: density ~0.5, a hard edge.
    img = np.full((CROP_H, 64), 30, np.uint8)
    img[:CROP_H // 2] = 220
    ink, contrast, sharp = RDC.crop_stats(img)
    assert 0.4 < ink < 0.6, ink
    assert 170.0 < contrast < 200.0, contrast
    assert sharp > 0.0

    # A smooth left-to-right ramp has the same endpoints but almost no gradient
    # energy, so its Laplacian variance is far lower than the hard edge's.
    ramp = np.tile(np.linspace(30, 220, 64, dtype=np.uint8), (CROP_H, 1))
    assert RDC.crop_stats(ramp)[2] < sharp

    # Blurring the hard edge lowers sharpness without changing its polarity.
    import cv2
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    assert RDC.crop_stats(blurred)[2] < sharp

    # A flat field carries no ink and no edges.
    flat = np.full((CROP_H, 64), 128, np.uint8)
    assert RDC.crop_stats(flat)[2] == 0.0


def test_stats_of_aggregates_mean_and_median():
    a = np.full((CROP_H, 64), 30, np.uint8)
    a[8:] = 220
    b = np.full((CROP_H, 64), 200, np.uint8)
    b[:20] = 40
    s = RDC.stats_of([a, b])
    assert s["n"] == 2
    assert 0.0 < s["ink_mean"] < 1.0
    assert s["contrast_mean"] > 100.0
    assert RDC.stats_of([])["ink_mean"] == 0.0


# ── 3/4. build a cache from injected pages: schema + drop rule ───────────────

def _page(tmp_path, name, n_lines, n_boxes, h, w=200):
    """A fake receipt photo plus the boxes the injected detector will report."""
    a = np.full((h, w), 240, np.uint8)
    boxes, y = [], 4
    for _ in range(n_boxes):
        a[y:y + 18, 8:150] = 20
        boxes.append((y, y + 18, 4, 160))
        y += 26
    p = tmp_path / f"{name}-receipt.jpg"
    Image.fromarray(a).save(p)
    lines = [f"LINE {i} 100" for i in range(n_lines)]
    return (name, p, lines), boxes


def _make_pages(tmp_path, n_good=10):
    pages, by_shape = [], {}
    heights = []
    for i in range(n_good):
        h = 100 + i * 5
        page, boxes = _page(tmp_path, f"9{i:03d}", 3, 3, h)
        pages.append(page)
        by_shape[(h, 200)] = boxes
        heights.append(h)
    # One page whose transcript (5 lines) cannot be aligned to 3 boxes: it must
    # be dropped, not guessed.
    bad_h = 900
    bad, _ = _page(tmp_path, "9999", 5, 3, bad_h)
    pages.append(bad)
    by_shape[(bad_h, 200)] = [(4, 22, 4, 160), (30, 48, 4, 160), (56, 74, 4, 160)]

    def detect(gray):
        return by_shape[gray.shape]
    return pages, detect


def test_build_cache_schema_and_split_disjointness(tmp_path):
    pages, detect = _make_pages(tmp_path)
    out = tmp_path / "crops_train.npz"
    rep = RDC.build_cache(pages, out_path=out, seed=0, detect=detect,
                          verbose=False)

    d = np.load(out, allow_pickle=True)
    # Exactly the trainer's cache schema (render_corpus.write_cache).
    assert set(d.files) == {"images", "widths", "labels", "variants",
                            "splits", "page"}, d.files
    images, widths = d["images"], d["widths"]
    assert images.dtype == np.uint8
    assert images.shape[1] == CROP_H
    assert int(widths.max()) <= MAX_W
    assert len(d["labels"]) == images.shape[0] == len(widths)
    assert set(str(v) for v in d["variants"]) == {"real"}
    # Every packed row is zero-padded beyond its own width.
    for i, w in enumerate(widths):
        assert np.all(images[i, :, w:] == 0)
    # Labels are charset-clean, so the model can actually emit them.
    for lab in d["labels"]:
        assert set(str(lab)) <= set(R.CHARSET), lab

    # DROP-ON-MISMATCH: the 5-line vs 3-box page is absent and reported.
    ids = list(str(x) for x in d["page"])
    assert "9999" not in ids
    dropped = {x["page"] for x in rep["dropped"]}
    assert "9999" in dropped
    assert any("mismatch" in x["reason"] for x in rep["dropped"]
               if x["page"] == "9999")

    # SPLIT DISJOINTNESS by DOCUMENT.
    splits = np.array([str(x) for x in d["splits"]])
    page = np.array([str(x) for x in d["page"]])
    train = set(page[splits == "train"])
    test = set(page[splits == "test"])
    val = set(page[splits == "val"])
    assert train and test, "both splits must be non-empty for this to mean anything"
    assert not (train & test), "a page appears in BOTH train and test"
    assert not (train & val)
    # Held-out real pages are excluded from the training split (recommended file).
    held_out = val | test
    assert not (train & held_out)
    assert held_out, "no held-out real pages -- the split is not a split"


def test_page_crops_drops_all_illegible_page(tmp_path):
    # A single transcript line of punctuation only -> train_label is empty.
    page, boxes = _page(tmp_path, "8888", 1, 1, 120)
    page = (page[0], page[1], ["-- :: --"])
    res, reason, n = RDC.page_crops(page, detect=lambda gray: boxes)
    assert res is None and "legible" in reason and n == 1


# ── 5. nearest-profile helper ────────────────────────────────────────────────

def test_nearest_profile_picks_the_closest_vector():
    vecs = {
        "studio_clean": (0.05, 200.0, 900.0),
        "low_light": (0.30, 40.0, 60.0),
        "handheld_heavy": (0.20, 90.0, 200.0),
    }
    name, dist = RDC.nearest_profile((0.06, 195.0, 880.0), vecs)
    assert name == "studio_clean"
    assert dist >= 0.0
    name2, _ = RDC.nearest_profile((0.31, 38.0, 55.0), vecs)
    assert name2 == "low_light"
