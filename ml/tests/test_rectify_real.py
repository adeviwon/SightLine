"""
Tests for ml/src/rectify_real.py.

The four required behaviours:
  1. the homography -> rectangle mapping (a known trapezoid warps to the
     expected rectangle),
  2. the exact transcript-count rule (a page whose counts differ is DROPPED,
     never trimmed),
  3. the written cache's schema / shape / width bounds,
  4. train/held-out document disjointness.

All pure cv2/numpy/geometry -- no model, no training, no GPU.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rectify_real as RR  # noqa: E402  (import error here = file missing)
import build_dataset as B  # noqa: E402


# ── 1. homography -> rectangle mapping ───────────────────────────────────────

def _trapezoid_canvas(quad, shape=(200, 200), bg=255, ink=0):
    import cv2
    c = np.full(shape, bg, np.uint8)
    # quad order is TL,TR,BL,BR; a polygon needs TL,TR,BR,BL or it self-intersects.
    b = quad[[0, 1, 3, 2]]
    cv2.fillPoly(c, [np.asarray(b, np.int32)], ink)
    return c


def test_rect_size_is_the_quads_own_extent():
    quad = np.array([[20, 20], [180, 20], [40, 180], [160, 180]], np.float32)
    # top edge 160, bottom edge 120 -> W=160; both vertical edges ~161.2 -> H=161
    w, h = RR.rect_size(quad)
    assert (w, h) == (160, 161), (w, h)


def test_known_trapezoid_warps_to_the_expected_rectangle():
    quad = np.array([[20, 20], [180, 20], [40, 180], [160, 180]], np.float32)
    src = _trapezoid_canvas(quad)                 # ink exactly filling the quad
    out = RR.rectify(src, quad)
    w, h = RR.rect_size(quad)
    assert out.shape == (h, w)
    assert out.dtype == np.uint8
    # The whole rectified frame is receipt: the warp cropped to the QUAD, not to
    # its bounding box, so it is ink everywhere and far darker than the source
    # (which is mostly white page around the trapezoid).
    assert out.mean() < 20.0, out.mean()
    assert out.mean() < src.mean() / 3.0
    # every corner of the destination rectangle is ink -- the quad's corners
    # were mapped onto the rectangle's corners in the order TL,TR,BL,BR.
    for y, x in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        assert out[y, x] == 0, (y, x, out[y, x])


def test_rectify_is_a_genuine_projective_map_not_a_crop():
    # A bright dot at the exact centroid of a strongly skewed quad must land at
    # (approximately) the centre of the rectangle. A bounding-box crop would
    # place it off-centre; a projective map puts it in the middle.
    import cv2
    quad = np.array([[30, 60], [170, 20], [60, 190], [190, 150]], np.float32)
    src = np.full((240, 240), 255, np.uint8)
    cx, cy = quad.mean(axis=0)
    cv2.circle(src, (int(round(cx)), int(round(cy))), 3, 0, -1)
    out = RR.rectify(src, quad)
    h, w = out.shape
    ys, xs = np.nonzero(out < 128)
    assert len(ys) > 0
    assert abs(ys.mean() - h / 2) < h * 0.08, (ys.mean(), h)
    assert abs(xs.mean() - w / 2) < w * 0.08, (xs.mean(), w)


def test_rectify_is_deterministic():
    quad = np.array([[20, 20], [180, 25], [40, 180], [165, 175]], np.float32)
    src = _trapezoid_canvas(quad)
    assert np.array_equal(RR.rectify(src, quad), RR.rectify(src, quad))


def test_key_of_maps_page_name_to_homography_key():
    assert RR.key_of("1000-receipt") == "1000"
    assert RR.key_of("1019") == "1019"


# ── 2. the exact transcript-count rule ───────────────────────────────────────

def test_align_labels_drops_on_any_mismatch_never_trims():
    boxes = [(0, 10, 0, 50), (12, 22, 0, 50)]
    lines = ["A", "B", "C"]                       # one more line than box
    pairs, reason = RR.align_labels(boxes, lines)
    assert pairs is None
    assert "count mismatch" in reason
    # and it is symmetric: too many boxes is also a drop, not a truncation
    pairs2, reason2 = RR.align_labels(boxes + [(24, 34, 0, 50)], ["A", "B"])
    assert pairs2 is None and "count mismatch" in reason2


def test_align_labels_pairs_in_reading_order_when_counts_match():
    boxes = [(0, 10, 0, 50), (12, 22, 0, 50)]
    pairs, reason = RR.align_labels(boxes, ["A", "B"])
    assert reason is None
    assert [p[1] for p in pairs] == ["A", "B"]


def test_page_crops_drops_a_mismatched_page_and_returns_no_crops(tmp_path):
    # A real image on disk, but a bander that deliberately reports the wrong
    # number of boxes: the page must be DROPPED, with a count-mismatch reason
    # and ZERO crops -- not silently trimmed to the transcript length.
    jp = tmp_path / "0001-receipt.jpg"
    Image.fromarray(np.full((60, 120), 255, np.uint8)).save(jp)
    page = ("0001-receipt", jp, ["LINE ONE", "LINE TWO"])

    def wrong_bander(gray, flags):
        return [(0, 10, 0, 100)] * 3              # 3 boxes vs 2 lines

    res, reason, n_boxes, was_rect = RR.page_crops(page, {}, bander=wrong_bander)
    assert res is None
    assert "count mismatch" in reason
    assert n_boxes == 3

    def right_bander(gray, flags):
        return [(0, 10, 0, 100), (12, 22, 0, 100)]

    res2, reason2, _, _ = RR.page_crops(page, {}, bander=right_bander)
    assert reason2 is None
    crops, labels = res2
    assert len(crops) == 2 and labels == ["LINE ONE", "LINE TWO"]


# ── 3. written cache schema / shape / width bounds ───────────────────────────

def _fake_pages(n=20):
    """n pages of small crops in the module's own per_page contract."""
    rng = np.random.RandomState(0)
    per_page = {}
    for i in range(n):
        cs, ls = [], []
        for j in range(3):
            a = rng.randint(0, 255, (14 + j, 40 + 5 * j)).astype(np.uint8)
            cs.append(B.crop_to_array(Image.fromarray(a)))
            ls.append("PAGE%d LINE%d" % (i, j))
        per_page["%04d-receipt" % i] = (cs, ls)
    return per_page


def test_written_cache_has_the_trainers_schema_and_bounds(tmp_path):
    out = tmp_path / "crops_rectified.npz"
    rep = RR.build_cache(_fake_pages(), out_path=out, verbose=False)
    assert out.exists()
    z = np.load(out, allow_pickle=True)
    for k in ("images", "widths", "labels", "variants", "splits", "page"):
        assert k in z, k
    imgs, widths = z["images"], z["widths"]
    # crop H is the trainer's CROP_H; every crop is right-padded within maxW.
    assert imgs.shape[1] == RR.CROP_H == B.CROP_H
    assert imgs.ndim == 3 and imgs.shape[0] == len(z["labels"]) == len(widths)
    assert widths.max() <= RR.MAX_W == B.MAX_W
    assert widths.min() >= B.MIN_W
    assert imgs.shape[2] == int(widths.max())
    assert set(np.unique(z["splits"])) <= {"train", "val", "test"}
    assert set(np.unique(z["variants"])) == {"real"}
    assert rep["crops"] == imgs.shape[0]


# ── 4. train / held-out document disjointness ────────────────────────────────

def test_split_is_document_disjoint(tmp_path):
    out = tmp_path / "crops_rectified.npz"
    RR.build_cache(_fake_pages(20), out_path=out, verbose=False)
    z = np.load(out, allow_pickle=True)
    pages, splits = list(z["page"]), list(z["splits"])
    by = {}
    for p, s in zip(pages, splits):
        by.setdefault(s, set()).add(p)
    assert by.get("train"), "no train pages -- split degenerate"
    assert not (by.get("train") & (by.get("val", set()) | by.get("test", set()))), (
        "a document leaked across the split")
    # every page carries exactly one split
    assert len(pages) == len(splits)


def test_split_is_a_pure_function_of_seed(tmp_path):
    a = RR.build_cache(_fake_pages(20), out_path=tmp_path / "r1.npz", verbose=False)
    b = RR.build_cache(_fake_pages(20), out_path=tmp_path / "r2.npz", verbose=False)
    assert a["split_counts"] == b["split_counts"]
    assert a["pages_by_split"] == b["pages_by_split"]


# ── the real cache, when present, is disjoint too ────────────────────────────

@pytest.mark.skipif(not RR.CACHE.exists(), reason="rectified cache not built")
def test_shipped_rectified_cache_is_disjoint_and_bounded():
    z = np.load(RR.CACHE, allow_pickle=True)
    pages, splits = list(z["page"]), list(z["splits"])
    tr = {p for p, s in zip(pages, splits) if s == "train"}
    ho = {p for p, s in zip(pages, splits) if s != "train"}
    assert tr and ho and not (tr & ho)
    assert z["images"].shape[1] == RR.CROP_H
    assert int(z["widths"].max()) <= RR.MAX_W
    assert z["images"].shape[2] == int(z["widths"].max())
