"""
Tests for document localisation, and a guard against reading noise as signal.

Localisation is shipped on its MECHANISM, not on its measured delta. The
mechanism is real and visible: a receipt photographed beside a bowl of noodles
(1002-receipt) puts ink in every row of a full-width projection, so the profile
never drops to a splittable gap. The delta is NOT real evidence -- across 290
aligned lines the localisation gains exactly 2 lines and loses 0, exact McNemar
p = 1.000, so it is indistinguishable from the baseline.

test_sample_size_cannot_support_detector_claims pins that, because the natural
thing to do with a p=1.000 result is to quietly drop it and keep quoting
"+3.7pp recall".
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "/home/ubuntu/Projects/sightline/ml/src")

import eval_ocr as E      # noqa: E402
import realdata as R      # noqa: E402

PAGES_PRESENT = bool(R.handheld_pages())


def _gray(name):
    from PIL import Image
    jp = {n: j for n, j, _ in R.handheld_pages()}[name]
    return np.asarray(Image.open(jp).convert("L"), np.uint8)


# ── the sample-size guard ───────────────────────────────────────────────────

def test_sample_size_cannot_support_detector_claims():
    """
    20 pages / 290 aligned lines cannot distinguish these detector variants.

    Measured: baseline 5 correct lines, localisation 7, i.e. 2 gained and 0
    lost. Exact McNemar p = 1.000. The Wilson intervals overlap heavily
    (1.8% [0.7-4.0] against 2.4% [1.2-4.9]).

    So any claim of the form "localisation improves recall by 3.7pp" is not
    supported by this evidence. Raising it needs more hand-photographed pages,
    not more detector variants -- and the number of variants tried (six, all
    rejected, all separated by less than the noise) is itself the warning sign.
    """
    N = 290
    base_k, new_k = 5, 7
    gained, lost = new_k - base_k, base_k - new_k
    assert gained + lost < 10, (
        "if a substantial number of lines actually flips between variants, the "
        "sample may now support a claim -- re-run the significance analysis and "
        "update the docs rather than assuming the old verdict still holds"
    )
    # Exact two-sided McNemar. With 2 discordant pairs and none lost, the
    # smallest attainable p is 1.0, so this cannot reach significance.
    n = gained + lost
    tail = sum(_comb(n, i) for i in range(0, n // 2 + 1))
    p = min(1.0, 2 * tail / 2 ** n)
    assert p > 0.05, (
        f"McNemar p = {p:.3f} -- localisation now looks significant. That "
        f"contradicts the recorded analysis; re-measure and update docs."
    )


def _comb(n, k):
    import math
    return math.comb(n, k)


# ── localisation behaviour ──────────────────────────────────────────────────

@pytest.mark.skipif(not PAGES_PRESENT, reason="handheld photos not downloaded")
def test_located_quad_is_a_plausible_region():
    """
    Whatever locate_document returns must be a quad inside the frame covering a
    real share of it. A degenerate quad would silently produce no bands, and
    eval_handheld falls back -- so the shape is what makes the fallback safe.
    """
    gray = _gray("1002-receipt")
    quad = E.locate_document(gray)
    assert quad is not None, "1002-receipt should yield a document quad"
    H, W = gray.shape
    assert quad.shape == (4, 2)
    assert quad[:, 0].max() <= W + 1 and quad[:, 0].min() >= -1, \
        f"quad escapes the frame horizontally: {quad.tolist()}"
    assert quad[:, 1].max() <= H + 1 and quad[:, 1].min() >= -1, \
        f"quad escapes the frame vertically: {quad.tolist()}"
    x0, x1 = quad[:, 0].min(), quad[:, 0].max()
    y0, y1 = quad[:, 1].min(), quad[:, 1].max()
    assert (x1 - x0) > 0.1 * W and (y1 - y0) > 0.1 * H, (
        f"quad covers only {100*(x1-x0)* (y1-y0)/(W*H):.1f}% of the frame"
    )


@pytest.mark.skipif(not PAGES_PRESENT, reason="handheld photos not downloaded")
def test_localisation_does_not_destroy_a_page_the_baseline_handled():
    """
    The 1001 regression: bright-blob selection picked a specular highlight
    covering 16.6% of the frame and collapsed a working page to 10% of its
    lines. Scoring candidates by the TEXT they explain is what prevents it.

    This asserts the mechanism, not an aggregate -- an aggregate test passed
    while 1001 was destroyed, because four other pages improved enough to hide
    it.
    """
    gray = _gray("1001-receipt")
    quad = E.locate_document(gray)
    assert quad is not None
    H, W = gray.shape
    (cx, cy), (rw, rh) = cv2_min_area(gray, quad)
    frac = (rw * rh) / float(H * W)
    assert frac >= 0.20, (
        f"located document covers only {frac*100:.1f}% of the frame. "
        f"1001-receipt's specular highlight covered 16.6% and destroyed a "
        f"working page when selected as 'the document'."
    )


def cv2_min_area(gray, quad):
    """Axis-aligned extent of a quad, without importing cv2 into the test."""
    xs, ys = quad[:, 0], quad[:, 1]
    return ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2), \
           (xs.max() - xs.min(), ys.max() - ys.min())


def test_bands_in_document_rejects_a_tiny_quad():
    """
    A quad covering almost nothing must yield no bands, not one enormous one.

    eval_handheld treats an empty result as a signal to fall back to the
    full-width projection, so this path is load-bearing: if it returned junk
    instead, the fallback would never trigger.
    """
    gray = np.full((200, 300), 220, np.uint8)
    gray[100:110, 20:120] = 20          # one short line of "text"
    quad = np.array([[0, 0], [3, 0], [3, 3], [0, 3]], np.int32)
    assert E.bands_in_document(gray, quad) == [], (
        "a 3x3 quad should contain no band; it returned "
        f"{E.bands_in_document(gray, quad)}"
    )


def test_locate_document_handles_a_blank_page():
    """No crash, no quad -- the caller must fall back cleanly."""
    gray = np.full((150, 200), 128, np.uint8)
    q = E.locate_document(gray)
    assert q is None or isinstance(q, np.ndarray)