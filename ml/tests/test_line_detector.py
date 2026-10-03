"""
Pin the line detector's actual failure mode, which is NOT the splitting rule.

Five of the twenty real hand-photographed pages collapse every way it is tried:
1002, 1005, 1006, 1009 and 1019. 1002-receipt has 26 transcript lines and
produces 2 bands; 1019-receipt produces 1. Measured cause: the row-ink profile
never approaches zero, so there is no gap for ANY rule to split at.

    1002-receipt   ink in 95.0% of rows
    1019-receipt   ink in 100.0% of rows, profile minimum of 3

Six fixes were built and measured against this, and every one failed or traded
away CER. They are recorded here so the next attempt starts from the
measurements instead of from intuition, and so the failure is visible in the
suite rather than only in a log.

Two of these tests use real pages and skip when the dataset is absent, because
the failure is specific to photographs and a synthetic fixture would invert the
property it is supposed to check.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "/home/ubuntu/Projects/sightline/ml/src")

import eval_ocr as E      # noqa: E402
import realdata as R      # noqa: E402

PAGES_PRESENT = bool(R.handheld_pages())


def _profile(gray):
    """Row-ink profile the way eval_handheld builds it."""
    ink = E.ink_mask(gray)
    binv = (ink.astype(np.uint8)) * 255
    H, W = gray.shape
    ys = np.where(binv.sum(axis=1) > W * 0.02)[0]
    if len(ys) > 20:
        binv = binv[max(0, ys[0] - 5):min(H, ys[-1] + 6), :]
    return (binv > 0).sum(axis=1)


# ── the saturation failure itself ────────────────────────────────────────────

@pytest.mark.skipif(not PAGES_PRESENT, reason="handheld photos not downloaded")
@pytest.mark.xfail(strict=True, reason=(
    "KNOWN DEFECT, not yet fixed. 1019-receipt's row-ink profile has a minimum "
    "of 3 against a maximum of 93, so the page is ink-saturated and 26 "
    "transcript lines collapse into 1 band. strict=True means that when someone "
    "fixes the paper localisation this turns into an XPASS FAILURE, which is "
    "the signal to delete this marker and re-measure detection recall."))
def test_the_profile_must_go_near_zero_between_lines():
    """
    THE core property: a row profile that never drops leaves no gap to split.

    This is the whole bug. Measured on the shipped detector:

        1002-receipt   ink in 95.0% of rows
        1019-receipt   ink in 100.0% of rows, profile minimum of 3 of 93

    so the "gaps" between lines are indistinguishable from text, every
    splitter collapses the page, and 1002/1005/1006/1009/1019 all fail the
    same way. Six fixes were built and measured against this; none paid off
    (see the rejected table in eval_ocr.py). The blocker is that the RECEIPT
    must be localised before line finding, and localisation only works once
    its mask is trustworthy rather than picking the largest bright blob.

    The threshold is 2 rather than 0 because a real inter-line gap should read
    as empty, and anything above a couple of pixels means the profile has no
    usable valley.
    """
    pages = {n: jp for n, jp, _ in R.handheld_pages()}
    if "1019-receipt" not in pages:
        pytest.skip("1019-receipt not in this dataset copy")
    from PIL import Image
    gray = np.asarray(Image.open(pages["1019-receipt"]).convert("L"), np.uint8)
    proj = _profile(gray)
    assert proj.min() <= 2, (
        f"1019-receipt row profile never approaches zero (min={proj.min()}, "
        f"max={proj.max()}). The page is ink-saturated, so no splitting rule "
        f"can find a gap."
    )


# ── the rejected alternatives, so they are not silently re-introduced ───────

def test_page_scaled_paper_window_did_not_pay_off():
    """
    The best candidate on the bands/line proxy, and it was still reverted.

    page-scaled window + margin 45 scored 0.817 bands/line against a 0.736
    baseline and raised REAL detection recall 67.9% -> 71.1% -- while pushing
    CER 56.5% -> 61.5%. The extra bands are real bands in the wrong place, so
    the ceiling rose and the text degraded. Quoting the recall alone would be
    cherry-picking, so the rejected values are kept for reproduction and the
    defaults are asserted here.
    """
    # Assert on the VALUE the detector resolves, not on source text. The
    # previous version of this test grepped the module for "PAPER_DELTA" and a
    # rename to E_PAPER passed it -- verified by mutation, which is the only
    # way to know a gate has teeth.
    assert E.SHIPPED_WINDOW_PX == 31, (
        f"the shipped paper window is {E.SHIPPED_WINDOW_PX}, not 31. The "
        f"page-scaled variant was measured and REVERTED: detection recall rose "
        f"to 71.1% but CER degraded from 56.5% to 61.5%."
    )
    assert E.SHIPPED_DELTA == 18, (
        f"the shipped ink margin is {E.SHIPPED_DELTA}, not 18. See above -- "
        f"margin 45 raised recall and regressed CER."
    )


def test_ink_mask_defaults_to_the_shipped_estimate():
    """
    ink_mask() must default to the shipped values.

    eval_handheld calls ink_mask(a) with no arguments, so if the default ever
    drifts, the detector silently changes behaviour and no other test notices.
    Checked by VALUE: the no-argument call must equal an explicit call at the
    shipped constants.
    """
    rng = np.random.default_rng(0)
    gray = rng.integers(60, 200, size=(120, 160), dtype=np.uint8)
    assert np.array_equal(E.ink_mask(gray),
                          E.ink_mask(gray, E.SHIPPED_WINDOW_PX,
                                     E.SHIPPED_DELTA)), (
        "ink_mask's defaults drifted from SHIPPED_WINDOW_PX/SHIPPED_DELTA"
    )


def test_line_bands_is_the_shared_implementation():
    """
    eval_handheld must call the shared helpers rather than carry private copies.

    A test that exercises a private copy of the algorithm it is testing passes
    while the shipped code diverges. This pins the call sites.
    """
    body = Path(E.__file__).read_text().split("def eval_handheld", 1)[-1]
    assert "line_bands(" in body, (
        "eval_handheld stopped calling the shared line_bands(); it must not "
        "carry its own copy of the splitting loop"
    )
    assert "ink_mask(" in body, (
        "eval_handheld stopped calling the shared ink_mask(); a private "
        "binarisation copy is exactly what let the paper-estimation bug hide"
    )
    assert "medianBlur" not in body, (
        "eval_handheld is calling cv2.medianBlur directly again instead of "
        "going through ink_mask()"
    )


def test_paper_window_is_odd_and_scales_with_height():
    """cv2.medianBlur requires an odd kernel, and these pages vary 258-1000px."""
    for h in (258, 500, 1000, 2000):
        k = E.paper_window(h)
        assert k % 2 == 1, f"medianBlur window must be odd, got {k} for {h}px"
        assert k >= 9, f"window collapsed to {k} at {h}px"
    assert E.paper_window(1000) > E.paper_window(258), (
        "the window must scale with page height: these pages range 258-1000px "
        "and a fixed window is a different fraction of a line gap on each"
    )


def test_line_bands_finds_a_known_layout():
    """A synthetic profile with known gaps must produce exactly those bands."""
    proj = np.zeros(200, np.int64)
    for start in (20, 70, 120):
        proj[start:start + 20] = 100
    bands = E.line_bands(proj)
    assert len(bands) == 3, f"expected 3 bands, got {bands}"
    for (s, e) in bands:
        assert e - s >= 8, "bands shorter than the minimum height leaked through"


def test_line_bands_merges_a_profile_with_no_gaps():
    """
    The saturation case, isolated. One solid run MUST produce one band.

    This is what happens to 1002 and 1019 in the field, and asserting the
    behaviour documents that a solid profile cannot be split rather than
    leaving it to be rediscovered.
    """
    proj = np.full(200, 100, np.int64)
    assert len(E.line_bands(proj)) == 1, (
        "a profile with no gaps should collapse to one band; if this returns "
        "several, line_bands is inventing splits and the saturated-page "
        "numbers need re-measuring"
    )