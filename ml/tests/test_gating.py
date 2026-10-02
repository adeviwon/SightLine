"""
Tests for the degradation gate (ml/src/gating.py).

THE BUG THIS SUITE EXISTS FOR
-----------------------------
`weightFromImageData()` in app/js/gating.js passed a WHOLE IMAGE to a metric
whose anchors (SHARP_CLEAN 0.080 / SHARP_BAD 0.008) had been calibrated on
64x256 patches. Variance-of-Laplacian is a per-pixel statistic, so a 900x580
render measures ~3.5x lower than a patch of the same content, and every clean
profile came out at weight 0.43-0.66 — i.e. full restoration applied to a
perfect scan. That is precisely the failure the gate exists to prevent, and it
shipped.

js/gating.js documented the hazard in a SCALE WARNING header and then did it
anyway. A comment is not a control, so these tests are the control:

  test_scale_matters          — the anchors genuinely are scale-dependent
  test_image_weight_matches    — the tiled path reproduces per-tile behaviour
  test_clean_profiles_not_     — the regression itself, as a named test, so it
  restored                       cannot come back silently
  test_clean_tile_not_         — a single clean patch must not be restored
  restored
  test_median_not_mean         — blank margins must not dilute a blurred page

These are deliberately anchored to the real capture corpus rather than
synthetic noise: a synthetic blob can satisfy a formula check while telling you
nothing about whether the gate separates the distributions it must separate.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import capture  # noqa: E402
import gating  # noqa: E402
import seedutil  # noqa: E402


def _render(profile_name, doc_index=0):
    """One real degraded render, deterministically seeded."""
    params = capture.CAPTURE_PROFILES[profile_name]
    ds = 123 * 7919 + seedutil.name_hash(profile_name) % 100003
    lines, _ = capture.DOCS[doc_index % len(capture.DOCS)]
    clean = capture.render_document(lines, seed=ds)
    deg = capture.apply_profile(clean, seed=ds, **params)
    return np.asarray(deg.convert("L"), np.float32) / 255.0


# ── the scale dependency itself ────────────────────────────────────────────────

def test_scale_matters():
    """
    Full-image and patch sharpness MUST differ, or the whole tiling fix is
    pointless. If this ever fails, the anchors are no longer scale-dependent
    and the tiled path can be simplified away.
    """
    gray = _render("studio_clean")
    full = gating.sharpness(gray)
    patch = gating.sharpness(gray[100:164, 48:304])
    assert full < patch, (
        f"expected full-image sharpness ({full:.5f}) to read lower than "
        f"patch-scale ({patch:.5f}); if not, the anchors may be "
        f"scale-independent and this test's premise is void")


def test_scale_misuse_would_restores_clean_image():
    """
    The regression, stated as an executable claim.

    Feeding a whole image straight to degradation_weight() — what the app did
    before the fix — must produce a NON-ZERO weight on a clean scan. If this
    ever passes as zero, the bug cannot recur through this path.
    """
    gray = _render("studio_clean")
    naive = gating.degradation_weight(gray)
    tiled = gating.image_weight(gray)
    assert naive > 0.3, (
        f"naive full-image weight is {naive:.3f}, expected >0.3 — the "
        f"historical bug. If this is now ~0 the premise changed; re-check "
        f"whether the tiled path is still needed")
    assert tiled < 0.05, (
        f"tiled weight is {tiled:.3f}, expected ~0.0 for a clean scan")


# ── the tiled path ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("profile", [
    "studio_clean", "off_axis", "glossy_glare", "paper_texture"])
def test_clean_profiles_not_restored(profile):
    """A clean capture must score ~0 — no inference, no artefacts added."""
    w = gating.image_weight(_render(profile))
    assert w < 0.05, f"{profile} is undamaged but scored weight {w:.3f}"


@pytest.mark.parametrize("profile", [
    "handheld_light", "handheld_heavy", "low_light", "worst_case"])
def test_blurred_profiles_fully_restored(profile):
    """A blurred capture must score ~1 — this is where the model earns its keep."""
    w = gating.image_weight(_render(profile))
    assert w > 0.95, f"{profile} is badly degraded but scored weight {w:.3f}"


def test_image_weight_matches_median_of_tiles():
    """image_weight() is exactly the median of per-tile weights, by definition."""
    gray = _render("jpeg_social")
    h, w = gray.shape[:2]
    manual = [
        gating.degradation_weight(gray[y:y + gating.PATCH_H, x:x + gating.PATCH_W])
        for y in range(0, h - gating.PATCH_H + 1, gating.PATCH_H)
        for x in range(0, w - gating.PATCH_W + 1, gating.PATCH_W)
    ]
    assert manual, "no full tiles fit; test premise broken"
    assert gating.image_weight(gray) == pytest.approx(float(np.median(manual)))


def test_single_tile_path():
    """An image already at tile size takes the single-patch path unchanged."""
    gray = _render("studio_clean")[100:164, 48:304]
    assert gray.shape == (gating.PATCH_H, gating.PATCH_W)
    assert gating.image_weight(gray) == pytest.approx(
        gating.degradation_weight(gray))


def test_image_smaller_than_tile():
    """Below tile size it must still return a sane weight, not crash."""
    tiny = np.linspace(0, 1, 30 * 40, dtype=np.float32).reshape(30, 40)
    w = gating.image_weight(tiny)
    assert 0.0 <= w <= 1.0


# ── median, specifically ───────────────────────────────────────────────────────

def test_median_not_mean():
    """
    Blank margins must not dilute a genuinely blurred page.

    The real handheld_light render splits into 20 damaged tiles (weight 1.0) and
    7 blank ones (weight 0.0), so:

        median 1.000   <- the page's real state
        mean   0.593   <- dragged down by the blank tiles

    Mean would under-restore a page that needs full restoration; that is the
    failure this choice prevents. Pinned against the real tile distribution
    rather than synthetic padding, because an over-padded fixture (more blanks
    than content tiles) inverts the relationship and tests nothing.
    """
    gray = _render("handheld_light")
    h, w = gray.shape[:2]
    tiles = [gating.degradation_weight(gray[y:y + gating.PATCH_H,
                                           x:x + gating.PATCH_W])
             for y in range(0, h - gating.PATCH_H + 1, gating.PATCH_H)
             for x in range(0, w - gating.PATCH_W + 1, gating.PATCH_W)]

    damaged = [t for t in tiles if t > 0.5]
    blank = [t for t in tiles if t == 0.0]
    assert damaged and blank, (
        f"fixture needs both damaged ({len(damaged)}) and blank "
        f"({len(blank)}) tiles; got {len(tiles)} total — the render changed "
        f"and this test no longer proves anything")

    assert float(np.median(tiles)) > float(np.mean(tiles)), (
        f"median ({np.median(tiles):.3f}) should exceed mean "
        f"({np.mean(tiles):.3f}) when blank tiles are present")
    assert gating.image_weight(gray) == pytest.approx(float(np.median(tiles)))
    assert gating.image_weight(gray) > 0.95, "dilution suppressed restoration"


# ── blank / edge cases ─────────────────────────────────────────────────────────

def test_blank_image_scores_zero():
    """Blank paper has no structure to recover; restoring it can only add noise."""
    blank = np.full((580, 900), 0.5, np.float32)
    assert gating.image_weight(blank) == 0.0


def test_content_min_guard():
    """A near-blank tile is treated as blank regardless of its variance."""
    flat = np.full((gating.PATCH_H, gating.PATCH_W), 0.5, np.float32)
    flat[10:14, 10:14] = 0.501          # tiny structure, contrast ~0.001
    assert gating.contrast(flat) < gating.CONTENT_MIN
    assert gating.degradation_weight(flat) == 0.0


def test_weight_bounded():
    """Never outside [0,1] for any real capture, whatever the profile."""
    for name in capture.CAPTURE_PROFILES:
        w = gating.image_weight(_render(name))
        assert 0.0 <= w <= 1.0, f"{name} produced out-of-range weight {w}"


# ── anchor integrity ───────────────────────────────────────────────────────────

def test_anchors_unchanged():
    """
    The calibrated constants. js/gating.js must match exactly or
    `node tools/gating_parity.js` fails — this is the Python side of that
    contract, so a drift is caught in pytest too.
    """
    assert gating.SHARP_CLEAN == 0.080
    assert gating.SHARP_BAD == 0.008
    assert gating.CONTENT_MIN == 0.12


def test_tile_geometry_matches_restorer():
    """
    The anchors are calibrated at PATCH_W x PATCH_H; if the model's input
    geometry changes, they are wrong and must be re-measured.
    """
    assert (gating.PATCH_W, gating.PATCH_H) == (256, 64)


def test_noise_not_used_in_weight():
    """
    Immerkaer's estimator reads CLEAN text as noisier than blurred text
    (glyph edges dominate the sum-of-absolute-Laplacian), so it is reported
    but must not influence the weight. Guard against a future edit wiring it in.
    """
    gray_clean = _render("studio_clean")
    gray_bad = _render("worst_case")
    assert gating.noise_sigma(gray_clean) > gating.noise_sigma(gray_bad), (
        "premise changed: clean should read noisier than blurred")
    # Same sharpness -> same weight, regardless of the noise reading.
    same = np.linspace(0, 1, gating.PATCH_H * gating.PATCH_W,
                       dtype=np.float32).reshape(gating.PATCH_H, gating.PATCH_W)
    assert (gating.degradation_weight(same, noise=0.0)
            == gating.degradation_weight(same, noise=99.0))
