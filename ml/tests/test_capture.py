"""
test_capture.py — the degradation simulator must be physically plausible.

These are the invariants that catch the two failure modes a synthetic-data
pipeline actually suffers from:

  1. A SHAPE BUG. If a degradation changes H or W, the 64x256 tiling in
     evaluate.py either crashes or silently misaligns every patch. So every
     profile must return an image with EXACTLY the source's size.
  2. A NO-OP DEGRADATION. If a function is accidentally wired to skip (or its
     kernel is normalised away), training pairs become (clean, clean) and the
     restorer learns nothing while the loss curve still looks healthy. So each
     degradation must provably change the pixels.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

# Project modules live in ml/src as plain modules (not an installed package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import capture  # noqa: E402

PROFILES = list(capture.CAPTURE_PROFILES)
# `studio_clean` is the identity baseline by design — it is the control arm,
# so "does this profile change the image" does not apply to it.
DEGRADING = [p for p in PROFILES if p != "studio_clean"]


def arr(img):
    return np.asarray(img.convert("RGB"))


# ── 1. Shape preservation (catches tiling crashes) ───────────────────────

@pytest.mark.parametrize("profile", PROFILES)
def test_profile_preserves_exact_size(clean_img, profile):
    """A degradation that resizes the page breaks 64x256 tiling downstream."""
    out = capture.apply_profile(clean_img, seed=7, **capture.CAPTURE_PROFILES[profile])
    assert out.size == clean_img.size, (
        f"profile {profile!r} changed size {clean_img.size} -> {out.size}")


@pytest.mark.parametrize("profile", PROFILES)
def test_profile_preserves_mode_rgb(clean_img, profile):
    """evaluate.py's _np() calls .convert('L'); a mode change would slip past
    the size check but corrupt the pixel maths."""
    out = capture.apply_profile(clean_img, seed=7, **capture.CAPTURE_PROFILES[profile])
    assert out.mode == "RGB"


def test_perspective_preserves_size(clean_img):
    """Perspective is a homography, not a crop or a zoom: the canvas stays."""
    for strength in (0.01, 0.035, 0.05, 0.2):
        out = capture.perspective(clean_img, strength=strength, seed=3)
        assert out.size == clean_img.size


def test_all_degradations_preserve_size(clean_img):
    """Individual physical degradations, not just whole profiles."""
    cases = [
        lambda i: capture.motion_blur(i, 11, 30.0),
        lambda i: capture.defocus_blur(i, 3.0),
        lambda i: capture.sensor_noise(i, 3200, seed=1),
        lambda i: capture.hand_shadow(i, seed=1),
        lambda i: capture.perspective(i, 0.05, seed=1),
        lambda i: capture.lighting_gradient(i, 0.5, seed=1),
        lambda i: capture.glare(i, seed=1),
        lambda i: capture.jpeg(i, 40),
    ]
    for i, fn in enumerate(cases):
        assert fn(clean_img).size == clean_img.size, f"degradation #{i} resized"


# ── 2. No-op detection (catches a silently broken kernel) ───────────────

@pytest.mark.parametrize("profile", DEGRADING)
def test_profile_actually_degrades(clean_img, profile):
    """Every non-baseline profile must change the pixels."""
    out = capture.apply_profile(clean_img, seed=7, **capture.CAPTURE_PROFILES[profile])
    assert not np.array_equal(arr(out), arr(clean_img)), (
        f"profile {profile!r} is a NO-OP — it would produce (clean, clean) "
        f"training pairs and the restorer could not learn anything")


def test_studio_clean_is_the_identity_control(clean_img):
    """The baseline arm must be a true control: no degradation at all."""
    out = capture.apply_profile(clean_img, seed=7, **capture.CAPTURE_PROFILES["studio_clean"])
    assert np.array_equal(arr(out), arr(clean_img))


@pytest.mark.parametrize("name", ["motion_blur", "defocus_blur", "sensor_noise",
                                  "hand_shadow", "perspective",
                                  "lighting_gradient", "glare", "jpeg"])
def test_individual_degradation_changes_image(clean_img, name):
    """A single degradation in isolation must not be a no-op."""
    fn = getattr(capture, name)
    kwargs = {
        "motion_blur": dict(length=9, angle=25.0),
        "defocus_blur": dict(radius=2.2),
        "sensor_noise": dict(iso=3200, seed=1),
        "hand_shadow": dict(seed=1),
        "perspective": dict(strength=0.05, seed=1),
        "lighting_gradient": dict(strength=0.4, seed=1),
        "glare": dict(seed=1),
        "jpeg": dict(quality=50),
    }[name]
    assert not np.array_equal(arr(fn(clean_img, **kwargs)), arr(clean_img)), (
        f"{name}() returned the input unchanged")


# ── 3. Determinism (the "every number is reproducible" claim) ───────────

SEEDED = ["sensor_noise", "hand_shadow", "perspective",
          "lighting_gradient", "glare"]


@pytest.mark.parametrize("name", SEEDED)
def test_same_seed_gives_identical_bytes(clean_img, name):
    """evaluate.py states every number comes from a seeded, reproducible run."""
    fn = getattr(capture, name)
    kw = dict(seed=42) if name != "sensor_noise" else dict(iso=1600, seed=42)
    a = np.asarray(fn(clean_img, **kw).convert("RGB")).tobytes()
    b = np.asarray(fn(clean_img, **kw).convert("RGB")).tobytes()
    assert a == b, f"{name} is not deterministic for a fixed seed"


@pytest.mark.parametrize("name", SEEDED)
def test_different_seed_gives_different_bytes(clean_img, name):
    """A seed that does nothing is as broken as no seed at all."""
    fn = getattr(capture, name)
    kw = dict(seed=1) if name != "sensor_noise" else dict(iso=1600, seed=1)
    other = dict(seed=2) if name != "sensor_noise" else dict(iso=1600, seed=2)
    a = np.asarray(fn(clean_img, **kw).convert("RGB")).tobytes()
    b = np.asarray(fn(clean_img, **other).convert("RGB")).tobytes()
    assert a != b, f"{name} ignored its seed entirely"


@pytest.mark.parametrize("profile", PROFILES)
def test_apply_profile_is_reproducible(clean_img, profile):
    """The profile wrapper must be deterministic, not just its components."""
    p = capture.CAPTURE_PROFILES[profile]
    a = np.asarray(capture.apply_profile(clean_img, seed=99, **p)).tobytes()
    b = np.asarray(capture.apply_profile(clean_img, seed=99, **p)).tobytes()
    assert a == b


def test_motion_blur_direction_matters(clean_img):
    """
    Real motion blur is DIRECTIONAL — the kernel is the wrist's angle. An
    axis-aligned-only implementation (a separable box blur, or an angle that
    is ignored) would blur 0deg and 90deg identically and would make the
    'handheld' profiles far easier than the physical reality.
    """
    h = np.asarray(capture.motion_blur(clean_img, 9, 0.0)).astype(float)
    v = np.asarray(capture.motion_blur(clean_img, 9, 90.0)).astype(float)
    assert not np.array_equal(h, v), (
        "motion_blur is angle-independent — the kernel is not directional")
    assert np.abs(h - v).mean() > 1.0, "angle changes the result only trivially"


def test_motion_blur_longer_is_stronger(clean_img):
    """A longer exposure smear must destroy more detail, not less."""
    clean = np.asarray(clean_img.convert("L")).astype(float)
    errs = [np.abs(np.asarray(capture.motion_blur(clean_img, n, 25.0)
                              .convert("L")).astype(float) - clean).mean()
            for n in (3, 9, 15)]
    assert errs[0] < errs[1] < errs[2], f"error not monotone in kernel length: {errs}"


# ── 4. Physics: noise must scale with ISO ───────────────────────────────

def test_noise_increases_with_iso(flat_doc):
    """
    A higher ISO means a worse sensor. If the sigma did not grow with ISO the
    'low_light' and 'worst_case' profiles would not be the hard cases the
    headline metrics claim they are.
    """
    base = np.asarray(flat_doc.convert("L")).astype(float)
    stds = []
    for iso in (100, 400, 1600, 6400, 25600):
        noisy = np.asarray(capture.sensor_noise(flat_doc, iso, seed=5)
                           .convert("L")).astype(float)
        stds.append(float((noisy - base).std()))
    assert all(b >= a for a, b in zip(stds, stds[1:])), (
        f"noise std is not monotone in ISO: {stds}")
    assert stds[-1] > 1.5 * stds[0], f"ISO has almost no effect: {stds}"


def test_noise_is_zero_mean(flat_doc):
    """Sensor noise must not bias the brightness, or the restorer learns a
    brightness shift instead of an inverse."""
    base = np.asarray(flat_doc.convert("L")).astype(float)
    for iso in (800, 6400):
        d = (np.asarray(capture.sensor_noise(flat_doc, iso, seed=3)
                        .convert("L")).astype(float) - base)
        assert abs(float(d.mean())) < 0.15 * d.std(), (
            f"noise at ISO {iso} is biased by {d.mean():.3f} grey levels "
            f"(std {d.std():.3f}); sensor noise must be zero-mean")


# ── 5. Patch extraction contract ────────────────────────────────────────

def test_to_patch_contract(clean_img):
    """model.py is hard-wired to 64x256 float32 in [0,1]."""
    p = capture.to_patch(clean_img, y0=100)
    assert p.dtype == np.float32
    assert p.shape == (64, 256)
    assert 0.0 <= float(p.min()) and float(p.max()) <= 1.0


def test_to_patch_y0_alias(clean_img):
    """to_patch(img, y0=..) must mean the same band as to_patch(img, y=..)."""
    assert np.array_equal(capture.to_patch(clean_img, y0=164),
                          capture.to_patch(clean_img, y=164))


def test_to_patch_content_matches_crop(clean_img):
    """to_patch must be a faithful crop, not a resize or a blur."""
    p = capture.to_patch(clean_img, x=48, y=100, w=256, h=64)
    ref = np.asarray(clean_img.convert("L").crop((48, 100, 48 + 256, 100 + 64)),
                     dtype=np.float32) / 255.0
    assert np.array_equal(p, ref)


def test_to_patch_survives_a_degraded_profile(clean_img):
    """The real training path: patch a fully degraded page."""
    deg = capture.apply_profile(clean_img, seed=4, **capture.CAPTURE_PROFILES["worst_case"])
    p = capture.to_patch(deg, y0=100)
    assert p.dtype == np.float32 and p.shape == (64, 256)
    assert 0.0 <= float(p.min()) and float(p.max()) <= 1.0


# ── 6. Document corpus used as ground truth ─────────────────────────────

def test_every_doc_type_renders_and_degrades(clean_docs):
    """All three ground-truth document types must survive a full profile."""
    for lines, dtype in clean_docs:
        assert lines, f"{dtype} has no content"
        img = capture.render_document(lines, seed=3)
        assert img.size[0] >= 256 and img.size[1] >= 100 + 3 * 64, (
            f"{dtype} render is too small for the 3-band training crop")
        for prof in ("handheld_heavy", "worst_case"):
            out = capture.apply_profile(img, seed=3, **capture.CAPTURE_PROFILES[prof])
            assert out.size == img.size