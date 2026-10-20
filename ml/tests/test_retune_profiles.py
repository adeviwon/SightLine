"""
Tests for retune_profiles — the statistic wrappers, the seed convention, and the
severity/bracketing predicates the re-tune is judged by.

Fast by design: the statistic tests run on synthetic arrays, the ordering and
bracketing tests run on the measured table embedded in the module, and only two
small live checks render anything (a 3-page sample, and a 1-page subprocess pair).
The full hand-photographed detector run lives in `--compare`, not here.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import real_degradation_cache as RDC   # noqa: E402  the SAME stats -- don't reimplement
import retune_profiles as RP           # noqa: E402


# ── the statistic functions, on arrays with a KNOWN answer ───────────────────

def test_blank_page_scores_zero_ink_and_zero_sharpness():
    """Uniform paper: no ink population, no edges. All three stats must be 0."""
    a = np.full((32, 120), 255, np.uint8)
    ink, contrast, sharp = RDC.crop_stats(a)
    assert ink == 0.0
    assert contrast == 0.0
    assert sharp == 0.0


def test_known_ink_fraction_is_recovered():
    """A black block covering exactly 25% of the crop must read ~0.25 ink."""
    for frac in (0.25, 0.5, 0.75):
        a = np.full((40, 100), 255, np.uint8)
        a[:, : int(round(100 * frac))] = 0
        ink, contrast, sharp = RDC.crop_stats(a)
        assert ink == pytest.approx(frac, abs=0.02), (frac, ink)
        # Ink and paper are maximally separated on a two-tone crop.
        assert contrast == pytest.approx(255.0, abs=2.0)


def test_contrast_is_the_ink_paper_population_gap():
    """Contrast = mean(paper) - mean(ink), so a known gap must come back."""
    a = np.full((40, 100), 200, np.uint8)
    a[:, :50] = 40
    _ink, contrast, _sharp = RDC.crop_stats(a)
    assert contrast == pytest.approx(160.0, abs=2.0)


def test_laplacian_variance_falls_with_blur():
    """A known blurred edge scores a predictably LOWER Laplacian variance.

    Monotone across sigma, and by sigma=3 the edge's energy is almost gone.
    """
    a = np.zeros((64, 64), np.uint8)
    a[:, 32:] = 255
    sharp = []
    for sigma in (0.0, 1.0, 2.0, 3.0):
        b = a if sigma == 0 else cv2.GaussianBlur(a, (0, 0), sigma)
        sharp.append(RDC.crop_stats(b)[2])
    assert all(sharp[i] > sharp[i + 1] for i in range(len(sharp) - 1)), sharp
    assert sharp[-1] < 0.15 * sharp[0], sharp


def test_a_fully_dark_crop_has_no_paper_population():
    """All-dark: the split is degenerate, so contrast must be 0, not garbage."""
    a = np.zeros((32, 64), np.uint8)
    ink, contrast, sharp = RDC.crop_stats(a)
    assert contrast == 0.0
    assert sharp == 0.0


# ── the profile table: determinism ───────────────────────────────────────────

def test_profile_sample_is_reproducible_in_process():
    """Same seed, same page set -> byte-identical crops, twice."""
    prof = {"handheld_light": RP.TUNED["handheld_light"]}
    a = RP.synthetic_native(prof, pages_per_doc=1)
    b = RP.synthetic_native(prof, pages_per_doc=1)
    assert len(a["handheld_light"]) == len(b["handheld_light"]) > 0
    for x, y in zip(a["handheld_light"], b["handheld_light"]):
        assert np.array_equal(x, y)


def test_profile_seed_is_process_stable():
    """THE regression guard: different PYTHONHASHSEED -> identical corpus.

    Seeding from `hash(name)` salted per process once produced three different
    corpora from one seed. This runs the fingerprint in two interpreters with
    different hash seeds and fails if the digests disagree.
    """
    out = []
    for h in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": h}
        env.pop("PYTHONPATH", None)
        env.pop("VIRTUAL_ENV", None)
        r = subprocess.run(
            [sys.executable, str(SRC / "retune_profiles.py"), "--fingerprint"],
            cwd=str(RP.ROOT), env=env, capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stderr[-2000:]
        out.append(r.stdout.strip())
    a = hashlib.sha256(out[0].encode()).hexdigest()
    b = hashlib.sha256(out[1].encode()).hexdigest()
    assert a == b, (out[0], out[1])


# ── severity ordering ────────────────────────────────────────────────────────

def test_severity_ordering_holds_after_the_retune():
    """light <= heavy <= low_light in ink; >= in contrast and sharpness."""
    t = RP.embedded_table()
    assert RP.severity_ok(t, "native", "after")
    assert RP.severity_ok(t, "trainer", "after")


def test_severity_ordering_was_BROKEN_before_the_retune():
    """The test has teeth: the shipped profiles were not ordered at all.

    (Before: low_light is the DARKEST profile yet reports the LOWEST ink and the
    HIGHEST sharpness of the three, because Otsu splits mid-grey on a blurred,
    shadowed crop. That is the simulator bug, not a ladder.)
    """
    t = RP.embedded_table()
    for geo in ("native", "trainer"):
        assert not RP.severity_ok(t, geo, "before"), geo


def test_severity_ordering_is_monotone_in_each_metric():
    for geo in ("native", "trainer"):
        s = [RP.AFTER_STATS[geo][k]
             for k in ("handheld_light", "handheld_heavy", "low_light")]
        assert s[0]["ink"] < s[1]["ink"] < s[2]["ink"]
        assert s[0]["contrast"] > s[1]["contrast"] > s[2]["contrast"]
        assert s[0]["sharp"] > s[1]["sharp"] > s[2]["sharp"]


# ── bracketing reality ───────────────────────────────────────────────────────

def test_every_camera_profile_brackets_real_after_the_retune():
    rep = RP.bracket_report(RP.embedded_table(), "trainer", "after")
    for k, v in rep.items():
        assert v["ok"], (k, v)
        assert v["worst_ratio"] <= RP.BRACKET_FACTOR


def test_the_old_profiles_were_far_outside_the_bracket():
    """The headline: 17-52x away before, inside the bracket after."""
    rep = RP.bracket_report(RP.embedded_table(), "trainer", "before")
    worst_before = min(v["worst_ratio"] for v in rep.values())
    assert worst_before > RP.BRACKET_FACTOR, rep
    rep_after = RP.bracket_report(RP.embedded_table(), "trainer", "after")
    for k in rep:
        assert rep_after[k]["worst_ratio"] < rep[k]["worst_ratio"] / 3.0, k


def test_real_row_lies_inside_the_synthetic_span():
    """Real is bracketed on contrast and sharpness by the re-tuned profiles.

    Ink is the exception and it is structural: the renderer's own tight crops
    run denser than a real photo even with NO degradation (studio_clean), so no
    parameter choice can put a synthetic profile BELOW the real 0.227.
    """
    t = RP.embedded_table()
    real = t["real"]["before"]["trainer"]
    inks = [t[k]["after"]["trainer"]["ink"] for k in RP.CAMERA]
    cons = [t[k]["after"]["trainer"]["contrast"] for k in RP.CAMERA]
    shps = [t[k]["after"]["trainer"]["sharp"] for k in RP.CAMERA]
    assert min(cons) < real["contrast"] < max(cons)
    assert min(shps) < real["sharp"] < max(shps)
    # Ink: real sits BELOW every synthetic profile, including the clean one.
    assert min(inks) > real["ink"]
    assert t["studio_clean"]["after"]["trainer"]["ink"] > real["ink"]


# ── the parameters themselves ────────────────────────────────────────────────

PHYSICAL_KNOBS = {"defocus", "motion", "motion_angle", "iso", "noise_iso",
                  "light", "jpeg", "hand_shadow", "perspective", "rot", "glare"}


def test_retune_only_sets_physical_capture_knobs():
    """No statistic rescaler: the re-tune may only set knobs apply_profile reads.

    Guarding the failure mode the task names explicitly -- matching a statistic
    by transforming the output would be fitting the metric, not the phenomenon.
    """
    for prof, params in RP.TUNED.items():
        assert set(params) <= PHYSICAL_KNOBS, (prof, set(params) - PHYSICAL_KNOBS)


def test_studio_clean_and_non_camera_profiles_are_untouched():
    for k, v in RP.BEFORE.items():
        if k in RP.CAMERA:
            continue
        assert RP.AFTER[k] == v, k
    assert RP.AFTER["studio_clean"] == {}


def test_camera_profiles_actually_changed():
    for k in RP.CAMERA:
        assert RP.AFTER[k] != RP.BEFORE[k], k
        assert RP.AFTER[k] == RP.TUNED[k], k


def test_tuned_profiles_are_less_blurred_than_the_shipped_ones():
    """The physical direction of the fix: every blur/carried-degradation knob
    the profile still has must be <= the shipped value."""
    for k in RP.CAMERA:
        b, a = RP.BEFORE[k], RP.TUNED[k]
        for knob in ("defocus", "motion", "light"):
            if knob in b and knob in a:
                assert a[knob] <= b[knob], (k, knob, b[knob], a[knob])
        if "motion" in a:
            assert a["motion"] <= b.get("motion", a["motion"]), k


# ── the live check: the PARAMS, not just the embedded numbers ────────────────

def test_live_sample_matches_the_embedded_after_table():
    """Re-render a small sample and confirm the tuned params reproduce the table.

    This is what stops the embedded constants from going stale: the numbers in
    AFTER_STATS must be what `capture.apply_profile(**TUNED)` actually produces.
    """
    prof = {k: RP.TUNED[k] for k in RP.CAMERA}
    crops = RP.synthetic_native(prof, pages_per_doc=2)
    assert all(len(crops[k]) > 20 for k in RP.CAMERA)
    for k in RP.CAMERA:
        nat, trn = RP.geometry_stats(crops[k])
        want = RP.AFTER_STATS["trainer"][k]
        assert trn["ink"] == pytest.approx(want["ink"], abs=0.05), (k, trn)
        assert trn["contrast"] == pytest.approx(want["contrast"], rel=0.20), (k, trn)
        assert trn["sharp"] == pytest.approx(want["sharp"], rel=0.40), (k, trn)
        # And the live sample must itself stay inside the bracket.
        rep = RP.bracket_report(
            {**RP.embedded_table(),
             k: {"before": RP.embedded_table()[k]["before"],
                 "after": {"native": nat, "trainer": trn}}},
            "trainer", "after")
        assert rep[k]["worst_ratio"] <= RP.BRACKET_FACTOR, (k, rep[k])
