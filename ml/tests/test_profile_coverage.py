"""
test_profile_coverage.py — the coverage check must not lie about coverage.

The failure this file exists to prevent is specific and has already cost this
project four train/serve mismatches: a profile list that SILENTLY DROPS a name the
caller asked for, a corpus that claims a profile it does not contain, or a render
that is not reproducible — each of which makes a build look like it closed a gap it
left open. So the pinned properties are:

  1. a requested profile that does not exist is NAMED and fatal, never dropped;
  2. the default list is exactly the profiles the app can be served (everything in
     `capture.CAPTURE_PROFILES` except the one marked unreadable to a human) and it
     is the same list `render_corpus.PROFILES` already declares;
  3. a page rendered under the existing seed convention (zlib.crc32 via
     `render_corpus.page_seed`, never the per-process `hash()`) is byte-identical
     on a second render;
  4. every crop that goes into a corpus survives the trainer's own transform
     (`build_dataset.crop_to_array`) inside [CROP_H, MIN_W..MAX_W].
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_dataset as B            # noqa: E402
import capture as C                  # noqa: E402
import profile_coverage as PC        # noqa: E402
import render_corpus as RC           # noqa: E402

NEW = ["off_axis", "jpeg_social"]


# ── 1. the profile list ─────────────────────────────────────────────────────

def test_unknown_profile_is_named_and_fails():
    with pytest.raises(PC.UnknownProfile) as e:
        PC.resolve_profiles("studio_clean,off_axix")
    msg = str(e.value)
    assert "off_axix" in msg, "the bad name must be in the message, not a summary"
    # and a typo'd list must not reach the renderer at all
    with pytest.raises(PC.UnknownProfile):
        PC.resolve_profiles("studio_clean,Off_Axis")     # case is not aliased
    with pytest.raises(PC.UnknownProfile):
        PC.resolve_profiles("worst_cas")


def test_resolve_keeps_capture_profile_order_and_dedupes():
    assert PC.resolve_profiles("jpeg_social,studio_clean,jpeg_social") == \
        ["studio_clean", "jpeg_social"]


def test_default_list_is_every_profile_the_app_can_be_served():
    got = PC.resolve_profiles("")
    assert got == PC.app_servable_profiles()
    assert set(got) == set(C.CAPTURE_PROFILES) - PC.SUB_HUMAN
    # the corpus's own list must be that set, or the delivered --profiles list is
    # incomplete and the coverage gap survives the next build
    assert set(got) == set(RC.PROFILES), "render_corpus.PROFILES drifted from the app's world"
    assert "worst_case" not in got, "SUB_HUMAN must never be a training target"
    # the two the app is measurably served and the corpus does not contain
    assert {"off_axis", "jpeg_social"} <= set(got)


def test_bench_profiles_are_read_from_the_app_and_are_real_profiles():
    got = PC.app_bench_profiles()
    assert len(got) >= 2 and set(got) <= set(C.CAPTURE_PROFILES), got
    assert {"studio_clean", "off_axis", "jpeg_social"} <= set(got)


# ── 2. the sampled render: determinism and the trainer's schema ─────────────

def test_sampled_render_is_deterministic_and_lands_in_the_trainer_schema():
    rows, bad = PC.validate_sample(NEW, pages_per_profile=1, seed=0)
    assert not bad, "sampled render failed: " + "; ".join(bad)
    assert len(rows) == len(NEW)
    for r in rows:
        assert r["crops"] == r["lines"], "a rendered line produced no crop"
        assert r["deterministic"], "second render differs: %s" % r["page"]
        assert r["ok"]
        for h, w in r["packed"]:
            assert h == B.CROP_H, "crops must be CROP_H tall"
            assert B.MIN_W <= w <= B.MAX_W, "width outside the trainer's bounds"
        assert r["stats"]["sharp"] > 0


def test_render_is_seed_stable_across_processes():
    """The seed convention is zlib.crc32, not hash(): pinned as an exact value."""
    assert RC.page_seed("prescription-9000") == RC.page_seed("prescription-9000")
    a, _ = PC.render_pages(["jpeg_social"], 1, seed=0), None
    b, _ = PC.render_pages(["jpeg_social"], 1, seed=0), None
    assert a[0][2][0].tobytes() == b[0][2][0].tobytes()


def test_check_schema_clamps_and_reports_bounds():
    import numpy as np
    wide = (np.random.RandomState(0).rand(B.CROP_H * 3, 4000) * 255).astype("uint8")
    h, w, ok = PC.check_schema(wide)
    assert (h, w) == (B.CROP_H, B.MAX_W) and ok, "a wide crop must be clamped, not dropped"
    # a tall thin crop would scale to under MIN_W; it must be clamped up, not dropped
    h, w, ok = PC.check_schema(np.zeros((100, 1), np.uint8))
    assert h == B.CROP_H and w == B.MIN_W and ok
    h, w, ok = PC.check_schema(np.zeros((B.CROP_H, 200), np.uint8))
    assert (h, w, ok) == (B.CROP_H, 200, True)


# ── 3. the coverage table itself ────────────────────────────────────────────

def _corpus(profile, n=40, ink=0.1, contrast=100.0, sharp=500.0):
    return {profile: {"n": n, "ink_mean": ink, "ink_median": ink, "ink_p25": ink * .8,
                      "ink_p75": ink * 1.2, "contrast_mean": contrast,
                      "contrast_median": contrast, "contrast_p25": contrast * .9,
                      "contrast_p75": contrast * 1.1, "sharp_mean": sharp,
                      "sharp_median": sharp, "sharp_p25": sharp * .8,
                      "sharp_p75": sharp * 1.2, "width_median": 300.0}}


def test_a_profile_absent_from_the_corpus_is_reported_absent():
    cs = _corpus("studio_clean")
    app = {"off_axis": {"n": 11, "ink_mean": .12, "ink_median": .12,
                        "contrast_mean": 90.0, "contrast_median": 90.0,
                        "sharp_mean": 900.0, "sharp_median": 900.0}}
    rows = PC.coverage_table(cs, app, full=["studio_clean", "off_axis"])
    assert rows["off_axis"]["in_corpus"] is False
    assert rows["off_axis"]["app_crops"] == 11
    assert rows["off_axis"]["nearest"] == "studio_clean"
    assert rows["off_axis"]["gaps"]["sharp"]["xIQR"] > 0
    assert rows["studio_clean"]["in_corpus"] is True


def test_no_app_dump_still_produces_a_table():
    rows = PC.coverage_table(_corpus("off_axis"), None)
    assert set(rows) == set(PC.app_servable_profiles())
    assert rows["off_axis"]["in_corpus"] is True
    assert rows["jpeg_social"]["in_corpus"] is False
    assert rows["jpeg_social"]["nearest"] is None       # nothing to place it against


# ── 4. the plan and the delivered commands ──────────────────────────────────

def test_plan_counts_and_commands_cover_the_proposed_list():
    props = PC.resolve_profiles("")
    plan = PC.build_plan(props, 500, 15.67, 0.5)
    assert plan["pages"] == 500 * len(C.DOCS)
    assert plan["pages"] == 1500
    assert plan["crops"] > plan["pages"], "crops are cut per line, many per page"
    assert plan["minutes"] > 0
    cmds = "\n".join(PC.merge_commands(props, 500))
    assert "--profiles " + ",".join(props) in cmds
    for p in props:
        assert p in cmds
    assert "data/render_corpus/crops_train.npz" in cmds
    assert "eval-render --split test" in cmds and "eval-ocr" in cmds
    assert "eval_field_truth.py" in cmds
    assert "retrain3" in cmds, "the list must say to wait for the running training job"
