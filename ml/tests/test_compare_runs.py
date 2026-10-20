"""
test_compare_runs.py — the run-vs-run comparison tool.

The repo has been burned twice by quoting a few-point gain over a dozen
documents as a result. So the tool that every later accuracy experiment is
judged by is itself under test, on the things that make a delta trustworthy:

  * the Wilson 95% interval;
  * an exact McNemar test when the two runs scored the SAME items;
  * refusal to crown a winner the sample cannot support;
  * a paired test REFUSED and SAID SO when the runs scored different items;
  * a missing artifact NAMED, never crashed on.

These assert the tool's METHOD and its refusals, not the artifact values (those
change when the models are retrained).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compare_runs as C  # noqa: E402


# ── statistics ───────────────────────────────────────────────────────────────

def test_wilson_matches_hand_computed_50_of_100():
    # Standard Wilson 95% for 50/100, computed by hand: [0.404, 0.596].
    lo, hi = C.wilson(50, 100)
    assert lo == pytest.approx(0.404, abs=0.002)
    assert hi == pytest.approx(0.596, abs=0.002)


def test_wilson_edge_counts():
    assert C.wilson(0, 10)[0] == 0.0
    assert C.wilson(10, 10)[1] == pytest.approx(1.0)
    assert C.wilson(0, 0) == (0.0, 0.0)


def test_mcnemar_significant_and_inconclusive():
    # 10 discordant one way vs 2 the other (n=12) is significant;
    # 3 vs 2 (n=5) is not.
    assert C.mcnemar_exact(10, 2) < 0.05
    assert C.mcnemar_exact(3, 2) > 0.05
    assert C.mcnemar_exact(0, 0) == 1.0


# ── synthetic artifacts ──────────────────────────────────────────────────────

def _rec(i, correct):
    """One shipped-pipeline record; `correct` decides whether the classification
    agrees with the expected category."""
    return {
        "profile": "studio_clean",
        "doc": "d%d" % i,
        "expected_category": "medical",
        "category": "medical" if correct else "other",
        "expected_fields": [],
        "fields": [],
    }


def _hackart(records):
    n = len(records)
    ok = sum(1 for r in records
             if r.get("category") and r["category"] == r["expected_category"])
    return {
        "ok": True,
        "missing": [],
        "notes": [],
        "classifier": {"accuracy_mean": 0.9, "macro_f1_mean": 0.9, "n_test": 10},
        "ocr": {"test_word_acc": 0.3, "test_crops": 100},
        "pipeline": {"n": n, "classification": ok / n if n else 0.0,
                     "records": records},
    }


def _row(rows, metric):
    return [r for r in rows if r["metric"] == metric][0]


# ── shape detection ──────────────────────────────────────────────────────────

def test_detect_shape_by_keys_not_filename():
    assert C.detect_shape({"headline": {}, "per_profile": {}, "arms": []}) \
        == "eval_results"
    assert C.detect_shape({"ok": True, "classifier": {}, "pipeline": {},
                           "missing": []}) == "hackathon"
    assert C.detect_shape({"something": 1}) is None


# ── the verdicts ─────────────────────────────────────────────────────────────

def test_identical_runs_are_delta_zero_and_inconclusive_everywhere():
    a = _hackart([_rec(i, i % 2 == 0) for i in range(12)])
    so, sn, rows = C.compare(a, a)
    assert so == sn == "hackathon"
    assert rows
    for r in rows:
        assert r["delta"] == 0
        assert r["verdict"].startswith("INCONCLUSIVE")


def test_rows_carry_the_sample_size_the_verdict_rests_on():
    a = _hackart([_rec(i, True) for i in range(7)])
    _, _, rows = C.compare(a, a)
    r = _row(rows, "classification@pipeline")
    assert r["n"] == 7          # 7 items scored
    assert r["k_old"] == 7      # 7 of them successful


def test_paired_mcnemar_flags_a_real_change():
    # old correct on docs 0..9, new correct on docs 10,11 -> b=10, c=2.
    old = _hackart([_rec(i, i < 10) for i in range(12)])
    new = _hackart([_rec(i, i >= 10) for i in range(12)])
    _, _, rows = C.compare(old, new)
    r = _row(rows, "classification@pipeline")
    assert r["method"] == "mcnemar"
    assert r["discordant"] == (10, 2)
    assert r["p"] < 0.05
    assert r["verdict"].startswith("REGRESSED")   # new scored worse


def test_paired_mcnemar_is_inconclusive_when_the_change_is_noise():
    # old correct on 0..2, new correct on 3,4 -> b=3, c=2, n=5 -> p=1.0.
    old = _hackart([_rec(i, i < 3) for i in range(5)])
    new = _hackart([_rec(i, i >= 3) for i in range(5)])
    _, _, rows = C.compare(old, new)
    r = _row(rows, "classification@pipeline")
    assert r["method"] == "mcnemar"
    assert r["discordant"] == (3, 2)
    assert r["p"] > 0.05
    assert r["verdict"].startswith("INCONCLUSIVE")


def test_paired_test_is_refused_and_says_so_when_item_counts_differ():
    old = _hackart([_rec(i, i < 6) for i in range(12)])
    new = _hackart([_rec(i, i < 6) for i in range(11)])
    _, _, rows = C.compare(old, new)
    r = _row(rows, "classification@pipeline")
    assert r["method"] != "mcnemar"          # did not compare anyway
    assert "refus" in r["note"].lower()      # and said so
    assert "12" in r["note"] and "11" in r["note"]
    assert r["verdict"].startswith("INCONCLUSIVE")


# ── missing artifacts ────────────────────────────────────────────────────────

def test_load_json_names_a_missing_artifact(tmp_path):
    doc, err = C.load_json(tmp_path / "nope.json")
    assert doc is None
    assert "nope.json" in err
    assert "missing" in err.lower()


def test_main_names_the_missing_file_instead_of_crashing(capsys, tmp_path):
    rc = C.main([str(tmp_path / "gone.json"), str(tmp_path / "also_gone.json")])
    out = capsys.readouterr().out
    assert rc != 0
    assert "gone.json" in out


def test_main_rejects_an_unrecognised_shape(capsys, tmp_path):
    p = tmp_path / "weird.json"
    p.write_text('{"something": 1}')
    rc = C.main([str(p), str(p)])
    out = capsys.readouterr().out
    assert rc != 0
    assert "unrecognised" in out.lower()
    assert "something" in out


# ── the real artifacts ───────────────────────────────────────────────────────

def test_real_artifacts_of_both_shapes_compare_without_error():
    a = ROOT / "artifacts/eval_results.json"
    b = ROOT / "artifacts/hackathon_metrics.json"
    if not a.exists() or not b.exists():
        pytest.skip("eval artifacts not present")
    da, _ = C.load_json(a)
    db, _ = C.load_json(b)
    so, sn, rows = C.compare(da, db)
    assert so == "eval_results"
    assert sn == "hackathon"
    assert rows
    assert all(r["verdict"] for r in rows)
