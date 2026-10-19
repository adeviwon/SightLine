"""
test_bench_hackathon.py — the Imperial College London Hackathon scoring report.

The brief asks for specific things to be MEASURED, so the benchmark that reports
them is itself under test:

  * step 3 "Test and evaluate" requires the CLASSIFIER's accuracy and macro-F1
    (held-out), and the OCR's word accuracy and CER on held-out data;
  * every number must be traceable to an artifact or a command, never typed by
    hand into a document (a judge can re-run it);
  * the report must not silently invent a number when an artifact is missing —
    it has to say which measurement is absent.

These tests assert the report's SHAPE and its traceability, not the values: the
values come from the artifacts and change when the models are retrained.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bench_hackathon as B  # noqa: E402


# ── the scoring primitives ───────────────────────────────────────────────────

def test_cer_counts_insertions_deletions_and_substitutions():
    # "abcd" -> "abxd": one substitution over 4 reference characters.
    assert B.cer("abcd", "abxd") == pytest.approx(0.25)
    # one insertion
    assert B.cer("abc", "abxc") == pytest.approx(1 / 3)
    # one deletion
    assert B.cer("abcd", "abd") == pytest.approx(0.25)
    # identical
    assert B.cer("abc", "abc") == 0.0


def test_cer_and_word_accuracy_agree_with_evaluate_on_normalised_text():
    import evaluate
    gt = "Account Number 40218877"
    hyp = "account number 40218877"
    # evaluate.word_accuracy is whitespace-token set membership after norm()
    assert B.word_accuracy(gt, hyp) == pytest.approx(evaluate.word_accuracy(gt, hyp))
    assert B.cer(evaluate.norm(gt), evaluate.norm(hyp)) == 0.0


def test_score_pair_returns_both_metrics():
    s = B.score_pair("Sort Code 40-11-04", "SOR CODE 401104")
    assert set(s) == {"word_accuracy", "cer", "gt_words", "gt_chars"}
    assert 0.0 <= s["word_accuracy"] <= 1.0
    assert s["cer"] >= 0.0


# ── the report ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def report():
    return B.build_report(include_pipeline=False)


def test_report_exposes_the_two_required_classifier_metrics(report):
    c = report["classifier"]
    assert 0.0 < c["accuracy_mean"] <= 1.0
    assert 0.0 < c["macro_f1_mean"] <= 1.0
    assert c["n_test"] > 0
    assert len(c["seeds"]) >= 2, "a single seed is a lucky seed, not a metric"


def test_report_exposes_the_held_out_ocr_metrics(report):
    o = report["ocr"]
    assert 0.0 < o["test_word_acc"] <= 1.0
    assert o["params"] > 0
    assert o["charset"], "the output alphabet is part of the claim"


def test_every_number_carries_its_source(report):
    """No hand-typed numbers: each block names the artifact it came from."""
    for block in ("classifier", "ocr"):
        assert report[block]["source"], block + " has no source"
        assert Path(ROOT / report[block]["source"]).exists(), report[block]["source"]


def test_report_maps_the_brief_requirements_to_evidence(report):
    req = report["requirements"]
    for key in ("understand_the_problem", "build_the_ai_tool",
                "test_and_evaluate", "check_for_fairness", "present"):
        assert key in req, "the five brief steps must all be mapped"
        assert req[key]["evidence"], key + " maps to nothing"


def test_missing_artifact_is_reported_as_absent_not_invented(tmp_path):
    """A missing measurement must be named, never defaulted to a plausible 0.0."""
    r = B.build_report(include_pipeline=False, artifacts_dir=tmp_path)
    assert r["ok"] is False
    assert r["missing"], "nothing was reported missing although no artifact exists"
    assert r["classifier"] is None and r["ocr"] is None


def test_written_report_is_valid_json_with_a_verdict(tmp_path):
    out = tmp_path / "hackathon_metrics.json"
    B.write_report(out, include_pipeline=False)
    d = json.loads(out.read_text())
    assert d["ok"] is True
    assert d["requirements"]["test_and_evaluate"]["evidence"]


# ── the shipped-pipeline aggregate ───────────────────────────────────────────
# The browser bench (app/tools/ios_full_check.js --json) writes RAW per-sample
# records: what the app classified, what fields it extracted, and the text it
# read. It deliberately does NOT compute metrics — the metric definitions live in
# one place (here), so the browser and the Python cannot drift apart.

RECORDS = [
    {
        "doc": "banking", "profile": "studio_clean",
        "expected_category": "banking", "expected_fields": ["40218877"],
        "expected_text": "Account Number 40218877",
        "category": "banking", "fields": ["40218877"],
        "ocr_text": "ACCOUNT NUMBER 40218877", "error": None, "ms": 900,
    },
    {
        "doc": "legal", "profile": "studio_clean",
        "expected_category": "legal", "expected_fields": ["2500"],
        "expected_text": "Claim 2500",
        "category": "general", "fields": [],
        "ocr_text": "CLAM 25OO", "error": None, "ms": 1100,
    },
]


def test_aggregate_pipeline_scores_fields_words_and_cer():
    a = B.aggregate_pipeline(RECORDS)
    assert a["n"] == 2
    assert a["field_recovery"] == pytest.approx(0.5)      # 1 of 2 documents
    assert a["classification"] == pytest.approx(0.5)      # 1 of 2 documents
    assert a["word_accuracy"] == pytest.approx(0.5)       # (1.0 + 0.0) / 2
    assert a["cer"] == pytest.approx((0.0 + 3 / 9) / 2)   # CLAIM2500 -> CLAM25OO
    assert set(a["by_profile"]) == {"studio_clean"}


def test_aggregate_pipeline_counts_failed_scans_as_zero_not_missing():
    recs = RECORDS + [{
        "doc": "prescription", "profile": "handheld_light",
        "expected_category": "medical", "expected_fields": ["500mg"],
        "expected_text": "Take 500mg daily",
        "category": None, "fields": [], "ocr_text": "",
        "error": "the OCR model failed to initialise", "ms": 200,
    }]
    a = B.aggregate_pipeline(recs)
    assert a["n"] == 3
    assert a["errors"] == 1
    assert a["field_recovery"] == pytest.approx(1 / 3)
    # a scan that ERRORED is a 0 for every metric, never skipped
    assert a["word_accuracy"] == pytest.approx((1.0 + 0.0 + 0.0) / 3)
    # a HANDHELD profile stays in the headline: hard is not the same as unreadable
    assert a["headline"]["n"] == 3 and a["sub_human"]["n"] == 0


def test_only_the_repos_sub_human_profiles_leave_the_headline():
    recs = RECORDS + [{
        "doc": "prescription", "profile": "worst_case",
        "expected_category": "medical", "expected_fields": ["500mg"],
        "expected_text": "Take 500mg daily",
        "category": "general", "fields": [], "ocr_text": "TAKE 5OOMG",
        "error": None, "ms": 300,
    }]
    a = B.aggregate_pipeline(recs)
    assert a["n"] == 3
    assert a["sub_human"]["n"] == 1
    assert a["headline"]["n"] == 2
    # the worst_case row cannot move the headline
    assert a["headline"]["field_recovery"] == pytest.approx(0.5)
    assert a["headline"]["classification"] == pytest.approx(0.5)


def test_aggregate_pipeline_reports_the_metric_definitions_it_used():
    a = B.aggregate_pipeline(RECORDS)
    assert a["metric_definitions"]["word_accuracy"].startswith("fraction of")
    assert "insertions" in a["metric_definitions"]["cer"]


def test_pipeline_block_is_absent_when_the_raw_records_are(tmp_path):
    r = B.build_report(include_pipeline=True, artifacts_dir=tmp_path)
    assert r["pipeline"] is None
    assert any("pipeline" in m for m in r["missing"])


# ── the browser -> python contract ───────────────────────────────────────────
# ios_full_check.js --json writes the records; this file reads them. Two
# languages, one shape: if the JS side renames a field, the report would silently
# score zero. The contract is asserted against the real file when it exists.

RAW = ROOT / "artifacts/pipeline_raw.json"
RAW_KEYS = {"doc", "profile", "expected_category", "expected_fields",
            "expected_text", "category", "fields", "ocr_text",
            "ocr_confidence", "error", "ms"}


@pytest.mark.skipif(not RAW.exists(), reason="run `bash run.sh bench-pipeline` first")
def test_browser_records_match_the_contract_this_module_scores():
    recs = json.loads(RAW.read_text())
    assert recs, "the bench wrote no records"
    for r in recs:
        missing = RAW_KEYS - set(r)
        assert not missing, "%s record is missing %s" % (r.get("doc"), sorted(missing))
    # and the samples the manifest declares must all be present exactly once
    man = json.loads(Path("/tmp/sightline_samples/manifest.json").read_text())
    want = {(s["doc"], s["profile"]) for s in man["samples"]}
    got = [(r["doc"], r["profile"]) for r in recs]
    assert sorted(got) == sorted(want), "records do not match the manifest's samples"


@pytest.mark.skipif(not RAW.exists(), reason="run `bash run.sh bench-pipeline` first")
def test_the_report_scores_the_real_records():
    a = B.aggregate_pipeline(json.loads(RAW.read_text()))
    assert a["n"] == 12
    assert a["headline"]["n"] + a["sub_human"]["n"] == 12
    # every metric is a rate in [0, 1]; a broken aggregate would drift outside
    for block in (a, a["headline"], a["sub_human"]):
        for k in ("field_recovery", "classification", "word_accuracy", "cer"):
            assert 0.0 <= block[k] <= 1.0, (k, block[k])
