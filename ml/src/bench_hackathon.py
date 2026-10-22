"""
bench_hackathon.py — the Imperial College London Hackathon scoring report.

WHAT THE BRIEF ASKS TO BE MEASURED, and where each number comes from:

  step 3 "Test and evaluate"
    * classifier accuracy + macro-F1 on held-out data, over several seeds
      -> artifacts/step3_classifier_metrics.json  (ml/experiments/step3_f1.py)
    * the recogniser's word accuracy on held-out data, and its size/alphabet
      -> artifacts/ocr_training.json              (bash run.sh train-ocr)
    * the recogniser's word accuracy + CER against whole held-out pages,
      detector included
      -> bash run.sh eval-ocr                     (printed; see docs/03 section 3)
  the shipped pipeline end to end (the browser app, WebKit = iOS Safari's engine)
      -> artifacts/pipeline_raw.json              (bash run.sh bench-pipeline)
         artifacts/pipeline_raw_shipped.json      (the SHIPPED build's saved records, on the
                                                  CURRENT bench fixtures -- what this report scores)
         artifacts/pipeline_raw_deskew.json       (the previous fixture generation, kept as a record)

This module does not measure anything itself: it READS those artifacts and
reports them in one place, with the source of every number attached, and it
NAMES anything that is missing rather than defaulting it to a plausible zero. A
judge should be able to re-run each command and get the same numbers, so a
hand-typed figure in a document is the failure mode this exists to prevent.

    bash run.sh bench-hackathon              # report (reads what exists)
    bash run.sh bench-hackathon --run-pipeline   # also run the browser bench first
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate  # noqa: E402  (norm/word_accuracy live there: one definition)


# ── metrics ──────────────────────────────────────────────────────────────────

def cer(ref, hyp):
    """Character error rate: edit distance / reference length.

    Insertions, deletions and substitutions each cost 1. Standard definition, and
    the same one eval_ocr.py reports for whole pages, so the two are comparable.
    """
    r, h = str(ref), str(hyp)
    if not r:
        return 0.0 if not h else float(len(h))
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1,              # deletion
                           cur[j - 1] + 1,           # insertion
                           prev[j - 1] + (rc != hc)))  # substitution
        prev = cur
    return prev[-1] / len(r)


def word_accuracy(gt, hyp):
    """Identical to evaluate.word_accuracy — imported, not reimplemented."""
    return evaluate.word_accuracy(gt, hyp)


def score_pair(gt, hyp):
    """Both required text metrics for one (ground truth, hypothesis) pair.

    CER runs on NORMALISED text: `norm()` strips everything that is not [a-z0-9],
    and the OCR alphabet has no punctuation by design, so scoring raw text would
    charge the model for a period it is not allowed to emit.
    """
    return {
        "word_accuracy": word_accuracy(gt, hyp),
        "cer": cer(evaluate.norm(gt), evaluate.norm(hyp)),
        "gt_words": len(gt.split()),
        "gt_chars": len(evaluate.norm(gt)),
    }


# ── the report ───────────────────────────────────────────────────────────────

CLS_ART = "artifacts/step3_classifier_metrics.json"
OCR_ART = "artifacts/ocr_training.json"

REQUIREMENTS = {
    "understand_the_problem": {
        "brief": "Step 1 — the problem, who is affected, what data the AI needs",
        "evidence": ["docs/12_HACKATHON_SUBMISSION.md#1-understand-the-problem",
                     "docs/13_ROUND1_SUBMISSION_KIT.md"],
    },
    "build_the_ai_tool": {
        "brief": "Step 2 — the model, the training data, how it is deployed",
        "evidence": ["docs/12_HACKATHON_SUBMISSION.md#2-build-the-ai-tool",
                     "docs/01_ARCHITECTURE.md",
                     "ml/src/ocr_model.py", "ml/src/train_ocr.py"],
    },
    "test_and_evaluate": {
        "brief": "Step 3 — the required metrics, on held-out data, reproducible",
        "evidence": ["artifacts/step3_classifier_metrics.json",
                     "artifacts/ocr_training.json",
                     "artifacts/pipeline_raw_shipped.json",
                     "docs/03_RESULTS.md#3-results", "docs/14_HACKATHON_BENCHMARK.md"],
        "commands": ["bash run.sh bench-hackathon", "bash run.sh eval-ocr",
                     "bash run.sh bench-pipeline"],
    },
    "check_for_fairness": {
        "brief": "Step 4 — who the tool would exclude, and what we would do",
        "evidence": ["docs/12_HACKATHON_SUBMISSION.md#4-check-for-fairness",
                     "ml/src/privacy_audit.py"],
    },
    "present": {
        "brief": "Step 5 — the pitch, the demo, the deployment path",
        "evidence": ["docs/13_ROUND1_SUBMISSION_KIT.md",
                     "docs/11_JUDGE_TECHNICAL_DEEP_DIVE.md",
                     "docs/PRIVACY.md"],
    },
}


def _read(artifacts_dir, rel):
    p = Path(artifacts_dir) / Path(rel).name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def build_report(include_pipeline=True, artifacts_dir=None):
    """Collect every required number, with its source, and name what is absent.

    `artifacts_dir` is the directory the artifacts live in (default: the repo's
    artifacts/), so a test can point it at an empty directory and assert that the
    report names what is missing instead of inventing it.
    """
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else ROOT / "artifacts"
    missing = []
    report = {
        "ok": False,
        "requirements": REQUIREMENTS,
        "classifier": None,
        "ocr": None,
        "pipeline": None,
        "missing": missing,
        "notes": [],
    }

    cls = _read(artifacts_dir, CLS_ART)
    if cls and "accuracy_mean" in cls:
        report["classifier"] = {
            "accuracy_mean": cls["accuracy_mean"],
            "macro_f1_mean": cls["macro_f1_mean"],
            "accuracy_sd": cls.get("accuracy_sd"),
            "macro_f1_sd": cls.get("macro_f1_sd"),
            "seeds": cls.get("seeds", []),
            "n_test": cls.get("n_test", 0),
            "saturated": cls.get("saturated"),
            "caveat": cls.get("caveat"),
            "source": CLS_ART,
        }
    else:
        missing.append("classifier metrics (" + CLS_ART + ")")

    ocr = _read(artifacts_dir, OCR_ART)
    if ocr and "test_word_acc" in ocr:
        report["ocr"] = {
            "test_word_acc": ocr["test_word_acc"],
            "best_val_word_acc": ocr.get("best_val_word_acc"),
            "params": ocr["params"],
            "charset": ocr["charset"],
            "crops_total": ocr.get("crops_total"),
            "test_crops": ocr.get("test_crops"),
            "epochs": ocr.get("epochs"),
            "source": OCR_ART,
            "page_level": {
                "command": "bash run.sh eval-ocr",
                "note": "word accuracy + CER on whole held-out pages, detector "
                        "included; the printed table in docs/03 section 3 records "
                        "the 20-page hand-photographed set at 67.9% line recall, "
                        "1.8% line-exact, 56.5% CER (the 193-page set-B "
                        "reproduction is 71.6% / 2.4% / 58.4%, HANDOFF.md "
                        "section 3) — hand-held photographs are the honest hard "
                        "case, NOT 58.4% CER alone",
            },
        }
    else:
        missing.append("recogniser metrics (" + OCR_ART + ")")

    if include_pipeline:
        raw = _read(artifacts_dir, PIPE_RAW)
        if raw and isinstance(raw, list):
            report["pipeline"] = aggregate_pipeline(raw)
        elif raw and raw.get("records"):
            report["pipeline"] = aggregate_pipeline(raw["records"])
        else:
            missing.append("shipped-pipeline bench (" + PIPE_RAW
                           + ") — the shipped build's saved records; write a fresh "
                           "run with `bash run.sh bench-pipeline` -> "
                           + PIPE_BENCH_OUT)

    report["notes"].append(
        "The classifier figure is the held-out n=%s number over %s seeds. The "
        "end-to-end pipeline numbers are behaviour on synthetic renders: the "
        "recogniser never trained on them, but the classifier's templates ARE in "
        "its corpus, so its end-to-end agreement is not a generalisation claim."
        % (report["classifier"]["n_test"] if report["classifier"] else "?",
           len(report["classifier"]["seeds"]) if report["classifier"] else "?"))

    report["ok"] = bool(report["classifier"]) and bool(report["ocr"])
    return report


def write_report(out_path, include_pipeline=True, artifacts_dir=None):
    report = build_report(include_pipeline=include_pipeline, artifacts_dir=artifacts_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _pct(x):
    return "n/a" if x is None else "%.1f%%" % (100 * x)


# ── the shipped pipeline (browser bench, raw records -> metrics) ──────────────
# ios_full_check.js drives the real app in WebKit (= iOS Safari's engine) and
# writes ONE RECORD PER SAMPLE: what it classified, what fields it extracted, and
# the text it read. It computes no metrics on purpose — the definitions below are
# the only definitions, so the browser and this file cannot drift apart.

# Which records this report SCORES: the SHIPPED build's saved run. A fresh
# `bench-pipeline` run goes to PIPE_BENCH_OUT, so re-running the browser bench
# cannot silently overwrite the shipped record this file reports on.
PIPE_RAW = "artifacts/pipeline_raw_shipped.json"   # the SHIPPED build on the CURRENT
#   (regenerated) bench fixtures. This pointed at pipeline_raw_deskew.json, which was the
#   shipped build at the time but was measured on fixtures rendered BEFORE the capture
#   profiles were re-tuned -- 17-67x harsher than real photographs -- so the repo's own
#   one-command reproduction printed a number from a distribution the model never trained
#   on. ml/tests/test_fixture_freshness.py fails if the fixtures drift again.
PIPE_BENCH_OUT = "artifacts/pipeline_raw.json"

# The ONLY profiles kept out of the headline are the ones the repo already marks
# as unreadable to a human (evaluate.SUB_HUMAN): a photo nobody can read is not
# evidence about the model. Everything else — including the handheld and glare
# profiles — stays IN the headline, and the per-profile table below shows how
# each one did. No second, invented exclusion set: assess_profile-style bucketing
# that flatters the average is how a benchmark stops meaning anything.
SUB_HUMAN = set(evaluate.SUB_HUMAN)

METRIC_DEFINITIONS = {
    "word_accuracy": "fraction of ground-truth words present in the read text, "
                     "compared after norm() (evaluate.word_accuracy)",
    "cer": "character error rate: (substitutions + insertions + deletions) / "
           "ground-truth characters, on norm()ed text",
    "field_recovery": "fraction of the document's ground-truth fields present in "
                      "the extracted fields (a masked value counts if it is a "
                      "suffix of the expected one)",
    "classification": "fraction of documents whose category matches the "
                      "expected category",
}


def _recovered(expected, got):
    """Is `expected` present in the extracted fields?

    The app is allowed to MASK a sensitive value (it keeps the last four digits),
    which is a product decision, not a miss — so a short extracted value counts
    when it is a suffix of the expected one. Anything shorter than 4 characters
    has to match exactly, or "0" would "recover" every number containing a zero.
    """
    e, g = evaluate.norm(expected), evaluate.norm(got)
    if not e or not g:
        return False
    return e == g or (len(g) >= 4 and e.endswith(g))


def _rate_metrics(recs):
    n = len(recs)
    if not n:
        return {"n": 0, "field_recovery": 0.0, "classification": 0.0,
                "word_accuracy": 0.0, "cer": 0.0, "errors": 0}
    fields = 0.0
    got_fields = 0.0
    cls = 0.0
    wa = 0.0
    ce = 0.0
    errs = 0
    for r in recs:
        exp = r.get("expected_fields") or []
        got_fields += len(exp)             # the denominator counts a failed scan
        if r.get("error"):
            errs += 1                      # a failed scan scores 0, it is not skipped
            continue
        got = r.get("fields") or []
        fields += sum(1 for e in exp if any(_recovered(e, g) for g in got))
        cls += 1.0 if r.get("category") and r.get("category") == r.get("expected_category") else 0.0
        wa += evaluate.word_accuracy(r.get("expected_text") or "", r.get("ocr_text") or "")
        ce += cer(evaluate.norm(r.get("expected_text") or ""),
                  evaluate.norm(r.get("ocr_text") or ""))
    return {
        "n": n,
        "errors": errs,
        "field_recovery": fields / got_fields if got_fields else 0.0,
        "classification": cls / n,
        "word_accuracy": wa / n,
        "cer": ce / n,
    }


def aggregate_pipeline(records):
    """Turn raw browser records into the required metrics.

    Totals cover EVERY record, errors included as zeros. `headline` is the same
    measure with only the repo's SUB_HUMAN profiles removed (a photo no human can
    read is not evidence about the model); `by_profile` is the detail behind it.
    """
    recs = list(records)
    good = [r for r in recs if r.get("profile") not in SUB_HUMAN]
    bad = [r for r in recs if r.get("profile") in SUB_HUMAN]
    by_profile = {}
    for p in sorted({r.get("profile") for r in recs}):
        by_profile[p] = _rate_metrics([r for r in recs if r.get("profile") == p])
    out = _rate_metrics(recs)
    out.update({
        "headline": _rate_metrics(good),
        "headline_profiles": sorted({r.get("profile") for r in good}),
        "sub_human": _rate_metrics(bad),
        "sub_human_profiles": sorted({r.get("profile") for r in bad}),
        "by_profile": by_profile,
        "metric_definitions": METRIC_DEFINITIONS,
        "source": PIPE_RAW,
        "records": [{"doc": r.get("doc"), "profile": r.get("profile"),
                     "expected_category": r.get("expected_category"),
                     "category": r.get("category"),
                     "expected_fields": r.get("expected_fields"),
                     "fields": r.get("fields"),
                     "ocr_confidence": r.get("ocr_confidence"),
                     "error": r.get("error"), "ms": r.get("ms")}
                    for r in recs],
    })
    return out


def run_pipeline_bench():
    """Run the browser bench (WebKit) that writes artifacts/pipeline_raw.json.

    `--suites=bench` is the suite that emits one record per sample (`R.record` is
    only called there); `--suites=scan` runs the two scan suites, which emit no
    records, so the JSON would come out empty. This matches `run.sh bench-pipeline`.
    """
    cmd = ["node", "app/tools/ios_full_check.js", "--engine=webkit",
           "--suites=bench", "--json=" + PIPE_BENCH_OUT]
    print("[bench] " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/hackathon_metrics.json"))
    ap.add_argument("--run-pipeline", action="store_true",
                    help="run the browser bench first (WebKit, ~5 min)")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="skip the shipped-pipeline block entirely")
    a = ap.parse_args()

    if a.run_pipeline:
        run_pipeline_bench()

    rep = write_report(a.out, include_pipeline=not a.no_pipeline)

    print("\n=== Imperial brief, step 3 — required metrics ===")
    c = rep["classifier"]
    print("  classifier accuracy   : %s  (n=%s, %s seeds)   [%s]"
          % (_pct(c["accuracy_mean"]) if c else "MISSING",
             c["n_test"] if c else "-", len(c["seeds"]) if c else "-", CLS_ART))
    print("  classifier macro-F1   : %s"
          % (_pct(c["macro_f1_mean"]) if c else "MISSING"))
    o = rep["ocr"]
    print("  recogniser word acc   : %s  (held-out crops, %s params)  [%s]"
          % (_pct(o["test_word_acc"]) if o else "MISSING",
             o["params"] if o else "-", OCR_ART))
    print("  recogniser whole-page : run `bash run.sh eval-ocr` (word acc + CER,")
    print("                          detector included) — see docs/03 section 3")
    p = rep["pipeline"]
    if p:
        h, d = p["headline"], p["sub_human"]
        print("  shipped app end-to-end (readable profiles, n=%s):" % h["n"])
        print("      field recovery %s   classification %s   word acc %s   CER %s"
              % (_pct(h["field_recovery"]), _pct(h["classification"]),
                 _pct(h["word_accuracy"]), _pct(h["cer"])))
        if d["n"]:
            # sub_human_profiles, not "degraded_profiles": the latter key does not
            # exist on aggregate_pipeline's output, so this branch used to raise
            # KeyError the moment a bench included a SUB_HUMAN profile.
            print("    sub-human profiles (reported, not counted; n=%s: %s):"
                  % (d["n"], ", ".join(p["sub_human_profiles"])))
            print("      field recovery %s   classification %s   word acc %s   CER %s"
                  % (_pct(d["field_recovery"]), _pct(d["classification"]),
                     _pct(d["word_accuracy"]), _pct(d["cer"])))
    else:
        print("  shipped app end-to-end: MISSING — run `bash run.sh bench-pipeline`")

    if rep["missing"]:
        print("\nABSENT (named, not defaulted):")
        for m in rep["missing"]:
            print("  - " + m)
    print("\nwrote " + a.out)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
