"""
compare_runs.py — diff two eval artifacts and say whether a change is real.

WHY IT EXISTS
  Every later accuracy experiment is judged by diffing its artifact against the
  previous one. The repo has been burned twice by quoting a few-point gain over
  a dozen documents as a result, so this tool will not name a winner the sample
  cannot support: a change that is not bigger than the noise is INCONCLUSIVE.

WHAT IT COMPUTES
  For every named metric present in both artifacts:
    * old value, new value, delta;
    * a Wilson 95% interval for each rate, with the counts it rests on;
    * a verdict that is ONLY IMPROVED / REGRESSED / INCONCLUSIVE.

  The verdict is measured against the noise:
    * if the two runs scored the SAME items (paired), an EXACT McNemar test on
      the discordant pairs — p > 0.05 => INCONCLUSIVE — and the discordant
      counts (b/c) are printed;
    * if they did not score the same items, the two Wilson 95% intervals must
      be DISJOINT to claim a winner; overlap => INCONCLUSIVE;
    * a mean score (not a rate) has no interval to compare and is never called.

  Two identical artifacts therefore print delta 0 and INCONCLUSIVE everywhere.

HONESTY RULES (the same discipline as bench_hackathon.py)
  * a missing or unreadable artifact is NAMED, never defaulted;
  * an unrecognised artifact shape is named with its keys, never guessed at;
  * the counts a verdict rests on (n items, k successes) are always printed.

SHAPES HANDLED (detected by key presence, never by filename)
  * eval_results : artifacts/eval_results.json      (headline, per_profile, arms)
  * hackathon    : artifacts/hackathon_metrics.json (classifier, ocr, pipeline)

    bash run.sh py ml/src/compare_runs.py OLD.json NEW.json
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_hackathon as bh  # noqa: E402  (bh._recovered: one definition of the
#   "a masked value still counts as recovered" rule, shared with the bench)

ALPHA = 0.05
Z = 1.959963984540054  # 95%, two-sided


# ── statistics (stdlib only) ─────────────────────────────────────────────────

def wilson(k, n, z=Z):
    """Wilson score interval for a binomial proportion — the one that behaves at
    k=0 and k=n, unlike the normal approximation."""
    if not n or n <= 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value for `b` pairs correct->wrong and `c`
    wrong->correct. Under H0 each discordant pair is a fair coin, so
    p = 2 * P(X <= min(b, c)), X ~ Binomial(b + c, 0.5), capped at 1.0.
    Exact (not the chi-square approximation) because the discordant count here
    is small by construction."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2.0 ** n))


def _ci_overlap(a, b):
    if not a or not b:
        return True          # nothing to separate them on
    return a[0] <= b[1] and b[0] <= a[1]


# ── artifact loading & shape detection ───────────────────────────────────────

def load_json(path):
    """Return (doc, error). A missing or bad artifact is NAMED, never defaulted."""
    p = Path(path)
    if not p.exists():
        return None, "missing artifact: %s" % path
    try:
        return json.loads(p.read_text()), None
    except Exception as e:
        return None, "unreadable artifact: %s (%s)" % (path, e)


def detect_shape(doc):
    """Both shapes, by KEY PRESENCE — never by filename."""
    if not isinstance(doc, dict):
        return None
    if {"classifier", "pipeline", "missing"} <= set(doc) and "headline" not in doc:
        return "hackathon"
    if {"headline", "per_profile", "arms"} <= set(doc):
        return "eval_results"
    return None


# ── metric extraction ────────────────────────────────────────────────────────

def _mk(value, n, is_rate, higher_better):
    k = None
    if is_rate and isinstance(n, int) and n > 0:
        k = max(0, min(n, int(round(value * n))))
    return {"value": float(value), "n": n, "k": k,
            "is_rate": is_rate, "higher_better": higher_better}


def _records_items(records):
    """Per-item binary outcomes for the pipeline metrics that are defined on the
    very items their aggregate is: classification (one per document) and
    field_recovery (one per expected field, recovered or not).

    Identical item ids across two runs is what LICENSES the paired McNemar test;
    a run that scored different documents simply will not match, and the test is
    then refused rather than run on apples and oranges."""
    cls, fld, seen = {}, {}, {}
    for r in records:
        base = "%s|%s" % (r.get("profile"), r.get("doc"))
        i = seen.get(base, 0)
        seen[base] = i + 1
        uid = "%s#%d" % (base, i)
        cat, exp = r.get("category"), r.get("expected_category")
        cls[uid] = bool(cat) and cat == exp
        got = r.get("fields") or []
        for j, e in enumerate(r.get("expected_fields") or []):
            fld["%s#f%d" % (uid, j)] = (not r.get("error")) and any(
                bh._recovered(e, g) for g in got)
    out = {"classification@pipeline": cls}
    if fld:
        out["field_recovery@pipeline"] = fld
    return out


def extract(doc):
    """(shape, metrics, items). `metrics` is name -> value/counts; `items` is
    name -> {item_id: succeeded}, populated only where a per-item outcome
    exists (the hackathon pipeline records)."""
    shape = detect_shape(doc)
    metrics, items = {}, {}
    if shape == "eval_results":
        arms = doc.get("arms") or sorted(doc.get("headline", {}))
        for arm in arms:
            h = (doc.get("headline") or {}).get(arm)
            if not isinstance(h, dict):
                continue
            n = h.get("n_docs")
            for key, hb in (("field_accuracy", True), ("word_accuracy", True),
                            ("exact_dosage_rate", True),
                            ("mean_ocr_confidence", True)):
                if h.get(key) is not None:
                    metrics["%s@%s" % (key, arm)] = _mk(
                        h[key], n, key != "mean_ocr_confidence", hb)
    elif shape == "hackathon":
        cl = doc.get("classifier") or {}
        if cl.get("accuracy_mean") is not None:
            metrics["accuracy@classifier"] = _mk(
                cl["accuracy_mean"], cl.get("n_test"), True, True)
        if cl.get("macro_f1_mean") is not None:
            metrics["macro_f1@classifier"] = _mk(
                cl["macro_f1_mean"], cl.get("n_test"), True, True)
        oc = doc.get("ocr") or {}
        if oc.get("test_word_acc") is not None:
            metrics["word_accuracy@ocr_crops"] = _mk(
                oc["test_word_acc"], oc.get("test_crops"), True, True)
        pl = doc.get("pipeline") or {}
        head = pl.get("headline") if isinstance(pl.get("headline"), dict) else pl
        if head.get("n") is not None:
            for key, hb in (("field_recovery", True), ("classification", True),
                            ("word_accuracy", True), ("cer", False)):
                if head.get(key) is not None:
                    metrics["%s@pipeline" % key] = _mk(
                        head[key], head.get("n"), True, hb)
        if isinstance(pl.get("records"), list):
            items = _records_items(pl["records"])
    return shape, metrics, items


# ── the comparison & the verdict ─────────────────────────────────────────────

def _verdict(row, alpha):
    delta = row["delta"]
    hb = row["higher_better"]
    p = row["p"]
    if row["method"] == "mcnemar":
        sig = p is not None and p <= alpha
        if delta != 0 and sig:          # a zero delta is never a win, whatever p
            return ("IMPROVED" if (delta > 0) == hb else "REGRESSED") \
                + " (p=%.4g)" % p
        return "INCONCLUSIVE (p=%.4g)" % p
    if row["method"] == "wilson":
        if delta != 0 and not _ci_overlap(row["old_ci"], row["new_ci"]):
            return ("IMPROVED" if (delta > 0) == hb else "REGRESSED") \
                + " (95% CIs disjoint)"
        return "INCONCLUSIVE (95% CIs overlap)"
    return "INCONCLUSIVE (no test: not a rate)"


def compare(old_doc, new_doc, alpha=ALPHA):
    """(shape_old, shape_new, rows). One row per metric name present in either
    artifact; a metric present in only one gets N/A, not a verdict."""
    so, mo, io = extract(old_doc)
    sn, mn, inn = extract(new_doc)

    rows = []
    for name in sorted(set(mo) | set(mn)):
        o, n = mo.get(name), mn.get(name)
        row = {"metric": name, "method": "none", "discordant": None, "p": None,
               "note": "", "old_ci": None, "new_ci": None, "n": None,
               "k_old": None, "k_new": None, "delta": None,
               "old": o["value"] if o else None,
               "new": n["value"] if n else None,
               "higher_better": (o or n)["higher_better"]}
        if o is None or n is None:
            row["verdict"] = "N/A (only in %s)" % ("new" if o is None else "old")
            rows.append(row)
            continue

        row["delta"] = n["value"] - o["value"]

        # counts / intervals per side: prefer the item-derived counts (the true
        # denominator), fall back to the artifact's own n.
        for side, m, its in (("old", o, io.get(name)), ("new", n, inn.get(name))):
            if its:
                k, nn = sum(its.values()), len(its)
                row[side + "_ci"] = wilson(k, nn)
            else:
                k, nn = m["k"], m["n"]
                row[side + "_ci"] = wilson(k, nn) if (m["is_rate"] and nn) else None
            if side == "old":
                row["n"], row["k_old"] = nn, k
            else:
                row["k_new"] = k

        # paired test only when both runs scored the SAME items.
        oi, ni = io.get(name), inn.get(name)
        if oi is not None and ni is not None:
            if set(oi) == set(ni):
                b = sum(1 for i in oi if oi[i] and not ni[i])   # right -> wrong
                c = sum(1 for i in oi if not oi[i] and ni[i])   # wrong -> right
                row["method"] = "mcnemar"
                row["discordant"] = (b, c)
                row["p"] = mcnemar_exact(b, c)
            else:
                row["note"] = ("paired test refused: the runs scored different "
                               "items (old n=%d, new n=%d) — judging by interval "
                               "overlap instead" % (len(oi), len(ni)))
                row["method"] = "wilson" if (o["is_rate"] and n["is_rate"]) else "none"
        else:
            row["method"] = "wilson" if (o["is_rate"] and n["is_rate"]) else "none"

        row["verdict"] = _verdict(row, alpha)
        rows.append(row)
    return so, sn, rows


# ── output ───────────────────────────────────────────────────────────────────

def _f(x, signed=False):
    if x is None:
        return "n/a"
    return ("%+.4f" if signed else "%.4f") % x


def _ci(ci):
    return "-" if not ci else "[%.3f,%.3f]" % ci


def render(old_path, new_path, so, sn, rows):
    out = ["compare_runs: OLD %s  [shape=%s]" % (old_path, so or "unrecognised"),
           "              NEW %s  [shape=%s]" % (new_path, sn or "unrecognised"),
           ""]
    out.append("%-34s %8s %8s %8s %6s %11s %-17s %-17s  %s"
               % ("metric", "old", "new", "delta", "n", "k(old/new)",
                  "old 95% CI", "new 95% CI", "verdict"))
    out.append("-" * 150)
    for r in rows:
        disc = ""
        if r["discordant"]:
            disc = "  pairs %d/%d" % r["discordant"]
        out.append("%-34s %8s %8s %8s %6s %11s %-17s %-17s  %s%s"
                   % (r["metric"], _f(r["old"]), _f(r["new"]),
                      _f(r["delta"], signed=True), r["n"] if r["n"] is not None else "-",
                      "%s/%s" % (r["k_old"] if r["k_old"] is not None else "-",
                                 r["k_new"] if r["k_new"] is not None else "-"),
                      _ci(r["old_ci"]), _ci(r["new_ci"]), r["verdict"], disc))
    notes = [(r["metric"], r["note"]) for r in rows if r["note"]]
    if notes:
        out.append("")
        out.append("notes:")
        for m, t in notes:
            out.append("  %s: %s" % (m, t))
    tally = Counter(r["verdict"].split(" ")[0] for r in rows)
    out.append("")
    out.append("summary: %d metrics — %s" % (
        len(rows), ", ".join("%s=%d" % (k, v) for k, v in sorted(tally.items()))))
    out.append("verdict: IMPROVED/REGRESSED only when the paired exact McNemar "
               "p<=0.05, or (unpaired) the two 95% Wilson intervals are disjoint; "
               "otherwise INCONCLUSIVE.")
    return "\n".join(out)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: compare_runs.py OLD.json NEW.json")
        return 2
    old_doc, err = load_json(argv[0])
    if err:
        print("MISSING/UNREADABLE — " + err)
        return 2
    new_doc, err = load_json(argv[1])
    if err:
        print("MISSING/UNREADABLE — " + err)
        return 2

    so, sn, rows = compare(old_doc, new_doc)
    for label, path, shape, doc in (("OLD", argv[0], so, old_doc),
                                    ("NEW", argv[1], sn, new_doc)):
        if shape is None:
            print("unrecognised artifact shape: %s %s (keys: %s)"
                  % (label, path, sorted(doc)[:12]))
            return 3

    print(render(argv[0], argv[1], so, sn, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
