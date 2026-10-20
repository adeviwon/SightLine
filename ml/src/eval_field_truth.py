"""
eval_field_truth.py — the CORRECT ground truth for field recovery.

WHY THIS EXISTS (the two measurements it replaces)
--------------------------------------------------
Field recovery has now been measured twice and both numbers were unusable,
because the ground truth could not express the task:

  * `evaluate.FIELDS` lists TWO tokens per document type ("500mg","400mg") while
    the extractor deliberately speaks ~8 TYPED fields (patient, prescriber,
    pharmacy, medication, dates...). Scoring spoken values against that list
    gives precision 0 by construction.
  * recovery counted only byte-identical (or masked-suffix) matches, so a dosage
    read correctly and spoken as "AMOXICILLIN 500MG CAPSULES" scored a MISS.

Net effect: an extractor that transcribed a page character-for-character
reported 0/300.

THE RIGHT GROUND TRUTH IS ALREADY ON DISK. `data/render_corpus/crops.npz`
carries the EXACT text that was drawn for every line crop (labels are drawn by
`capture.render_document`, never detected). So for every label the pipeline
spots on a line, the correct VALUE is the continuation of that line's true text
after that same label. One ground-truth value per spotted field, produced by the
renderer; neither path can game it.

WHAT IS MEASURED (per capture profile and overall, at label_spot 0.75 and 0.833)
  (a) BASELINE     — the whole-page regex that ships today: evaluate.fields_found
                     over the concatenated OCR text (the Python twin of
                     app/js/pipeline.js).
  (b) SPOT-EXTRACT — field_proximity.extract_fields(lines, spots) values, scored
                     no matter what validation said.
  (c) SPOT+VALIDATE— only the values that came back ok=True (the product rule).

  For each: value PRECISION (of values SPOKEN, how many match the true
  continuation), value RECALL (of true label/value pairs present, how many the
  path produced correctly), and for (c) the refusal rate and — the product-safety
  number — WRONG-BUT-SPOKEN: values that passed validation and are WRONG.
  Wilson 95% CIs on every rate, with the counts. Paired exact McNemar between
  (a) and (c) on the same pages, judged in compare_runs.compare's own vocabulary
  (IMPROVED / REGRESSED / INCONCLUSIVE).

HOW A SPOT BECOMES GROUND TRUTH
  For each spot on a line, the spot's canonical label is located in the TRUE
  line text (case- and punctuation-insensitive). If it is ABSENT there, the spot
  is a FALSE POSITIVE and is counted as one — never skipped. Otherwise the
  ground-truth value is the true line's remainder after the label, truncated at
  the next label matched on the same line (that is the same span field_proximity
  reads), and an empty remainder carries no value.

    bash run.sh py ml/src/eval_field_truth.py --split test --limit-pages 120

--with-policy (Lane N) scores the SPEAK/REFUSE decision layer (field_policy.decide)
on the same pages and the same ground truth: recall, precision, refusals and
WRONG-BUT-SPOKEN per profile with Wilson 95% CIs and counts, paired exact
McNemar of the policy's recall against (a) and against (c), and the two
refusal buckets (spot right / regex wrong and the reverse) with their strings.
Without the flag every number above is unchanged.

WIRING is reused, not re-invented: render_corpus.load (corpus),
eval_render_corpus.load_model (torn-read guard), eval_field_recovery.read_page_lines
/ spot_lines / sha256_prefix, label_spot, field_proximity, evaluate,
bench_hackathon._recovered (the repo's one masked-value rule) and compare_runs
(wilson + the verdict rule).
"""

from __future__ import annotations

import argparse
import sys
from bisect import bisect_left
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import bench_hackathon as BH          # noqa: E402  (_recovered: one masked rule)
import compare_runs as CR             # noqa: E402  (wilson + the verdict rule)
import eval_field_recovery as FR      # noqa: E402  (read_page_lines/spot_lines/sha)
import eval_render_corpus as ER       # noqa: E402  (load_model: torn-read guard)
import evaluate as E                  # noqa: E402  (fields_found IS the baseline)
import field_policy as FPOL           # noqa: E402  (the speak/refuse decision layer)
import field_proximity as FP          # noqa: E402
import render_corpus as RC            # noqa: E402

DEFAULT_CKPT = "models/ocr/crnn.pt"          # the retrained recogniser
PAIRED_CKPT = "models/ocr/crnn_before_render.pt"   # the frozen one before it

# The two label_spot operating points: 0.75 is the precision-first default,
# 0.833 the strict 1-edit-per-6 setting. Both are reported; the choice is the
# product's.
THRESHOLDS = (0.75, 0.833)
PRIMARY = 0.75

PATHS = ("baseline", "spot", "valid")

# The policy variants measured in the SAME run (same pages, same ground truth).
# "policy" is the rule set in field_policy's docstring; the other two are the
# trade made explicit — what it costs to speak only what can be checked.
POLICY_VARIANTS = {
    "policy": {},
    "typed-only": {"speak_free_text": False},
    "typed+strict": {"strict_critical": True, "speak_free_text": False},
}

POLICY_NOTES = {
    "policy": "critical kinds speak on agreement or validated single path; free text "
              "speaks if in the doc-type vocabulary and not a fragment",
    "typed-only": "the same, minus free text: what the policy can CHECK is spoken, "
                  "what it cannot is refused",
    "typed+strict": "and safety-critical values speak only on two-path agreement",
}

# A substring counts as a match only when the shorter normalised string is at
# least this long: "0" must not "recover" every number containing a zero.
MIN_SUBSTRING = 4

wilson = CR.wilson          # re-exported: one Wilson definition, from compare_runs


# ── the match rule (one rule for every path) ──────────────────────────────────

def _norm(s):
    return E.norm(s)


def match_value(spoken, truth):
    """Does a SPOKEN value convey the true continuation?

    Case/punctuation-insensitive. Equal, or (>= 4 chars) one contains the other
    — a value spoken as "AMOXICILLIN 500MG" conveys "500MG", and the regex
    baseline's token "500MG" is contained in the whole traced line — or the
    repo's masked-suffix rule (the app may keep only the last four digits).
    A wrong-but-plausible value (the right shape, the wrong digits) is NOT
    rescued: nothing here tolerates a digit that differs.
    """
    ns, nt = _norm(spoken), _norm(truth)
    if not ns or not nt:
        return False
    if ns == nt:
        return True
    if len(ns) >= MIN_SUBSTRING and ns in nt:
        return True
    if len(nt) >= MIN_SUBSTRING and nt in ns:
        return True
    return BH._recovered(nt, ns)


# ── locating a spotted label inside the TRUE line text ────────────────────────

def _idx_map(text):
    """(normalised text, index of each normalised char in the original)."""
    nt, idx = [], []
    for i, ch in enumerate(str(text).upper()):
        low = ch.lower()
        if low.isdigit() or "a" <= low <= "z":
            nt.append(low)
            idx.append(i)
    return "".join(nt), idx


def label_span(label, true_text, start_at=0):
    """(start, end) of `label` in `true_text`, case/punctuation-insensitively.

    Searches at or after `start_at` (true-text coordinates) so two printed
    occurrences of the same label map to two distinct spans. Returns None when
    the label does not occur — that is the FALSE-POSITIVE signal.
    """
    nt, idx = _idx_map(true_text)
    nl = _norm(label)
    if not nt or not nl:
        return None
    p = nt.find(nl, bisect_left(idx, start_at))
    if p < 0:
        return None
    return idx[p], idx[p + len(nl) - 1] + 1


# ── the model-free core: one page, scored against the rendered truth ──────────

def ground_truth(true_texts, ocr_lines, threshold=PRIMARY):
    """([(label, value)], n_false_positive_spots) from the spots on a page.

    A spot's label is located in the true text of ITS OWN line; the value is the
    remainder after it up to the next matched label on that line. A spot whose
    label is absent from the true line is counted as a false positive.
    """
    spots = FR.spot_lines(ocr_lines, threshold)
    by_line = {}
    for sp in spots:
        by_line.setdefault(int(sp.get("line_index", 0)), []).append(sp)

    pairs, n_fp = [], 0
    for li, line_spots in by_line.items():
        if not 0 <= li < len(true_texts):
            n_fp += len(line_spots)          # a spot on a line with no truth
            continue
        true = str(true_texts[li])
        line_spots = sorted(line_spots,
                            key=lambda s: (int(s.get("start", 0)), int(s.get("end", 0))))
        matched, last_end = [], 0
        for sp in line_spots:
            span = label_span(sp.get("label", ""), true, last_end)
            if span is None:
                n_fp += 1
                continue
            matched.append((span[0], span[1], sp.get("label", "")))
            last_end = span[1]
        matched.sort(key=lambda m: m[0])
        for j, (s, e, lab) in enumerate(matched):
            nxt = matched[j + 1][0] if j + 1 < len(matched) else len(true)
            val = true[e:nxt].strip()
            if val:
                pairs.append((lab, val))
    return pairs, n_fp


def score_page(true_texts, ocr_lines, doc_type, threshold=PRIMARY,
               policy_variants=None):
    """One page -> the counts the three-path table rests on.

    ``true_texts`` are the rendered line texts (render_corpus labels), ``ocr_lines``
    the recognised {"text", "box"} dicts, in the same page order.
    ``policy_variants`` is None (no policy measured) or {name: decide(**kwargs)}:
    each variant is scored against the SAME ground truth, so the policy rows and
    the three paths above are comparable item for item (which is what licenses
    the paired test).
    """
    true_texts = list(true_texts or [])
    ocr_lines = list(ocr_lines or [])
    pairs, n_fp = ground_truth(true_texts, ocr_lines, threshold)
    gt_values = [v for _, v in pairs]

    text = " ".join(str(ln.get("text", "")) for ln in ocr_lines)
    spots = FR.spot_lines(ocr_lines, threshold)
    fields = FP.extract_fields(ocr_lines, spots)

    baseline_spoken = list(E.fields_found(text, doc_type))
    spot_spoken = [r["value"] for r in fields.values()]
    valid_spoken = [r["value"] for r in fields.values() if r["ok"]]

    def hits(spoken):
        return sum(1 for v in spoken if any(match_value(v, g) for g in gt_values))

    def recovered(spoken):
        return [g for g in gt_values if any(match_value(v, g) for v in spoken)]

    base_rec, spot_rec, valid_rec = (recovered(baseline_spoken),
                                     recovered(spot_spoken), recovered(valid_spoken))
    row = {
        "doc_type": doc_type, "threshold": threshold,
        "gt_pairs": pairs, "gt_values": gt_values, "n_gt": len(gt_values),
        "n_spots": len(spots), "n_fp_spots": n_fp,
        "baseline_spoken": baseline_spoken, "baseline_hit": hits(baseline_spoken),
        "baseline_recovered": base_rec,
        "spot_spoken": spot_spoken, "spot_hit": hits(spot_spoken),
        "spot_recovered": spot_rec,
        "valid_spoken": valid_spoken, "valid_hit": hits(valid_spoken),
        "valid_recovered": valid_rec,
        "valid_wrong": [v for v in valid_spoken
                        if not any(match_value(v, g) for g in gt_values)],
        "refused": sum(1 for r in fields.values() if not r["ok"]),
        "n_fields": len(fields),
        # eval_field_recovery.paired_vs_baseline / _artifact compatibility
        "expected": gt_values, "baseline": base_rec, "recovered": valid_rec,
    }
    if policy_variants:
        regex_fields = FPOL.from_regex_tokens(doc_type, baseline_spoken)
        row["policy"] = {}
        for name, kw in policy_variants.items():
            decisions = FPOL.decide(regex_fields, fields, doc_type=doc_type, **kw)
            scores = _policy_scores(decisions, gt_values, " ".join(true_texts),
                                    fields, regex_fields)
            row["policy"][name] = scores
            # flat alias so _artifact/paired can score this variant like any path
            row[_slug(name) + "_recovered"] = scores["recovered"]
    return row


def _slug(name):
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")


def _policy_scores(decisions, gt_values, true_text, fields, regex_fields):
    """One policy variant on one page.

    ``wrong`` is the product-safety number: values the policy SPOKE that convey
    no ground-truth value. ``wrong_printed`` is the honest sub-count — a value
    that IS printed on the page but has no ground-truth item because the label
    that would anchor it was never spotted. It is reported, never folded in.
    """
    def kind(d):
        return FPOL.kind_of(d["field"], fields.get(d["field"]))

    def right(v):
        return any(match_value(v, g) for g in gt_values)

    spoken = [d for d in decisions if d["decision"] == "speak"]
    hit = [d for d in spoken if right(d["value"])]
    wrong = [d for d in spoken if not right(d["value"])]
    rec = [g for g in gt_values if any(match_value(d["value"], g) for d in spoken)]

    by_kind, spoken_kind = {}, {}
    for d in wrong:
        by_kind[kind(d)] = by_kind.get(kind(d), 0) + 1
    for d in spoken:
        spoken_kind[kind(d)] = spoken_kind.get(kind(d), 0) + 1

    # The other path's values of THIS kind — what a refusal was weighed against.
    rival = {}
    for f, r in fields.items():
        rival.setdefault(FPOL.kind_of(f, r), []).append(str(r.get("value") or ""))
    rival_regex = {}
    for f, raw in (regex_fields or {}).items():
        vals = raw if isinstance(raw, (list, tuple)) else [raw]
        rival_regex.setdefault(FPOL.kind_of(f), []).extend(str(v) for v in vals)

    refusals = [d for d in decisions if d["decision"] == "refuse"]
    contradictions, lost, b1, b2 = [], [], [], []
    for d in refusals:
        k = kind(d)
        others = (rival_regex if d["source"] == "spot" else rival).get(k, [])
        other_right = any(right(o) for o in others if o)
        d_right = bool(d["value"]) and right(d["value"])
        if "disagree" in d["why"]:
            contradictions.append((d, others))
        if d_right:
            lost.append(d)
        if d["source"] == "both":
            continue                      # both paths said it: no rival to weigh
        if d_right and not other_right:
            b1.append((d, others))
        elif other_right and not d_right:
            b2.append((d, others))
    return {"spoken": [d["value"] for d in spoken], "recovered": rec,
            "n_spoken": len(spoken), "n_hit": len(hit),
            "wrong": [d["value"] for d in wrong],
            "wrong_printed": [d["value"] for d in wrong if match_value(d["value"], true_text)],
            "refused": len(refusals), "n": len(decisions),
            "by_kind": by_kind, "spoken_kind": spoken_kind,
            "contradictions": contradictions, "lost": lost, "b1": b1, "b2": b2}


# ── aggregation ───────────────────────────────────────────────────────────────

def rate(k, n):
    return 0.0 if not n else k / n


def summarise(rows):
    rows = list(rows)
    out = {"pages": len(rows), "gt": 0, "fp_spots": 0, "refused": 0,
           "n_fields": 0, "wrong": 0}
    for key in PATHS:
        out[key] = {"spoken": 0, "prec_k": 0, "recall_k": 0, "gt": 0}
    for r in rows:
        out["gt"] += r["n_gt"]
        out["fp_spots"] += r["n_fp_spots"]
        out["refused"] += r["refused"]
        out["n_fields"] += r["n_fields"]
        out["wrong"] += len(r["valid_wrong"])
        for key in PATHS:
            out[key]["spoken"] += len(r[key + "_spoken"])
            out[key]["prec_k"] += r[key + "_hit"]
            out[key]["recall_k"] += len(r[key + "_recovered"])
            out[key]["gt"] = out["gt"]
    return out


# ── the paired comparison, judged by compare_runs itself ─────────────────────

def _artifact(rows, key):
    """A hackathon-shaped artifact whose records carry one outcome per
    (page, true value) — the shape compare_runs scores. Identical expected lists
    on both sides is what LICENSES the paired exact McNemar test."""
    n = sum(len(r["gt_values"]) for r in rows)
    k = sum(len(r[key]) for r in rows)
    return {
        "classifier": {}, "missing": [],
        "pipeline": {
            "n": n, "field_recovery": (k / n) if n else 0.0,
            "records": [{"profile": r.get("profile"), "doc": r.get("doc"),
                         "category": r["doc_type"], "expected_category": r["doc_type"],
                         "expected_fields": r["gt_values"], "fields": list(r[key])}
                        for r in rows]},
    }


def paired(rows, old_key, new_key):
    """The compare_runs row for two paths scored on the same pages/items."""
    _, _, crows = CR.compare(_artifact(rows, old_key), _artifact(rows, new_key))
    return next(r for r in crows if r["metric"] == "field_recovery@pipeline")


# ── the harness: corpus -> recogniser -> the core above ──────────────────────

def page_groups(data, limit=0):
    """[{"doc","doc_type","profile","crops","true"}] in corpus reading order.

    A page is the unit because proximity needs EVERY line of it, and the line
    texts are the recogniser's crops' OWN labels — the rendered truth.
    """
    groups, seen = [], {}
    for crop, label, doc, prof in zip(data["crops"], data["labels"],
                                      data["doc"], data["profile"]):
        doc = str(doc)
        if doc not in seen:
            seen[doc] = len(groups)
            groups.append({"doc": doc, "doc_type": doc.rsplit("-", 1)[0],
                           "profile": str(prof), "crops": [], "true": []})
        groups[seen[doc]]["crops"].append(crop)
        groups[seen[doc]]["true"].append(str(label))
    return groups[:limit] if limit else groups


def _ci(ci):
    return "n/a" if not ci else "[%.3f,%.3f]" % ci


def _r(k, n):
    return "-" if not n else "%.3f" % (k / n)


def print_table(rows_by_th, meta, thresholds=THRESHOLDS):
    print("\n" + "=" * 108)
    print("%s   sha256 %s   %d pages, %d line crops"
          % (meta["path"], meta["sha256"], meta["pages"], meta["lines"]))
    print("=" * 108)
    profiles = sorted({r["profile"] for r in rows_by_th[thresholds[0]]})
    for th in thresholds:
        rows = rows_by_th[th]
        print("\n-- label_spot threshold %.3f --  (baseline = evaluate.fields_found "
              "over the whole page)" % th)
        print("%-14s %-14s %7s %6s %-15s %6s %6s %-15s %s"
              % ("profile", "path", "spoken", "prec", "prec 95% CI", "gt", "recall",
                 "recall 95% CI", "safety"))
        for prof in profiles + ["ALL"]:
            sub = rows if prof == "ALL" else [r for r in rows if r["profile"] == prof]
            s = summarise(sub)
            for key, name in (("baseline", "(a) regex"), ("spot", "(b) spot"),
                              ("valid", "(c) spot+val")):
                ps, ph = s[key]["spoken"], s[key]["prec_k"]
                gs, rh = s[key]["gt"], s[key]["recall_k"]
                safety = ""
                if key == "valid":
                    safety = ("refuse %d/%d %s   WRONG-but-spoken %d"
                              % (s["refused"], s["n_fields"],
                                 _r(s["refused"], s["n_fields"]), s["wrong"]))
                print("%-14s %-14s %7d %6s %-15s %6d %6s %-15s %s"
                      % (prof if key == "baseline" else "", name, ps, _r(ph, ps),
                         _ci(wilson(ph, ps)) if ps else "n/a", gs, _r(rh, gs),
                         _ci(wilson(rh, gs)) if gs else "n/a", safety))
        print("  pages %d   true label/value pairs %d   spot FALSE POSITIVES %d "
              "(label absent from the true line)" % (s["pages"], s["gt"], s["fp_spots"]))


def print_paired(rows_by_th, meta, thresholds=THRESHOLDS):
    print("\npaired, same pages and same items, exact McNemar (compare_runs "
          "verdicts) — %s" % meta["path"])
    print("%-7s %-30s %10s %10s %9s %11s %s"
          % ("thresh", "comparison", "old k/n", "new k/n", "delta", "pairs b/c",
             "p / verdict"))
    for th in thresholds:
        rows = rows_by_th[th]
        for oldk, newk, label in (("baseline_recovered", "valid_recovered",
                                   "(a) regex -> (c) spot+val"),
                                  ("baseline_recovered", "spot_recovered",
                                   "(a) regex -> (b) spot")):
            r = paired(rows, oldk, newk)
            b, c = r["discordant"] or ("-", "-")
            p_str = "n/a" if r["p"] is None else "%.4g" % r["p"]
            print("%-7.3f %-30s %10s %10s %+9.3f %11s p=%-8s %s"
                  % (th, label, "%s/%s" % (r["k_old"], r["n"]),
                     "%s/%s" % (r["k_new"], r["n"]), r["delta"] or 0.0,
                     "%s/%s" % (b, c), p_str, r["verdict"]))
    print("  b = old(right) & new(wrong), c = old(wrong) & new(right); a result "
          "whose 95% CIs overlap is INCONCLUSIVE.")


# ── the policy: same pages, same ground truth, same verdict vocabulary ───────

def summarise_policy(rows, name):
    """Aggregate one policy variant over a set of page rows."""
    out = {"pages": len(rows), "gt": 0, "spoken": 0, "hit": 0, "rec": 0,
           "refused": 0, "n": 0, "wrong": 0, "printed": 0,
           "by_kind": {}, "spoken_kind": {}}
    for r in rows:
        p = r["policy"][name]
        out["gt"] += r["n_gt"]
        out["spoken"] += p["n_spoken"]
        out["hit"] += p["n_hit"]
        out["rec"] += len(p["recovered"])
        out["refused"] += p["refused"]
        out["n"] += p["n"]
        out["wrong"] += len(p["wrong"])
        out["printed"] += len(p["wrong_printed"])
        for k, v in p["by_kind"].items():
            out["by_kind"][k] = out["by_kind"].get(k, 0) + v
        for k, v in p["spoken_kind"].items():
            out["spoken_kind"][k] = out["spoken_kind"].get(k, 0) + v
    return out


def print_policy(rows_by_th, meta, thresholds=THRESHOLDS):
    """The decision layer's numbers, next to the three paths already printed."""
    print("\n" + "=" * 176)
    print("POLICY (field_policy.decide) — what is SPOKEN, scored against the same "
          "rendered ground truth   %s   sha256 %s"
          % (meta["path"], meta["sha256"]))
    print("=" * 176)
    print("  critical kinds = dosage money account sort_code date.  WRONG-but-spoken "
          "= a spoken value that conveys no ground-truth value; 'printed' = of those, "
          "the ones printed on the page (no gt item: the anchoring label was never spotted).")
    for name in POLICY_VARIANTS:
        print("  %-13s %s" % (name, POLICY_NOTES[name]))
    for th in thresholds:
        rows = rows_by_th[th]
        profiles = sorted({r["profile"] for r in rows})
        print("\n-- label_spot threshold %.3f --" % th)
        print("%-13s %-15s %6s %6s %-16s %5s %6s %-16s %-14s %5s %6s %-16s %7s"
              % ("variant", "profile", "spoken", "prec", "prec 95% CI", "gt",
                 "recall", "recall 95% CI", "refuse (rate)", "WRONG", "wrong",
                 "wrong 95% CI", "printed"))
        for name in POLICY_VARIANTS:
            for prof in profiles + ["ALL"]:
                sub = rows if prof == "ALL" else [r for r in rows if r["profile"] == prof]
                s = summarise_policy(sub, name)
                ps, ph, gs, rh = s["spoken"], s["hit"], s["gt"], s["rec"]
                w = s["wrong"]
                print("%-13s %-15s %6d %6s %-16s %5d %6s %-16s %-14s %5d %6s %-16s %7d"
                      % (name if prof == profiles[0] or prof == "ALL" else "",
                         prof, ps, _r(ph, ps), _ci(wilson(ph, ps)) if ps else "n/a",
                         gs, _r(rh, gs), _ci(wilson(rh, gs)) if gs else "n/a",
                         "%d/%d %s" % (s["refused"], s["n"], _r(s["refused"], s["n"])),
                         w, _r(w, ps), _ci(wilson(w, ps)) if ps else "n/a", s["printed"]))
        s = summarise_policy(rows, "policy")
        kinds = ", ".join("%s %d/%d" % (k, s["by_kind"].get(k, 0), s["spoken_kind"][k])
                          for k in sorted(s["spoken_kind"]))
        print("  policy: wrong-but-spoken by kind — %s" % kinds)
        for name in list(POLICY_VARIANTS)[1:]:
            v = summarise_policy(rows, name)
            print("  %-13s refusal cost vs 'policy': spoken %d -> %d, recall %s -> %s"
                  % (name, s["spoken"], v["spoken"], _r(s["rec"], s["gt"]),
                     _r(v["rec"], v["gt"])))

    print("\npaired against the same items, exact McNemar (compare_runs verdicts) — "
          "recall only; the policy's trade is recall FOR safety, so a REGRESSED here "
          "is the price, not the verdict on the product")
    print("%-7s %-34s %10s %10s %9s %11s %s"
          % ("thresh", "comparison", "old k/n", "new k/n", "delta", "pairs b/c",
             "p / verdict"))
    for th in thresholds:
        rows = rows_by_th[th]
        for newk, label in (("policy_recovered", "policy"),
                            ("typed_only_recovered", "typed-only"),
                            ("typed_strict_recovered", "typed+strict")):
            for oldk, oldname in (("baseline_recovered", "(a) regex"),
                                  ("valid_recovered", "(c) spot+val")):
                r = paired(rows, oldk, newk)
                b, c = r["discordant"] or ("-", "-")
                p_str = "n/a" if r["p"] is None else "%.4g" % r["p"]
                print("%-7.3f %-34s %10s %10s %+9.3f %11s p=%-8s %s"
                      % (th, "%s -> %s" % (oldname, label),
                         "%s/%s" % (r["k_old"], r["n"]), "%s/%s" % (r["k_new"], r["n"]),
                         r["delta"] or 0.0, "%s/%s" % (b, c), p_str, r["verdict"]))
    print("  Scored on RECALL (of true values recovered). Wrong-but-spoken is a "
          "quality-of-speech number, not an item-recovery one, so it has no paired "
          "test: it is quoted with its Wilson interval and its counts.")


def _bucket_line(d, others, doc, prof):
    other = ", ".join("'%s'" % o[:26] for o in others if o) or "-"
    return ("    %-18s %-14s %-14s '%s'   other path: %s\n        why: %s"
            % (doc, prof, d["field"], (d["value"] or "")[:26], other, d["why"][:96]))


def print_buckets(rows_by_th, threshold=PRIMARY, limit=4):
    """The two disagreement buckets, with the strings — the cases that decide
    whether AGREEMENT is the right rule at all."""
    rows = rows_by_th[threshold]
    cont, b1, b2, lost = [], [], [], []
    for r in rows:
        p = r["policy"]["policy"]
        cont += [(d, o, r["doc"], r["profile"]) for d, o in p["contradictions"]]
        b1 += [(d, o, r["doc"], r["profile"]) for d, o in p["b1"]]
        b2 += [(d, o, r["doc"], r["profile"]) for d, o in p["b2"]]
        lost += [(d, r["doc"], r["profile"]) for d in p["lost"]]
    print("\n" + "-" * 120)
    print("DISAGREEMENTS the policy refused (label_spot %.3f) — %d contradictions fired"
          % (threshold, len(cont)))
    for d, o, doc, prof in cont[:limit]:
        print(_bucket_line(d, o, doc, prof))
    print("  bucket 1 — REFUSED, spot was RIGHT and the regex path was wrong or silent: %d"
          % len(b1))
    for d, o, doc, prof in b1[:limit]:
        print(_bucket_line(d, o, doc, prof))
    print("  bucket 2 — REFUSED, regex was RIGHT and the spot path was wrong or silent: %d"
          % len(b2))
    for d, o, doc, prof in b2[:limit]:
        print(_bucket_line(d, o, doc, prof))
    print("  (for scale: %d refused values WERE right — %.1f%% of all refusals; they "
          "are the recall this policy gives up)" % (len(lost), 100.0 * len(lost) /
                                                    max(1, sum(r["policy"]["policy"]["refused"]
                                                               for r in rows))))
    print("-" * 120)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="field truth: score field recovery against the RENDERED line "
                    "continuation after each spotted label")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-pages", type=int, default=120,
                    help="cap pages (0 = the whole split)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="retrained recogniser (default %s)" % DEFAULT_CKPT)
    ap.add_argument("--with-policy", action="store_true",
                    help="also score field_policy.decide (speak/refuse) on the same "
                         "pages and ground truth: recall, precision, refusals, "
                         "wrong-but-spoken, paired McNemar, disagreement buckets")
    a = ap.parse_args(argv)

    data = RC.load(split=a.split)
    if not data["crops"].shape[0]:
        print("no crops in split=%s: build the corpus first "
              "(bash run.sh py ml/src/render_corpus.py --pages 500)" % a.split)
        return 1
    groups_all = page_groups(data)
    groups = page_groups(data, a.limit_pages)

    print("SightLine — FIELD TRUTH (baseline regex vs label-anchored extraction, "
          "ground truth = the rendered line)")
    print("split=%-5s pages=%d of %d   line crops=%d   thresholds %s   "
          "evaluate.FIELDS %s"
          % (a.split, len(groups), len(groups_all),
             sum(len(g["crops"]) for g in groups), list(THRESHOLDS), E.FIELDS))
    print("(baseline = evaluate.fields_found; extractor = label_spot + "
          "field_proximity.extract_fields)")

    import torch
    torch.set_num_threads(1)              # the box is loaded; extra threads crawl

    ckpts = []
    for p in (a.ckpt, PAIRED_CKPT):
        if p not in ckpts:
            ckpts.append(p)

    for ckpt in ckpts:
        if not Path(ckpt).exists():
            print("\nSKIPPED %s — no such checkpoint (named, not defaulted)" % ckpt)
            continue
        model = None
        for attempt in range(3):          # a mid-run save can be a torn read
            try:
                model, device = ER.load_model(ckpt)
                break
            except RuntimeError as e:
                if attempt == 2:
                    print("\nSKIPPED %s — %s" % (ckpt, e))
                    break
                import time
                time.sleep(2.0)
        if model is None:
            continue

        rows_by_th = {th: [] for th in THRESHOLDS}
        n_lines = 0
        for g in groups:
            lines = FR.read_page_lines(g["crops"], model, device)
            n_lines += len(lines)
            for th in THRESHOLDS:
                row = score_page(g["true"], lines, g["doc_type"], th,
                                 policy_variants=POLICY_VARIANTS if a.with_policy else None)
                row["doc"], row["profile"] = g["doc"], g["profile"]
                rows_by_th[th].append(row)
        meta = {"path": ckpt, "sha256": FR.sha256_prefix(ckpt),
                "pages": len(groups), "lines": n_lines}
        print_table(rows_by_th, meta)
        print_paired(rows_by_th, meta)
        if a.with_policy:
            print_policy(rows_by_th, meta)
            print_buckets(rows_by_th)
    return 0


if __name__ == "__main__":
    sys.exit(main())
