"""
eval_field_recovery.py — THE DECIDING MEASUREMENT for the label-spotting
strategy the product is now built on.

The reader no longer transcribes a page. It spots ONE KNOWN LABEL
(label_spot.spot_in_lines: 37 labels, digit-confusion tolerant) and extracts +
validates the value beside it (field_proximity.extract_fields: 7 kinds, every
value carries a spoken refusal reason). Each half has been measured alone
(label_spot: P 0.87 / R 0.41 on real OCR). Neither has ever been measured END
TO END. This file produces that number, per capture profile, on the HELD-OUT
render test split, so the team can decide to ship the strategy or kill it.

WHAT IS MEASURED (per profile and overall)
  a) fields_expected  sum over pages of len(evaluate.FIELDS[doc_type])
  b) fields_recovered extracted with ok=True whose value matches a ground-truth
                      field through evaluate.norm() — a masked value counts when
                      it is a suffix of the expected one, len >= 4
                      (bench_hackathon._recovered: the repo's one definition)
  c) precision_of_spoken  of the values the pipeline would SPEAK (ok=True), the
                      fraction that are correct. This is the product-safety
                      number: everything ok=False is refused, not spoken.
  d) refusal          values rejected by validation, and pages where the
                      spotter found no label at all
  e) the CURRENT product behaviour, paired on the same pages: keyword/regex
                      extraction over the whole page text — evaluate.fields_found
                      is the Python twin of app/js/pipeline.js. Spotting vs
                      baseline, with the discordant counts and an EXACT McNemar
                      p, judged in compare_runs.compare's own vocabulary
                      (IMPROVED / REGRESSED / INCONCLUSIVE).

HONESTY RULES (the repo's, not invented here)
  * Wilson 95% intervals on every rate, with the counts (k/n) it rests on;
  * a gain the sample cannot support is INCONCLUSIVE, however nice the delta;
  * the checkpoint is NAMED and its sha256 prefix printed, so the number is
    attributable; a torn read is REFUSED (eval_render_corpus.load_model);
  * a page with no crops/expected fields is scored, never skipped.

WHAT IT DOES NOT CLAIM
  The line crops come from the renderer's own layout, so this isolates the
  spotting/extraction path from the line DETECTOR exactly as
  eval_render_corpus.py isolates the recogniser. End-to-end with detection is a
  separate measurement.

    bash run.sh py ml/src/eval_field_recovery.py --split test --limit-pages 150
    bash run.sh py ml/src/eval_field_recovery.py --ckpt models/ocr/crnn.pt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import build_dataset as B            # noqa: E402
import eval_ocr as EO                # noqa: E402
import eval_render_corpus as ER      # noqa: E402
import evaluate as E                 # noqa: E402
import field_proximity as FP         # noqa: E402
import label_spot as LS              # noqa: E402
import render_corpus as RC           # noqa: E402
import bench_hackathon as BH         # noqa: E402  (_recovered: one definition of
#   "a masked value still counts as recovered")
import compare_runs as CR            # noqa: E402  (wilson + the verdict rule)

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CKPT = "models/ocr/crnn_before_render.pt"
RETRAINED_CKPT = "models/ocr/crnn.pt"

# The trade the product must make: 0.75 is the precision-first default in
# label_spot (measured P 0.87 / R 0.41); 0.833 is the strict 1-edit-per-6
# setting (R 0.32). Both are reported because the choice is the product's.
THRESHOLDS = (0.75, 0.833)
PRIMARY = 0.75


# ── the spotting/extraction core (no model, no corpus: unit-testable) ─────────

def spot_lines(lines, threshold=PRIMARY):
    """Spots for a page, each tagged with its ``line_index``.

    label_spot.spot_in_lines() is FROZEN without a threshold (its own measure()
    documents why), so a threshold sweep is what label_spot.measure does:
    spot_labels() per line, line_index re-attached. At 0.75 this is identical to
    spot_in_lines(lines) — pinned by tests/test_field_recovery.py.
    """
    out = []
    for i, line in enumerate(lines or []):
        text = (line or {}).get("text", "")
        for spot in LS.spot_labels(text, threshold):
            out.append(dict(spot, line_index=i))
    return out


def baseline_fields(text, doc_type):
    """The CURRENT product behaviour: ground-truth tokens present anywhere in the
    page text. evaluate.fields_found IS the Python twin of pipeline.js."""
    return set(E.fields_found(text or "", doc_type))


def score_page(lines, doc_type, threshold=PRIMARY):
    """One page. ``lines`` are already-recognised {"text", "box"} dicts.

    Returns the counts the headline rests on, plus the page text so the baseline
    is measured on exactly the same characters the spotter saw.
    """
    lines = list(lines or [])
    fields = FP.extract_fields(lines, spot_lines(lines, threshold))
    expected = list(E.FIELDS.get(doc_type, []))
    spoken = [r["value"] for r in fields.values() if r["ok"]]
    text = " ".join(str(ln.get("text", "")) for ln in lines)
    return {
        "doc_type": doc_type,
        "expected": expected,
        # RECOVERED = the ground-truth field was SPOKEN, by either rule. The
        # strict rule (_recovered) handles a value masked to its last four
        # digits; the containment rule handles a ground-truth token that the
        # pipeline correctly nested inside a longer spoken value (GT '500mg'
        # spoken as 'AMOXICILLIN 500MG'). Without containment the baseline wins
        # by construction: it is a substring search over the whole page text
        # (evaluate.fields_found), so it can look for '500mg' anywhere, while a
        # field extractor that anchors the dosage to its label and speaks the
        # dose line scored 0 for recovering the very token it had read correctly.
        "recovered": [e for e in expected
                      if any(BH._recovered(e, v) or E.norm(e) in E.norm(v)
                             for v in spoken)],
        "baseline": [e for e in expected if e in baseline_fields(text, doc_type)],
        "spoken": len(spoken),
        "spoken_ok": sum(1 for v in spoken
                         if any(BH._recovered(e, v) for e in expected)),
        "refused": sum(1 for r in fields.values() if not r["ok"]),
        "refusal_reasons": Counter(r["reason"] for r in fields.values()
                                   if not r["ok"]),
        "n_spots": len(spot_lines(lines, threshold)),
        "text": text,
    }


def aggregate(rows):
    """Sum the per-page counts. A page with no expected fields still contributes
    its (zero) denominator, and a page nothing was spotted on is COUNTED."""
    rows = list(rows)
    return {
        "pages": len(rows),
        "expected": sum(len(r["expected"]) for r in rows),
        "recovered": sum(len(r["recovered"]) for r in rows),
        "baseline": sum(len(r["baseline"]) for r in rows),
        "spoken": sum(r["spoken"] for r in rows),
        "spoken_ok": sum(r["spoken_ok"] for r in rows),
        "refused": sum(r["refused"] for r in rows),
        "fields": sum(r["spoken"] + r["refused"] for r in rows),
        "pages_no_spots": sum(1 for r in rows if not r["n_spots"]),
        "reasons": sum((r["refusal_reasons"] for r in rows), Counter()),
    }


def precision_of_spoken(rows):
    """(k, n) — correct spoken values over all spoken values. (0, 0) when the
    pipeline refused everything: no spoken value, no denominator, no claim."""
    a = aggregate(rows)
    return a["spoken_ok"], a["spoken"]


# ── the paired comparison, judged by compare_runs itself ─────────────────────

def _artifact(rows, key):
    """A hackathon-shaped artifact whose pipeline records carry one outcome per
    (page, expected field) — the shape compare_runs scores. `n` and
    `field_recovery` are the same counts the records imply, because
    compare_runs takes the metric VALUE from the header and the COUNT from the
    records; they must agree or the delta would describe a different run."""
    n = sum(len(r["expected"]) for r in rows)
    k = sum(len(r[key]) for r in rows)
    return {
        "classifier": {},
        "missing": [],
        "pipeline": {
            "n": n,
            "field_recovery": (k / n) if n else 0.0,
            "records": [
                {"profile": r["profile"], "doc": r["doc"],
                 "category": r["doc_type"], "expected_category": r["doc_type"],
                 "expected_fields": r["expected"], "fields": list(r[key])}
                for r in rows]},
    }


def paired_vs_baseline(rows, base_key="baseline", new_key="recovered"):
    """Spotting vs the regex baseline on the SAME pages.

    Run through compare_runs.compare so the pairing rule, the discordant count
    and the IMPROVED / REGRESSED / INCONCLUSIVE verdict are literally the ones
    the repo ships (exact McNemar when both runs scored the same items).
    `b` = baseline right & spotting wrong, `c` = spotting right & baseline wrong.
    """
    _, _, crows = CR.compare(_artifact(rows, base_key), _artifact(rows, new_key))
    row = next(r for r in crows if r["metric"] == "field_recovery@pipeline")
    b, c = row["discordant"] or (None, None)
    return {"verdict": row["verdict"], "method": row["method"], "b": b, "c": c,
            "p": row["p"], "n": row["n"], "k_new": row["k_new"],
            "k_old": row["k_old"], "delta": row["delta"],
            "new_ci": row["new_ci"], "old_ci": row["old_ci"]}


# ── the harness: corpus -> recogniser -> the core above ──────────────────────

def page_groups(data, limit_pages=0):
    """[(doc, doc_type, profile, [crops])] in corpus reading order.

    A page is the unit because proximity needs EVERY line of it: a label and its
    value can sit on different lines, so a crop-only sample cannot measure this.
    """
    groups, seen = [], {}
    for crop, doc, prof in zip(data["crops"], data["doc"], data["profile"]):
        doc = str(doc)
        if doc not in seen:
            seen[doc] = len(groups)
            groups.append((doc, doc.rsplit("-", 1)[0], str(prof), []))
        groups[seen[doc]][3].append(crop)
    return groups[:limit_pages] if limit_pages else groups


def read_page_lines(crops, model, device):
    """Recognise every crop of a page, in reading order, into line dicts.

    The box is the crop's own extent placed at the page origin (0, 0). It is a
    placeholder: field_proximity's cross-line fallback needs real page
    coordinates to pick the line BELOW a label, and with this box it finds none
    and returns None. That direction is the safe one — it can never invent a
    value off the wrong line — and it means this measurement scores the
    same-line path, which is where every ground-truth field on these documents
    is printed. ponytail: pass the renderer's line bands instead if cross-line
    fields ever matter.
    """
    lines = []
    for crop in crops:
        arr = B.crop_to_array(Image.fromarray(np.asarray(crop, np.uint8)))
        text, _conf = EO.recognise(model, arr, device)
        h, w = arr.shape
        lines.append({"text": str(text), "box": (0, 0, int(w), int(h))})
    return lines


def sha256_prefix(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


def _ci(ci):
    return "n/a" if not ci else "[%.3f,%.3f]" % ci


def _rate(k, n):
    return "-" if not n else "%.3f" % (k / n)


def run_checkpoint(path, groups, thresholds=THRESHOLDS):
    """Recognise every page once, score it at each threshold. -> (rows_by_th, meta)"""
    import torch
    torch.set_num_threads(1)              # the box is loaded; 4 threads crawl

    model, device = ER.load_model(path)
    rows_by_th = {th: [] for th in thresholds}
    n_lines = 0
    for doc, doc_type, profile, crops in groups:
        lines = read_page_lines(crops, model, device)
        n_lines += len(lines)
        for th in thresholds:
            row = score_page(lines, doc_type, th)
            row["doc"], row["profile"] = doc, profile
            rows_by_th[th].append(row)
    meta = {"path": str(path), "pages": len(groups), "lines": n_lines,
            "sha256": sha256_prefix(path)}
    return rows_by_th, meta


def print_table(rows_by_th, meta, thresholds=THRESHOLDS, title="checkpoint"):
    print("\n" + "=" * 132)
    print("%s: %s   sha256 %s   %d pages, %d line crops"
          % (title, meta["path"], meta["sha256"], meta["pages"], meta["lines"]))
    print("=" * 132)
    profiles = sorted({r["profile"] for r in rows_by_th[thresholds[0]]})
    for th in thresholds:
        rows = rows_by_th[th]
        a = aggregate(rows)
        print("\n-- label_spot threshold %.3f --" % th)
        print("%-16s %5s %5s %5s %8s %-16s %18s %-16s %8s %8s %9s"
              % ("profile", "pages", "exp", "rec", "rec", "rec 95% CI",
                 "prec-of-spoken", "prec 95% CI", "refused", "ref rate", "no-spot"))
        for prof in profiles + ["ALL"]:
            sub = rows if prof == "ALL" else [r for r in rows if r["profile"] == prof]
            s = aggregate(sub)
            k, n = s["spoken_ok"], s["spoken"]
            print("%-16s %5d %5d %5d %8s %-16s %18s %-16s %8d %8s %9d"
                  % (prof, s["pages"], s["expected"], s["recovered"],
                     _rate(s["recovered"], s["expected"]),
                     _ci(CR.wilson(s["recovered"], s["expected"])),
                     "%d/%d %s" % (k, n, _rate(k, n)),
                     _ci(CR.wilson(k, n)) if n else "n/a",
                     s["refused"], _rate(s["refused"], s["fields"]),
                     s["pages_no_spots"]))
        print("  baseline (whole-page keyword regex, evaluate.fields_found): "
              "%d/%d %s %s"
              % (a["baseline"], a["expected"], _rate(a["baseline"], a["expected"]),
                 _ci(CR.wilson(a["baseline"], a["expected"]))))
        top = ", ".join("%s x%d" % (why, c) for why, c in a["reasons"].most_common(4))
        print("  refusals %d/%d extracted fields%s"
              % (a["refused"], a["fields"], ("  (" + top + ")") if top else ""))


def print_paired(rows_by_th, meta, thresholds=THRESHOLDS):
    print("\npaired spotting vs regex baseline (same pages, exact McNemar) — %s"
          % meta["path"])
    print("%-8s %-6s %8s %8s %9s %16s %20s"
          % ("thresh", "n", "base k/n", "spot k/n", "delta", "b / c", "p / verdict"))
    for th in thresholds:
        rows = rows_by_th[th]
        c = paired_vs_baseline(rows)
        print("%-8.3f %-6s %8s %8s %9s %16s %20s"
              % (th, c["n"], "%s/%s" % (c["k_old"], c["n"]),
                 "%s/%s" % (c["k_new"], c["n"]),
                 "%+.3f" % (c["delta"] or 0.0),
                 "%s/%s" % (c["b"], c["c"]),
                 "p=%.4g %s" % (c["p"], c["verdict"])))
    print("  b = baseline right & spotting wrong, c = spotting right & baseline "
          "wrong; verdict vocabulary is compare_runs.compare's.")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="field recovery end to end: spot a label, extract beside "
                    "it, validate, refuse or speak")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-pages", type=int, default=150,
                    help="0 = every page of the split")
    ap.add_argument("--ckpt", default="",
                    help="path to a recogniser checkpoint (default %s)" % DEFAULT_CKPT)
    a = ap.parse_args(argv)

    data = RC.load(split=a.split)
    if not data["crops"].shape[0]:
        print("no crops in split=%s: build the corpus first "
              "(bash run.sh py ml/src/render_corpus.py --pages 500)" % a.split)
        return 1
    groups_all = page_groups(data)
    groups = page_groups(data, a.limit_pages)

    print("SightLine — FIELD RECOVERY end to end (spot a known label -> extract "
          "beside it -> validate -> refuse or speak)")
    print("split=%-5s pages=%d of %d   line crops=%d"
          % (a.split, len(groups), len(groups_all),
             sum(len(g[3]) for g in groups)))
    print("thresholds %s   primary %.3f   fields per doc type %s"
          % (list(THRESHOLDS), PRIMARY, E.FIELDS))

    # Which checkpoints to run: the frozen default, plus the retrain if it is
    # newer (label spotting can only spot what the recogniser produced).
    ckpts = [a.ckpt] if a.ckpt else [DEFAULT_CKPT]
    if not a.ckpt:
        d, r = Path(DEFAULT_CKPT), Path(RETRAINED_CKPT)
        if r.exists() and d.exists() and r.stat().st_mtime > d.stat().st_mtime:
            ckpts.append(RETRAINED_CKPT)

    results = {}
    for ckpt in ckpts:
        if not Path(ckpt).exists():
            print("\nSKIPPED %s — no such checkpoint (named, not defaulted)" % ckpt)
            continue
        try:
            rows_by_th, meta = run_checkpoint(ckpt, groups)
        except RuntimeError as e:            # the torn-read guard
            print("\nSKIPPED %s — %s" % (ckpt, e))
            continue
        results[ckpt] = rows_by_th
        print_table(rows_by_th, meta, title="checkpoint")

    for ckpt, rows_by_th in results.items():
        print_paired(rows_by_th, {"path": ckpt})

    # The retrain's effect on the PRODUCT metric, paired on the same pages.
    if len(results) == 2:
        (p1, r1), (p2, r2) = list(results.items())
        print("\nretrain effect (%s vs %s) on fields recovered, same pages:"
              % (Path(p1).name, Path(p2).name))
        for th in THRESHOLDS:
            old = _artifact(r1[th], "recovered")
            new = _artifact(r2[th], "recovered")
            _, _, crows = CR.compare(old, new)
            row = next(x for x in crows if x["metric"] == "field_recovery@pipeline")
            print("  threshold %.3f  %s/%s -> %s/%s  delta %+.3f  pairs b/c %s/%s  "
                  "p=%.4g  %s"
                  % (th, row["k_old"], row["n"], row["k_new"], row["n"],
                     row["delta"], (row["discordant"] or ("-", "-"))[0],
                     (row["discordant"] or ("-", "-"))[1], row["p"], row["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
