"""
second_reader.py — the INDEPENDENT SECOND READ of the same crop.

WHY THIS EXISTS
---------------
Lane N proved a ceiling. On 120 held-out pages the speak/refuse policy
(field_policy.decide) cut wrong-but-spoken from 276/704 (39%) to 14/323 (4.3%),
precision 0.957, and every remaining wrong value is a DATE one digit off
(DOB 04021979 read 04021879, 01102024, 01032026, 01102046). On all of those
pages the TRUE date appears NOWHERE in the recogniser's output, and both
extraction paths read the SAME OCR pass — so no decision rule over one pass can
catch it. Agreement between the regex path and the spot path detects PARSE
disagreement, never RECOGNITION error.

The only route left is a second read that is genuinely INDEPENDENT — not a
re-decode of the same logits (that shares the error by construction). This
module runs the SAME weights over a DIFFERENTLY PREPROCESSED (or rescaled)
version of the same crop and asks whether the two reads AGREE. If the one-digit
errors are preprocessing-dependent, the two readers disagree on exactly those
values and the ensemble refuses them; if the errors are correlated, the second
read is phone compute bought for nothing and this module says so.

READERS
-------
  r1_baseline  the CURRENT read, unchanged: build_dataset.crop_to_array then
               eval_ocr.recognise (via eval_field_recovery.read_page_lines).
  r2_clahe     tile-CLAHE (clipLimit 2.0, 8x8) on the crop, then crop_to_array
               then recognise — the app's `enhanced` candidate
               (app/js/pipeline.js; Python twin: evaluate.arm_restorer_clahe).
  r2b_otsu     CLAHE + GaussianBlur + Otsu (the app's `binary` candidate), the
               other preprocessing variant `candidates = [raw, enhanced, binary]`.
  r3_h48       crop_to_array at input height 48 instead of 32 — a geometric
               rescale, deterministic, no new content.

The ensemble speaks a safety-critical value ONLY when r1 and r2 produce the same
value after normalisation (evaluate.norm). These readers DO share the pixels, so
the question this measures is not philosophical: do the errors move when the
preprocessing moves?

MEASURED (safety-critical kinds: date, dosage, money, account, sort_code)
  * per reader   — precision, recall, wrong-but-spoken, Wilson 95% CIs + counts
  * DECORRELATION— of the values r1 got WRONG, how many did r2 get RIGHT (RESCUED)?
                   Of the values r1 got right, how many did r2 get wrong?
  * ensemble     — agreement rule precision / recall / wrong-but-spoken
  * irreducible core — values BOTH readers got wrong AND on the same string
                   (those are unfixable at this model; the honest floor)
  * paired exact McNemar (compare_runs verdicts) of ensemble recall against the
    typed-only policy on the SAME pages and items
  * cost — extra forward passes per page the second read adds

WIRING reuses, never re-invents: render_corpus.load, eval_render_corpus.load_model
(torn-read guard), eval_field_recovery.read_page_lines / sha256_prefix, eval_ocr,
label_spot, field_proximity, field_policy (the one validator + kind map),
evaluate (norm, fields_found, arm_restorer_clahe's CLAHE settings),
bench_hackathon._recovered + eval_field_truth.match_value (the one match rule),
compare_runs (wilson + the verdict rule).

    bash run.sh py ml/src/second_reader.py --split test --limit-pages 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np                      # noqa: E402
from PIL import Image                   # noqa: E402

import build_dataset as B               # noqa: E402  (crop_to_array: THE resize)
import compare_runs as CR               # noqa: E402  (wilson + the verdict rule)
import constrained_decode as CD         # noqa: E402  (imported for the R4 note)
import eval_field_recovery as FR        # noqa: E402  (read_page_lines, sha256 on R1)
import eval_field_truth as ET           # noqa: E402  (page_groups, ground_truth, paired)
import eval_ocr as EO                   # noqa: E402  (the ONE recognise)
import eval_render_corpus as ER         # noqa: E402  (load_model: torn-read guard)
import evaluate as E                    # noqa: E402  (norm, fields_found, CLAHE twin)
import field_policy as FPOL             # noqa: E402  (kind_of, CRITICAL_KINDS, decide)
import field_proximity as FP            # noqa: E402
import realdata as R                    # noqa: E402  (CROP_H/MIN_W/MAX_W)
import render_corpus as RC              # noqa: E402

DEFAULT_CKPT = "models/ocr/crnn.pt"
THRESHOLD = ET.PRIMARY                  # label_spot operating point (0.75)

CRITICAL = FPOL.CRITICAL_KINDS           # date dosage money account sort_code

# The second reader the ensemble agrees with r1 on. The others are measured too
# but r2 is the one the DECORRELATION table and the ensemble are built on.
PRIMARY_R2 = "r2_clahe"
DEFAULT_EXTRAS = ("r2_clahe", "r3_h48")


# ── the readers: crop uint8 [h, w] -> the [H, W] array the model eats ─────────

def _clahe(crop):
    """Tile-CLAHE, the app's `enhanced` candidate (evaluate.arm_restorer_clahe
    settings). Pure OpenCV, deterministic."""
    import cv2
    g = np.asarray(crop, np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def _clahe_otsu(crop):
    """CLAHE + blur + Otsu, the app's `binary` candidate. Deterministic."""
    import cv2
    g = cv2.GaussianBlur(_clahe(crop), (3, 3), 0)
    _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return g


def _crop_to_array_h(crop, h):
    """build_dataset.crop_to_array at a DIFFERENT input height. Same resize
    rule (aspect preserved, width clamped to MIN_W..MAX_W), only h changes."""
    g = Image.fromarray(np.asarray(crop, np.uint8)).convert("L")
    w, hh = g.size
    if w < 1 or hh < 1:
        return np.zeros((h, R.MAX_W), np.uint8)
    scale = h / float(hh)
    nw = max(R.MIN_W, min(R.MAX_W, int(round(w * scale))))
    return np.asarray(g.resize((nw, h), Image.BILINEAR), np.uint8)


def _arr_baseline(crop):
    return B.crop_to_array(Image.fromarray(np.asarray(crop, np.uint8)))


def _arr_clahe(crop):
    return B.crop_to_array(Image.fromarray(_clahe(crop)))


def _arr_otsu(crop):
    return B.crop_to_array(Image.fromarray(_clahe_otsu(crop)))


def _arr_h48(crop):
    return _crop_to_array_h(crop, 48)


# name -> the array transform. r1 is handled separately so it stays EXACTLY the
# shipped read (eval_field_recovery.read_page_lines) and cannot drift.
READERS = {
    "r2_clahe": _arr_clahe,
    "r2b_otsu": _arr_otsu,
    "r3_h48": _arr_h48,
}
BASELINE = "r1_baseline"


def reader_array(name, crop):
    """The model input array for one crop under one reader. Deterministic."""
    if name == BASELINE:
        return _arr_baseline(crop)
    return READERS[name](crop)


def read_page(crops, model, device, name=BASELINE):
    """One page -> [{"text","box"}] in reading order, under reader `name`.

    r1_baseline delegates to eval_field_recovery.read_page_lines so the baseline
    read is byte-for-byte the current one, not a reimplementation.
    """
    if name == BASELINE:
        return FR.read_page_lines(crops, model, device)
    fn = READERS[name]
    lines = []
    for crop in crops:
        arr = fn(np.asarray(crop, np.uint8))
        text, _conf = EO.recognise(model, arr, device)
        h, w = arr.shape
        lines.append({"text": str(text), "box": (0, 0, int(w), int(h))})
    return lines


# ── the pure logic: what one reader SPEAKS, and how two readers agree ─────────

def canon(s):
    """The one normalisation (evaluate.norm), so equality means what the scorer
    means by equality."""
    return E.norm(s)


def critical_spoken(lines, doc_type, threshold=THRESHOLD, speak_free_text=False):
    """{field: value} of the safety-critical values the typed-only policy speaks
    for one reader's lines. The policy is field_policy.decide — the one decision
    layer — restricted to CRITICAL kinds and to what it actually speaks.

    Returns (critical_values, decisions, fields) so a caller can re-score the
    whole decision set if it wants to.
    """
    spots = FR.spot_lines(lines, threshold)
    fields = FP.extract_fields(lines, spots)
    text = " ".join(str((ln or {}).get("text", "")) for ln in lines)
    regex = FPOL.from_regex_tokens(doc_type, E.fields_found(text, doc_type))
    decisions = FPOL.decide(regex, fields, doc_type=doc_type,
                            speak_free_text=speak_free_text)
    out = {}
    for d in decisions:
        if d["decision"] != "speak" or not d["value"]:
            continue
        if FPOL.kind_of(d["field"], fields.get(d["field"])) in CRITICAL:
            out.setdefault(d["field"], d["value"])
    return out, decisions, fields


def ensemble(r1, r2):
    """The agreement rule. r1/r2 are {field: value} critical values from each
    reader. Speak a field ONLY when both produced it and their normalised forms
    are equal; otherwise refuse, naming both strings.

    -> (spoken {field: value}, refusals [{field, value, other, reason}])
    """
    spoke, refused = {}, []
    for f in sorted(r1):
        v1 = r1[f]
        v2 = r2.get(f)
        if v2 is None:
            refused.append({"field": f, "value": v1, "other": None,
                            "reason": "second reader produced no value for %s" % v1})
        elif canon(v1) and canon(v1) == canon(v2):
            spoke[f] = v1
        else:
            refused.append({"field": f, "value": v1, "other": v2,
                            "reason": "readers disagree: %s vs %s" % (v1, v2)})
    return spoke, refused


def accord(r1, r2, gt_values, true_text=""):
    """The DECORRELATION accounting for two readers' critical values.

    Only GENUINE errors count: a value that is printed on the page but anchors no
    ground-truth item (the 400MG on an IBUPROFEN 400MG line) is not a recognition
    error and is reported separately, never folded into the rescue accounting —
    there is no true value for a second reader to have recovered.

    A value wrong in r1 and right in r2 is a RESCUE of the pair, never an error
    of it. A value both readers got wrong AND on the same string is the
    IRREDUCIBLE core — unfixable by agreement at this model. A value wrong in r1
    that r2 never produced is a refusal the ensemble would make anyway.
    """
    def right(v):
        return any(ET.match_value(v, g) for g in gt_values)

    r1_wrong = [f for f, v in r1.items() if not right(v)]
    printed = [f for f in r1_wrong if true_text and ET.match_value(r1[f], true_text)]
    genuine = [f for f in r1_wrong if f not in set(printed)]
    rescued = [f for f in genuine if f in r2 and right(r2[f])]
    core_same = [f for f in genuine if f in r2 and not right(r2[f])
                 and canon(r1[f]) == canon(r2[f])]
    both_wrong_diff = [f for f in genuine if f in r2 and not right(r2[f])
                       and canon(r1[f]) != canon(r2[f])]
    r1_wrong_r2_silent = [f for f in genuine if f not in r2]
    # the price: a value r1 had RIGHT that r2 contradicted — agreement refuses it
    r1_right_r2_wrong = [f for f, v in r1.items() if right(v) and f in r2
                         and not right(r2[f])]
    detail = [(f, r1[f], r2.get(f)) for f in genuine]
    return {
        "r1_n": len(r1), "r2_n": len(r2),
        "r1_wrong": len(r1_wrong), "r1_wrong_printed": len(printed),
        "r1_wrong_genuine": len(genuine),
        "r1_wrong_r2_right": len(rescued), "r1_wrong_r2_silent": len(r1_wrong_r2_silent),
        "both_wrong_agree_same": len(core_same), "both_wrong_differ": len(both_wrong_diff),
        "r1_right_r2_wrong": len(r1_right_r2_wrong),
        "rescued": rescued, "core_same": core_same, "both_wrong_diff_list": both_wrong_diff,
        "r1_wrong_detail": detail,
    }


def score_values(values, gt_values, true_text=""):
    """(spoken, hit, wrong_list, recovered_list, printed_count) for a set of
    spoken values against one page's ground-truth values — the repo's one match
    rule. `printed_count` is eval_field_truth's own sub-count: a wrong value that
    IS printed on the page but anchors no ground-truth item, because the label
    that would anchor it was never spotted. A '400MG' read off an IBUPROFEN
    400MG line is not a recognition error and must not be reported as one."""
    spoken = list(values)
    hit = [v for v in spoken if any(ET.match_value(v, g) for g in gt_values)]
    wrong = [v for v in spoken if not any(ET.match_value(v, g) for g in gt_values)]
    recovered = [g for g in gt_values if any(ET.match_value(v, g) for v in spoken)]
    printed = [v for v in wrong if true_text and ET.match_value(v, true_text)]
    return len(spoken), len(hit), wrong, recovered, len(printed)


# ── aggregation ───────────────────────────────────────────────────────────────

def new_acc():
    return {"spoken": 0, "hit": 0, "wrong": [], "printed": 0, "gt": 0, "rec": 0}


def add(acc, spoken, hit, wrong, recovered, printed, n_gt):
    acc["spoken"] += spoken
    acc["hit"] += hit
    acc["wrong"] += list(wrong)
    acc["printed"] += printed
    acc["rec"] += len(recovered)
    acc["gt"] += n_gt


def _r(k, n):
    return "-" if not n else "%.3f" % (k / n)


def _ci(ci):
    return "n/a" if not ci else "[%.3f,%.3f]" % ci


def _wilson(k, n):
    return CR.wilson(k, n) if n else None


def print_reader_table(accs, order, ens, meta):
    print("\n" + "=" * 118)
    print("SIGHTLINE — INDEPENDENT SECOND READ   %s   sha256 %s   %d pages, %d line crops"
          % (meta["path"], meta["sha256"], meta["pages"], meta["lines"]))
    print("=" * 118)
    print("Safety-critical kinds only (%s). 'wrong-but-spoken' = a spoken value "
          "that conveys no ground-truth value; 'printed' = of those, the ones "
          "printed on the page (no gt item: the anchoring label was never spotted)."
          % ", ".join(sorted(CRITICAL)))
    print("%-24s %7s %7s %-16s %6s %-16s %7s %6s %6s %-16s"
          % ("reader", "spoken", "prec", "prec 95% CI", "wrong", "wrong 95% CI",
             "printed", "gt", "recall", "recall 95% CI"))
    for name, a in list(accs.items()) + [("ENSEMBLE(agree)", ens)]:
        ps, ph, gs, rh, w = a["spoken"], a["hit"], a["gt"], a["rec"], len(a["wrong"])
        print("%-24s %7d %7s %-16s %6d %-16s %7d %6d %6s %-16s"
              % (name, ps, _r(ph, ps), _ci(_wilson(ph, ps)) if ps else "n/a",
                 w, _ci(_wilson(w, ps)) if ps else "n/a", a["printed"],
                 gs, _r(rh, gs), _ci(_wilson(rh, gs)) if gs else "n/a"))
    return


def print_decorr(tot, meta):
    print("\nDECORRELATION — of the values r1 got WRONG, how many did r2 get RIGHT?")
    print("  (GENUINE wrong only: a value printed on the page with no gt item is "
          "reported separately — it is not a recognition error.)")
    print("  r1 critical values           %d" % tot["r1_n"])
    print("  r2 critical values           %d" % tot["r2_n"])
    print("  r1 WRONG                     %d   (%d printed, no gt item; %d genuine)"
          % (tot["r1_wrong"], tot["r1_wrong_printed"], tot["r1_wrong_genuine"]))
    print("    of the GENUINE ones:")
    print("      r2 got RIGHT             %d   <- RESCUED (the whole question)"
          % tot["r1_wrong_r2_right"])
    print("      r2 was silent            %d" % tot["r1_wrong_r2_silent"])
    print("      r2 also wrong")
    print("          same string (CORE)   %d   <- irreducible, agreement cannot fix"
          % tot["both_wrong_agree_same"])
    print("          different string     %d   <- agreement REFUSES these" % tot["both_wrong_differ"])
    print("  r1 RIGHT but r2 WRONG        %d   <- the price: agreement would refuse a"
          % tot["r1_right_r2_wrong"])
    print("                                        value r1 had right")
    if tot["core_same"]:
        print("  irreducible-core strings (both wrong, same string):")
        for f, v in tot["core_same"][:8]:
            print("    %-18s %s" % (f, v))
    if tot["rescued"]:
        print("  rescued (r1 wrong, r2 right):")
        for f, v1, v2 in tot["rescued"][:8]:
            print("    %-18s %-14s -> %s" % (f, v1, v2))
    if tot["r1_wrong_detail"]:
        print("  every GENUINE r1-wrong critical value and what r2 produced:")
        for doc, f, v1, v2 in tot["r1_wrong_detail"][:12]:
            print("    %-16s %-16s r1=%-14s r2=%s"
                  % (doc, f, v1, ("(silent)" if v2 is None else v2)))


# ── the harness: corpus -> readers -> the pure logic above ────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="independent second read: does AGREEMENT between two reads of "
                    "the same crop catch the one-digit errors no policy can?")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-pages", type=int, default=120)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--extras", default=",".join(DEFAULT_EXTRAS),
                    help="comma-separated second readers, from %s"
                         % ",".join(sorted(READERS)))
    a = ap.parse_args(argv)

    extras = [x.strip() for x in a.extras.split(",") if x.strip()]
    for x in extras:
        if x not in READERS:
            ap.error("unknown reader %r (known: %s)" % (x, sorted(READERS)))

    data = RC.load(split=a.split)
    if not data["crops"].shape[0]:
        print("no crops in split=%s: build the corpus first" % a.split)
        return 1
    groups_all = ET.page_groups(data)
    groups = ET.page_groups(data, a.limit_pages)

    print("SightLine — LANE O, the independent second read")
    print("checkpoint %s   sha256 %s" % (a.ckpt, FR.sha256_prefix(a.ckpt)))
    print("split=%-5s pages=%d of %d   extras=%s   label_spot %.3f   ckpt=%s"
          % (a.split, len(groups), len(groups_all), extras, a.threshold, a.ckpt))
    print("R1 = crop_to_array + eval_ocr.recognise (unchanged)   R2 = CLAHE crop "
          "-> same   R3 = crop_to_array@48px")

    import torch
    torch.set_num_threads(1)

    model = None
    for attempt in range(3):
        try:
            model, device = ER.load_model(a.ckpt)
            break
        except RuntimeError as e:
            if attempt == 2:
                print("SKIPPED %s — %s" % (a.ckpt, e))
                return 1
            import time
            time.sleep(2.0)
    if model is None:
        return 1

    names = [BASELINE] + extras
    accs = {n: new_acc() for n in names}
    ens_acc = new_acc()
    tot = {"r1_n": 0, "r2_n": 0, "r1_wrong": 0, "r1_wrong_printed": 0,
           "r1_wrong_genuine": 0, "r1_wrong_r2_right": 0,
           "r1_wrong_r2_silent": 0, "both_wrong_agree_same": 0, "both_wrong_differ": 0,
           "r1_right_r2_wrong": 0, "rescued": [], "core_same": [], "r1_wrong_detail": []}
    rows = []
    n_crops = 0
    for g in groups:
        lines = {n: read_page(g["crops"], model, device, n) for n in names}
        n_crops += len(g["crops"])
        page_text = " ".join(str(t) for t in g["true"])
        # gt items are anchored on the BASELINE read's spots so EVERY reader and
        # the ensemble are scored on the identical item set (this is what makes
        # the McNemar paired).
        gt_pairs, _fp = ET.ground_truth(g["true"], lines[BASELINE], a.threshold)
        gt_values = [v for _, v in gt_pairs]

        crit = {}
        for n in names:
            crit[n], _dec, _fields = critical_spoken(lines[n], g["doc_type"], a.threshold)
            sp, hit, wrong, rec, printed = score_values(crit[n].values(), gt_values, page_text)
            add(accs[n], sp, hit, wrong, rec, printed, len(gt_values))

        ens_vals, _refusals = ensemble(crit[BASELINE], crit[PRIMARY_R2])
        sp, hit, wrong, rec, printed = score_values(ens_vals.values(), gt_values, page_text)
        add(ens_acc, sp, hit, wrong, rec, printed, len(gt_values))

        A = accord(crit[BASELINE], crit[PRIMARY_R2], gt_values, page_text)
        for k in ("r1_n", "r2_n", "r1_wrong", "r1_wrong_printed", "r1_wrong_genuine",
                  "r1_wrong_r2_right", "r1_wrong_r2_silent", "both_wrong_agree_same",
                  "both_wrong_differ", "r1_right_r2_wrong"):
            tot[k] += A[k]
        for f in A["rescued"]:
            tot["rescued"].append((f, crit[BASELINE][f], crit[PRIMARY_R2][f]))
        for f in A["core_same"]:
            tot["core_same"].append((f, crit[BASELINE][f]))
        for f, v1, v2 in A["r1_wrong_detail"]:
            tot["r1_wrong_detail"].append((g["doc"], f, v1, v2))

        # the shipped typed-only policy on R1: the reference row AND the paired
        # baseline. Scored on the SAME pages/items through eval_field_truth.
        prow = ET.score_page(g["true"], lines[BASELINE], g["doc_type"], a.threshold,
                             policy_variants={"typed-only": {"speak_free_text": False}})
        p = prow["policy"]["typed-only"]
        rows.append({"doc_type": g["doc_type"], "profile": g["profile"], "doc": g["doc"],
                     "gt_values": gt_values,
                     "ensemble_recovered": [g_ for g_ in gt_values
                                            if any(ET.match_value(v, g_)
                                                   for v in ens_vals.values())],
                     "tdyfull_recovered": p["recovered"],
                     "tdycrit_recovered": [g_ for g_ in gt_values
                                           if any(ET.match_value(v, g_)
                                                  for v in crit[BASELINE].values())],
                     "typed_only_spoken": p["n_spoken"], "typed_only_hit": p["n_hit"],
                     "typed_only_wrong": len(p["wrong"]), "typed_only_gt": len(gt_values)})
    meta = {"path": a.ckpt, "sha256": FR.sha256_prefix(a.ckpt),
            "pages": len(groups), "lines": n_crops}

    print_reader_table(accs, names, ens_acc, meta)
    print_decorr(tot, meta)

    # typed-only reference (all kinds) — the pinned baseline these numbers must
    # reproduce, then the critical-only typed-only row on the same item set.
    tf_sp = sum(r["typed_only_spoken"] for r in rows)
    tf_hit = sum(r["typed_only_hit"] for r in rows)
    tf_wrong = sum(r["typed_only_wrong"] for r in rows)
    tf_gt = sum(r["typed_only_gt"] for r in rows)
    tf_rec = sum(len(r["tdyfull_recovered"]) for r in rows)
    print("\nREFERENCE — typed-only policy on R1 (all kinds), the shipped baseline")
    print("  spoken %d  precision %s %s  wrong-but-spoken %d  recall %s %s"
          % (tf_sp, _r(tf_hit, tf_sp), _ci(_wilson(tf_hit, tf_sp)), tf_wrong,
             _r(tf_rec, tf_gt), _ci(_wilson(tf_rec, tf_gt))))
    print("  (this is the 0.957 precision / 0.444 recall / 14-wrong baseline the "
          "one-digit errors survive — reproducing it validates the path)")

    print("\nPAIRED, same pages and same items, exact McNemar (compare_runs verdicts)")
    print("%-34s %10s %10s %9s %11s %s"
          % ("comparison", "old k/n", "new k/n", "delta", "pairs b/c", "p / verdict"))
    for oldk, label in (("tdyfull_recovered", "typed-only(all) -> ENSEMBLE"),
                        ("tdycrit_recovered", "typed-only(crit) -> ENSEMBLE")):
        r = ET.paired(rows, oldk, "ensemble_recovered")
        b, c = r["discordant"] or ("-", "-")
        p_str = "n/a" if r["p"] is None else "%.4g" % r["p"]
        print("%-34s %10s %10s %+9.3f %11s p=%-8s %s"
              % (label, "%s/%s" % (r["k_old"], r["n"]), "%s/%s" % (r["k_new"], r["n"]),
                 r["delta"] or 0.0, "%s/%s" % (b, c), p_str, r["verdict"]))

    extra_passes = n_crops * len(extras)
    per_page = extra_passes / max(1, len(groups))
    print("\nCOST — %d line crops over %d pages; second read adds %d extra forward "
          "passes (%.1f per page) on top of the %d the baseline already runs"
          % (n_crops, len(groups), extra_passes + n_crops, per_page, n_crops))
    print("  readers run per page: %s" % ", ".join(names))
    print("\n(R4 digits-only constrained_decode is deliberately NOT scored: it "
          "re-decodes the SAME logits, so it is not independent — a tie-breaker, "
          "not a witness. Lane J measured it never regresses but fixes only 3/148.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
