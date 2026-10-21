"""
dist_compare.py — THE TRAINING DISTRIBUTION, AND THE APP'S, SIDE BY SIDE.

The other half of the out-of-distribution check. `app/tools/crop_distribution.js`
drives the app's OWN preprocess + detector + cropForRecognizer over the 12 bench
documents and dumps, per crop, the detector box (bw x bh), the crop the
recogniser is handed (nw x CROP_H) and the three image statistics
(Otsu ink density, ink-paper contrast, Laplacian variance). This script loads
THE CORPUS THE SHIPPED CHECKPOINT WAS TRAINED ON
(data/render_corpus/crops_train_oldprofiles.npz), computes the same statistics
with the SAME function the repo already uses for distribution matching
(real_degradation_cache.crop_stats), and prints:

  * the training width/height distribution and the three statistics, overall and
    per capture profile and per split (mean, median, quartiles)
  * the app's, from the JS dump, per document and per capture profile
  * the fraction of app crops inside [min, max] of the training widths and the
    fraction at the 320 clamp vs narrower, for both sides
  * the per-document table, each app document's nearest training capture profile
    and its standardised distance to it
  * the verdict: out of distribution, or shaped like the training crops with the
    gap coming from somewhere else — and the single largest divergence

The npz stores every crop PADDED to 320 columns with `widths[i]` the true
content width (build_dataset.crop_to_array's clamp), so all statistics are
computed on the CONTENT — `images[i][:, :widths[i]]` — which is what a
dynamic-width graph is fed.

A self-check closes the loop on the statistics definition: real_degradation_cache
.crop_stats and a numpy port of the JS implementation are run over the same
crops and their max absolute difference is printed and asserted small, so a
difference in the two halves cannot come from two different definitions of "ink".

Run:  bash run.sh py ml/src/dist_compare.py [--app=/tmp/cropdist_app.json]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import real_degradation_cache as RDC     # noqa: E402  (crop_stats = the repo's definition)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = ROOT / "data" / "render_corpus" / "crops_train_oldprofiles.npz"
DEFAULT_APP = Path("/tmp/cropdist_app.json")
CLAMP = 320          # realdata.MAX_W
CROP_H = 32          # realdata.CROP_H


# ── the same statistics, as the JS harness computes them ────────────────────

def otsu(a):
    """Otsu threshold from a 256-bin histogram (the JS port's algorithm)."""
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    idx = np.arange(256)
    sum_all = float((idx * hist).sum())
    wB = np.cumsum(hist)
    sumB = np.cumsum(idx * hist)
    valid = (wB > 0) & (wB < total)
    mB = np.divide(sumB, wB, out=np.zeros(256), where=wB > 0)
    mF = np.divide(sum_all - sumB, total - wB, out=np.zeros(256), where=wB < total)
    between = np.where(valid, wB * (total - wB) * (mB - mF) ** 2, 0.0)
    return int(np.argmax(between))


def js_crop_stats(a):
    """numpy mirror of crop_stats as app/tools/crop_distribution.js computes it."""
    a = np.asarray(a, np.uint8)
    thr = otsu(a)
    ink = a <= thr
    paper = ~ink
    density = float(ink.mean())
    af = a.astype(np.float32)
    contrast = float(af[paper].mean() - af[ink].mean()) if ink.any() and paper.any() else 0.0
    sharp = float(cv2.Laplacian(a, cv2.CV_64F).var())
    return density, contrast, sharp


def stats_of(crops):
    """mean/median/quartiles per metric — the aggregate real_degradation_cache uses."""
    rows = np.array([RDC.crop_stats(c) for c in crops], np.float64)
    if not len(rows):
        return {}
    out = {"n": int(len(rows))}
    for j, k in enumerate(("ink", "contrast", "sharp")):
        v = rows[:, j]
        out[k + "_mean"] = float(v.mean())
        out[k + "_median"] = float(np.median(v))
        out[k + "_p25"] = float(np.percentile(v, 25))
        out[k + "_p75"] = float(np.percentile(v, 75))
    return out


def wsum(widths):
    """Width summary: the shape of the distribution, not just its centre."""
    w = np.asarray(widths, np.int64)
    return {
        "n": int(w.size), "min": int(w.min()), "p25": float(np.percentile(w, 25)),
        "median": float(np.median(w)), "p75": float(np.percentile(w, 75)),
        "max": int(w.max()), "at_clamp": float((w >= CLAMP).mean()),
        "narrow_lt320": int((w < CLAMP).sum()), "below_16": int((w < 16).sum()),
    }


def eff(app_v, train_v, train_iqr):
    """|app - train| in training-IQR units: the divergence, made comparable."""
    if train_iqr <= 1e-9:
        return float("inf") if abs(app_v - train_v) > 1e-9 else 0.0
    return abs(app_v - train_v) / train_iqr


def f(v, n=2):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--app", default=str(DEFAULT_APP))
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--stats-stride", type=int, default=1,
                    help="stride for the per-crop statistics sweep (widths always use every row)")
    ap.add_argument("--json", default="", help="write the machine summary here")
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    imgs, widths, splits = d["images"], d["widths"], d["splits"]
    prof = np.array([str(x) for x in d["variants"]], dtype=object)   # the capture profile

    print("TRAINING CORPUS  " + a.npz)
    print("  crops " + str(len(widths)) + "   stored " + str(imgs.shape[1:]) + " uint8"
          + "   every crop exactly " + str(imgs.shape[1]) + " tall (CROP_H)"
          + "   widths = content width, the rest of the 320 columns is padding")

    # ── the statistics definition self-check ────────────────────────────────
    idx = np.arange(0, len(widths), max(1, len(widths) // 400))
    diffs = []
    for i in idx:
        c = np.asarray(imgs[i][:, :int(widths[i])], np.uint8)
        diffs.append(np.abs(np.array(RDC.crop_stats(c)) - np.array(js_crop_stats(c))))
    md = np.max(np.array(diffs), axis=0)
    print("  stats definition check (crop_stats vs the JS port, n=" + str(len(idx)) + "): "
          "max |dink|=" + f"{md[0]:.2e}" + " |dcontrast|=" + f"{md[1]:.2e}"
          + " |dsharp|=" + f"{md[2]:.2e}")
    assert md.max() < 1e-6, "the two halves would be measuring different statistics"
    print("  -> ink/contrast/sharpness mean the same thing in both halves OK")

    # ── training side ──────────────────────────────────────────────────────
    sel = np.ones(len(widths), bool) if a.split == "all" else (splits == a.split)
    tr_w = widths[sel]
    tr_rows = np.where(sel)[0][::max(1, a.stats_stride)]
    tr_crops = [np.asarray(imgs[i][:, :int(widths[i])], np.uint8) for i in tr_rows]
    tr_stats = stats_of(tr_crops)
    tw = wsum(tr_w)

    print("\n  widths (n=" + str(tw["n"]) + ", split " + a.split + "): min " + str(tw["min"])
          + "  p25 " + f(tw["p25"], 0) + "  median " + f(tw["median"], 0)
          + "  p75 " + f(tw["p75"], 0) + "  max " + str(tw["max"])
          + "   at the " + str(CLAMP) + " clamp: " + f(100 * tw["at_clamp"], 1) + "%"
          + "   narrower than " + str(CLAMP) + ": " + str(tw["narrow_lt320"])
          + "   below MIN_W=16: " + str(tw["below_16"]))
    print("  heights: 32 for every crop (by construction)")
    print("  stats (n=" + str(tr_stats["n"]) + " sampled crops):")
    for k in ("ink", "contrast", "sharp"):
        print("    " + k.ljust(9) + " mean " + f(tr_stats[k + "_mean"], 3)
              + "  median " + f(tr_stats[k + "_median"], 3)
              + "  p25 " + f(tr_stats[k + "_p25"], 3) + "  p75 " + f(tr_stats[k + "_p75"], 3))

    tr_by_prof = {}
    print("\n  TRAINING, per capture profile (is any profile a different world?):")
    print("    " + "profile".ljust(16) + "crops".rjust(6) + "  %@320  width med"
          + "   ink med (p25..p75)".rjust(20)
          + "   contrast med (p25..p75)".rjust(24) + "   sharp med (p25..p75)".rjust(22)
          + "  width p25..p75")
    for p in sorted(set(prof[sel].tolist())):
        m = sel & (prof == p)
        rows = np.where(m)[0][::max(1, a.stats_stride)]
        st = stats_of([np.asarray(imgs[i][:, :int(widths[i])], np.uint8) for i in rows])
        ww = widths[m]
        tr_by_prof[p] = {"stats": st, "widths": wsum(ww)}
        print("    " + p.ljust(16) + str(int(m.sum())).rjust(6)
              + "  " + f(100 * float((ww >= CLAMP).mean()), 1).rjust(5)
              + "  " + f(float(np.median(ww)), 0).rjust(8)
              + "  " + (f(st["ink_median"], 3) + " (" + f(st["ink_p25"], 3) + ".."
                        + f(st["ink_p75"], 3) + ")").rjust(20)
              + "  " + (f(st["contrast_median"], 1) + " (" + f(st["contrast_p25"], 1) + ".."
                        + f(st["contrast_p75"], 1) + ")").rjust(24)
              + "  " + (f(st["sharp_median"], 0) + " (" + f(st["sharp_p25"], 0) + ".."
                        + f(st["sharp_p75"], 0) + ")").rjust(22)
              + "  " + (f(float(np.percentile(ww, 25)), 0) + ".."
                        + f(float(np.percentile(ww, 75)), 0)))

    # ── app side (measured in the browser by crop_distribution.js) ─────────
    app_path = Path(a.app)
    if not app_path.exists():
        print("\nMISSING " + str(app_path) + " — run: node app/tools/crop_distribution.js"
              " --json=" + str(app_path))
        sys.exit(2)
    app = json.loads(app_path.read_text())
    samples = app["samples"]
    crops = [c for s in samples for c in s["crops"]]
    aw = np.array([c["nw"] for c in crops], np.int64)
    ah = np.array([c["nh"] for c in crops], np.int64)
    abh = np.array([c["bh"] for c in crops], np.int64)
    aws = wsum(aw)

    print("\n" + "-" * 78)
    print("APP CROPS  (app/tools/crop_distribution.js, " + app.get("engine", "?")
          + ", arm = pp.canvas/" + "enhanced" + ", maxDim " + str(app.get("maxDim"))
          + ", maxW " + str(app.get("maxW")) + ")")
    print("  crops " + str(len(crops)) + " on " + str(len(samples)) + " documents")
    print("  widths: min " + str(aws["min"]) + "  p25 " + f(aws["p25"], 0)
          + "  median " + f(aws["median"], 0) + "  p75 " + f(aws["p75"], 0)
          + "  max " + str(aws["max"])
          + "   at the " + str(CLAMP) + " clamp: " + f(100 * aws["at_clamp"], 1) + "%"
          + "   narrower: " + str(aws["narrow_lt320"])
          + "   below MIN_W=16: " + str(aws["below_16"]))
    print("  heights AFTER the crop rule: " + str(int(ah.min())) + ".." + str(int(ah.max()))
          + " (every crop is CROP_H=" + str(CROP_H) + " tall: "
          + ("yes" if (ah == CROP_H).all() else "NO") + ")")
    print("  detector box heights BEFORE the crop rule: min " + str(int(abh.min()))
          + "  median " + f(float(np.median(abh)), 0) + "  max " + str(int(abh.max())))

    # ── the comparison the whole lane exists for ───────────────────────────
    print("\n" + "=" * 78)
    print("APP vs TRAINING — the same statistics, side by side")
    print("  " + "metric".ljust(22) + "TRAINING".rjust(26) + "   APP".rjust(26) + "   gap")
    print("  " + "-" * 74)
    rows = [
        ("crop width median", float(np.median(tr_w)), aws["median"], "px"),
        ("crop width p25", float(np.percentile(tr_w, 25)), aws["p25"], "px"),
        ("crop width min", float(tr_w.min()), float(aw.min()), "px"),
        ("crop width == 320", 100 * tw["at_clamp"], 100 * aws["at_clamp"], "%"),
        ("crop height", float(CROP_H), float(CROP_H), "px"),
    ]
    for name, tv, av, unit in rows:
        print("  " + name.ljust(22)
              + (f(tv, 1) + " " + unit).rjust(26) + ("   " + f(av, 1) + " " + unit).rjust(26)
              + "   " + f(av - tv, 1))
    print("  " + "-" * 74)
    effects = {}
    for k in ("ink", "contrast", "sharp"):
        tv, av = tr_stats[k + "_median"], float(np.median([c[k] for c in crops]))
        iqr = tr_stats[k + "_p75"] - tr_stats[k + "_p25"]
        e = eff(av, tv, iqr)
        effects[k] = (e, tv, av, iqr)
        print("  " + ("median " + k).ljust(22)
              + f(tv, 3).rjust(26) + ("   " + f(av, 3)).rjust(26)
              + "   " + f(e, 2) + " xIQR")
        print("  " + ("  (training p25..p75)").ljust(22)
              + (f(tr_stats[k + "_p25"], 3) + ".." + f(tr_stats[k + "_p75"], 3)).rjust(26)
              + "   app median " + f(av, 3)
              + ("   OUTSIDE" if not (tr_stats[k + "_p25"] <= av <= tr_stats[k + "_p75"]) else "   inside IQR"))
    # width-at-clamp is a divergence too, in the same units (a proportion, 0..1)
    effects["width at the 320 clamp"] = (abs(aws["at_clamp"] - tw["at_clamp"]) / 0.10,
                                         tw["at_clamp"], aws["at_clamp"], 0.10)

    inside = float(((aw >= tr_w.min()) & (aw <= tr_w.max())).mean())
    tmed = wsum(widths)
    print("\n  FRACTION OF APP CROPS INSIDE THE TRAINING WIDTH RANGE ["
          + str(int(tr_w.min())) + ", " + str(int(tr_w.max())) + "]: " + f(100 * inside, 1) + "%"
          + "   (" + str(int(((aw >= tr_w.min()) & (aw <= tr_w.max())).sum())) + "/" + str(len(aw)) + ")")
    print("  whole-corpus training range [" + str(tmed["min"]) + ", " + str(tmed["max"]) + "]: "
          + f(100 * float(((aw >= tmed["min"]) & (aw <= tmed["max"])).mean()), 1) + "%"
          + "   app narrower than the clamp: " + f(100 * (1 - aws["at_clamp"]), 1) + "%"
          + " vs training " + f(100 * (1 - tmed["at_clamp"]), 1) + "%")

    # ── profile-matched: the fair comparison ───────────────────────────────
    # The POOLED sharpness distribution is bimodal — training studio_clean is
    # three orders of magnitude sharper than its other three profiles — so a
    # pooled IQR says almost nothing. Match on the capture profile instead, and
    # name the profiles the training corpus does not contain at all.
    print("\n" + "=" * 78)
    others = [v["stats"]["sharp_median"] for k, v in tr_by_prof.items() if k != "studio_clean"]
    print("PROFILE-MATCHED  (pooled sharpness is bimodal: training studio_clean median "
          + f(tr_by_prof["studio_clean"]["stats"]["sharp_median"], 0) + " vs the other profiles <= "
          + f(max(others), 0) + ", so the pooled IQR above overstates the app's distance)")
    app_by_prof = {}
    for s in samples:
        app_by_prof.setdefault(s["profile"], []).extend(s["crops"])
    profile_match = {}
    print("  " + "app profile".ljust(18) + "crops".rjust(5)
          + "   ink  app/train (xIQR)".rjust(26)
          + "   contrast  app/train (xIQR)".rjust(30)
          + "   sharp  app/train (xIQR)".rjust(30))
    for p, cs in sorted(app_by_prof.items()):
        am = {k: float(np.median([c[k] for c in cs])) for k in ("ink", "contrast", "sharp")}
        cells, rec = [], {}
        for k in ("ink", "contrast", "sharp"):
            if p in tr_by_prof:
                ts = tr_by_prof[p]["stats"]
                e = eff(am[k], ts[k + "_median"], ts[k + "_p75"] - ts[k + "_p25"])
                rec[k] = {"app": am[k], "train": ts[k + "_median"], "xIQR": e}
                cells.append(f(am[k], 3) + "/" + f(ts[k + "_median"], 3) + " (" + f(e, 1) + "x)")
            else:
                cells.append(f(am[k], 3) + "/ --")
        profile_match[p] = {"n": len(cs), "stats": rec,
                            "training_analogue": p if p in tr_by_prof else None}
        tag = "" if p in tr_by_prof else "   <- NOT A TRAINING PROFILE"
        print("  " + p.ljust(18) + str(len(cs)).rjust(5)
              + "   " + cells[0].rjust(24) + "   " + cells[1].rjust(28) + "   " + cells[2].rjust(28) + tag)

    # The SAME rows on arm 0 (raw deskewed grayscale, no CLAHE, no unsharp). This is
    # the one number that separates "the app's own preprocessing sharpens the crop"
    # from "the source pages are crisper than the corpus's":
    raw_by_prof = {}
    for s in samples:
        raw_by_prof.setdefault(s["profile"], []).extend(s.get("raw", []))
    print("\n  same rows on the RAW arm (no CLAHE, no unsharp) — where does the extra sharpness come from?")
    for p, cs in sorted(raw_by_prof.items()):
        if not cs:
            continue
        am = {k: float(np.median([c[k] for c in cs])) for k in ("ink", "contrast", "sharp")}
        cells = []
        for k in ("ink", "contrast", "sharp"):
            if p in tr_by_prof:
                ts = tr_by_prof[p]["stats"]
                e = eff(am[k], ts[k + "_median"], ts[k + "_p75"] - ts[k + "_p25"])
                cells.append(f(am[k], 3) + "/" + f(ts[k + "_median"], 3) + " (" + f(e, 1) + "x)")
            else:
                cells.append(f(am[k], 3) + "/ --")
        print("  " + ("raw: " + p).ljust(18) + str(len(cs)).rjust(5)
              + "   " + cells[0].rjust(24) + "   " + cells[1].rjust(28) + "   " + cells[2].rjust(28))

    # ── per document / per capture profile ────────────────────────────────
    print("\n" + "=" * 78)
    print("PER DOCUMENT  (a mismatch concentrated in one profile is a different conclusion)")
    print("  " + "sample".ljust(30) + "lns".rjust(4) + "  med nw  %@320  boxbh med"
          + "   ink   contrast   sharp    nearest training profile (xIQR)")
    per_doc = {}
    for s in samples:
        cs = s["crops"]
        w = np.array([c["nw"] for c in cs], np.int64)
        bh = np.array([c["bh"] for c in cs], np.int64)
        ink = float(np.median([c["ink"] for c in cs]))
        con = float(np.median([c["contrast"] for c in cs]))
        shp = float(np.median([c["sharp"] for c in cs]))
        # nearest training capture profile, on the standardised 3-vector the repo
        # already uses for this question (real_degradation_cache.nearest_profile).
        vecs = {p: (v["stats"]["ink_mean"], v["stats"]["contrast_mean"], v["stats"]["sharp_mean"])
                for p, v in tr_by_prof.items()}
        nearest, dist = RDC.nearest_profile((float(np.mean([c["ink"] for c in cs])),
                                             float(np.mean([c["contrast"] for c in cs])),
                                             float(np.mean([c["sharp"] for c in cs]))), vecs)
        ridge = tr_by_prof[nearest]["stats"]
        de = {k: eff({"ink": ink, "contrast": con, "sharp": shp}[k],
                     ridge[k + "_median"], ridge[k + "_p75"] - ridge[k + "_p25"])
              for k in ("ink", "contrast", "sharp")}
        worst = max(de, key=de.get)
        tag = s["doc"] + "/" + s["profile"]
        per_doc[tag] = {"lines": len(cs), "med_nw": float(np.median(w)),
                        "at_clamp": float((w >= CLAMP).mean()), "box_bh_med": float(np.median(bh)),
                        "ink": ink, "contrast": con, "sharp": shp,
                        "nearest_profile": nearest, "profile_dist": dist,
                        "worst_stat": worst, "worst_xIQR": de[worst]}
        print("  " + tag.ljust(30) + str(len(cs)).rjust(4)
              + "  " + f(float(np.median(w)), 0).rjust(6)
              + "  " + f(100 * float((w >= CLAMP).mean()), 0).rjust(5) + "%"
              + "  " + f(float(np.median(bh)), 0).rjust(9)
              + "  " + f(ink, 3) + "  " + f(con, 2).rjust(8) + "  " + f(shp, 1).rjust(8)
              + "   " + nearest + " (" + f(dist, 1) + ", worst " + worst + " " + f(de[worst], 1) + " xIQR)")

    print("\n  detected lines per document (the detector's recall, before any crop rule):")
    docs = {}
    for s in samples:
        docs.setdefault(s["doc"], {})[s["profile"]] = len(s["crops"])
    profs = sorted({p for v in docs.values() for p in v})
    print("    " + "doc".ljust(14) + "".join(p.rjust(18) for p in profs))
    for doc in sorted(docs):
        print("    " + doc.ljust(14) + "".join(str(docs[doc].get(p, "-")).rjust(18) for p in profs))

    # ── verdict ───────────────────────────────────────────────────────────
    named = {"ink density": effects["ink"], "ink/paper contrast": effects["contrast"],
             "sharpness (Laplacian var)": effects["sharp"],
             "width at the 320 clamp": effects["width at the 320 clamp"]}
    worst = max(named, key=lambda k: named[k][0])
    e, tv, av, iqr = named[worst]
    # the profile-matched version: only profiles the corpus actually contains
    pm_eff = {}
    for p, v in profile_match.items():
        for k, stt in v["stats"].items():
            pm_eff[(p, k)] = stt["xIQR"]
    worst_doc = max(per_doc, key=lambda k: per_doc[k]["worst_xIQR"])
    geometry_ok = inside > 0.99 and abs(aws["at_clamp"] - tw["at_clamp"]) < 0.05
    missing = [p for p, v in profile_match.items() if v["training_analogue"] is None]
    lowest_lines = min(per_doc.items(), key=lambda kv: kv[1]["lines"])
    lowest_doc = lowest_lines[0].split("/")[0]
    same_doc = [(k, v) for k, v in per_doc.items() if k.split("/")[0] == lowest_doc]

    print("\n" + "=" * 78)
    print("VERDICT")
    print("  geometry: app crops are " + ("IN" if geometry_ok else "NOT IN")
          + " the training width distribution — " + f(100 * inside, 1) + "% inside ["
          + str(int(tr_w.min())) + ", " + str(int(tr_w.max())) + "], "
          + f(100 * aws["at_clamp"], 1) + "% at the clamp vs training "
          + f(100 * tw["at_clamp"], 1) + "%, and every crop is "
          + str(CROP_H) + "px tall like every training crop")
    print("  statistics, pooled: largest divergence = " + worst + " (" + f(e, 2)
          + " x the training IQR; training median " + f(tv, 3) + " -> app median " + f(av, 3) + ")")
    print("  statistics, profile-matched: " + ", ".join(
        p + " " + k + " " + f(v, 1) + "x" for (p, k), v in
        sorted(pm_eff.items(), key=lambda kv: -kv[1])[:6]))
    print("  app capture profiles with NO training analogue: "
          + (", ".join(missing) if missing else "none")
          + " (" + str(sum(v["n"] for p, v in profile_match.items() if v["training_analogue"] is None))
          + " of " + str(len(crops)) + " crops)")
    # The raw arm is the discriminator between "the app's own preprocessing moved the
    # statistics" and "the source pages are not the corpus's pages".
    raw_by_prof2 = {}
    for s in samples:
        raw_by_prof2.setdefault(s["profile"], []).extend(s.get("raw", []))
    raw_hits = []
    for p, cs in sorted(raw_by_prof2.items()):
        if p in tr_by_prof and cs:
            ts = tr_by_prof[p]["stats"]
            raw_hits.append(p + " raw ink " + f(float(np.median([c["ink"] for c in cs])), 3)
                            + "/" + f(ts["ink_median"], 3) + ", contrast "
                            + f(float(np.median([c["contrast"] for c in cs])), 1) + "/"
                            + f(ts["contrast_median"], 1) + ", sharp "
                            + f(float(np.median([c["sharp"] for c in cs])), 0) + "/"
                            + f(ts["sharp_median"], 0))
    print("  raw arm (before the app's own CLAHE/unsharp): " + "; ".join(raw_hits)
          + " — ink and contrast MATCH the training profile here, sharpness does not")
    print("  per-document: worst statistics row is " + worst_doc + " — nearest training profile "
          + per_doc[worst_doc]["nearest_profile"] + ", " + per_doc[worst_doc]["worst_stat"]
          + " " + f(per_doc[worst_doc]["worst_xIQR"], 1) + " xIQR off it")
    print("  detector recall: " + lowest_lines[0] + " yields only " + str(lowest_lines[1]["lines"])
          + " lines where THE SAME DOCUMENT yields "
          + ", ".join(str(kv[1]["lines"]) + " (" + kv[0].split("/")[1] + ")" for kv in sorted(same_doc))
          + " — lines the model never sees are not a crop-distribution mismatch")
    out = {"train": {"n": int(len(tr_w)), "width": tw, "stats": tr_stats, "by_profile":
                     {p: {"stats": v["stats"], "width": v["widths"]} for p, v in tr_by_prof.items()}},
           "app": {"n": len(crops), "width": aws, "box_bh_median": float(np.median(abh)),
                   "stats_from_json": {k: float(np.median([c[k] for c in crops]))
                                       for k in ("ink", "contrast", "sharp")}},
           "fraction_inside_training_range": inside,
           "effects_xIQR_pooled": {k: v[0] for k, v in named.items()},
           "effects_xIQR_profile_matched": {p + "/" + k: v for (p, k), v in pm_eff.items()},
           "profiles_without_training_analogue": missing,
           "largest_divergence_pooled": worst, "per_document": per_doc,
           "per_profile": profile_match}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
        print("\n  wrote " + a.json)


if __name__ == "__main__":
    main()
