"""
profile_coverage.py — does the training corpus contain the photographic world the
app is served?

WHY THIS EXISTS
---------------
Four separate train/serve DISTRIBUTION mismatches have already moved this project's
accuracy more than any model change (the served width, the padding, the over-harsh
capture profiles, the preprocessing arm). This is the fifth, measured:

  * `app/tools/crop_distribution.js` drives the APP's own preprocess + detector +
    crop rule over the bench documents: 155 crops, of which 58 (37%) come from
    `off_axis` and `jpeg_social`.
  * The corpus the shipped checkpoint trained on was built from
    `ml/src/render_corpus.py`'s default `PROFILES[:4]` = studio_clean,
    handheld_light, handheld_heavy, low_light. So on 37% of what the app is
    served the model is asked about a photographic world it has never seen.

This tool answers, per profile the app can serve: IS IT IN THE CORPUS, and how far
are its statistics from the nearest corpus profile — in the SAME units
`ml/src/dist_compare.py` uses (|value - train median| / train IQR, via its `eff`),
computed with the SAME statistic definition (`real_degradation_cache.crop_stats`),
so the numbers are comparable with that report.

It also validates a SMALL sampled render of the proposed profile list under the
existing seed convention (`render_corpus.page_seed`, zlib.crc32 — never the
per-process `hash()`), and prints the ordered command list for the next full
corpus build with its expected page/crop counts and CPU cost.

IT NEVER RENDERS THE FULL CORPUS AND NEVER TRAINS. The full build is a decision
for the caller, made with the cost this tool measures.

USE
---
    bash run.sh py ml/src/profile_coverage.py                     # table + plan
    bash run.sh py ml/src/profile_coverage.py --sample-pages 3    # + sampled render
    bash run.sh py ml/src/profile_coverage.py --json /tmp/pc.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as B            # noqa: E402
import capture as C                  # noqa: E402
import dist_compare as DC            # noqa: E402  (eff/stats_of: the repo's own units)
import real_degradation_cache as RDC  # noqa: E402  (crop_stats/nearest_profile)
import render_corpus as RC           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "render_corpus"
SHIPPED_CACHE = CORPUS / "crops_train.npz"          # what retrain3 is training on now
OLD_CACHE = CORPUS / "crops_train_oldprofiles.npz"  # what the previous run trained on
APP_DUMP = Path("/tmp/cropdist_app.json")
BENCH_SCRIPT = ROOT / "app" / "tools" / "render_samples.py"
APP_FIXTURES = Path("/tmp/sightline_samples")

STATS = ("ink", "contrast", "sharp")

# worst_case is not a coverage target: evaluate.SUB_HUMAN marks it unreadable to a
# human, so a miss there is not a model failure and training on it teaches guessing.
# render_corpus.PROFILES already excludes it for the same reason.
SUB_HUMAN = {"worst_case"}


class UnknownProfile(ValueError):
    """A requested profile does not exist in capture.CAPTURE_PROFILES.

    Raised, never silently dropped: a dropped profile is a coverage gap the build
    believes it closed.
    """


def app_servable_profiles():
    """Every profile the app's pipeline can be served, in CAPTURE_PROFILES order.

    The app is served arbitrary photographs, so the coverage target is every
    profile the eval gate can reproduce, minus the one no human can read.
    """
    return [p for p in C.CAPTURE_PROFILES if p not in SUB_HUMAN]


def app_bench_profiles():
    """The profiles the app's OWN bench fixtures are rendered with.

    Read from app/tools/render_samples.py rather than retyped, so the two cannot
    drift: whichever list that file declares is the world crop_distribution.js
    measures the app's crops in.
    """
    m = re.search(r"^PROFILES\s*=\s*\[(.*?)\]", BENCH_SCRIPT.read_text(),
                  re.M | re.S)
    if not m:
        raise SystemExit("no PROFILES list in " + str(BENCH_SCRIPT))
    return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]


def resolve_profiles(spec):
    """A profile spec -> the validated list. Unknown names are named and fatal.

    spec: None or "" (the coverage target: every profile the app can serve),
          "bench" (just the app's bench fixture profiles), or a comma list.
    """
    known = set(C.CAPTURE_PROFILES)
    if spec is None or spec == "":
        return app_servable_profiles()
    if spec == "bench":
        return app_bench_profiles()
    want = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [p for p in want if p not in known]
    if bad:
        raise UnknownProfile(
            "unknown capture profile(s): " + ", ".join(bad)
            + "\n  known: " + ", ".join(sorted(known)))
    # keep CAPTURE_PROFILES order, drop duplicates
    return [p for p in C.CAPTURE_PROFILES if p in set(want)]


# ── statistics: the same definition and the same aggregate as dist_compare ────

def stats_of(crops):
    """median/p25/p75 per statistic over a list of uint8 crops."""
    return DC.stats_of(crops)


def corpus_stats(npz, stride=1):
    """{profile: stats} from the trainer's cache schema.

    Every crop is stored PADDED to 320 columns with `widths[i]` the true content
    width, so the statistics are computed on the CONTENT — what the dynamic-width
    graph is fed.
    """
    d = np.load(npz, allow_pickle=True)
    imgs, widths = d["images"], d["widths"]
    prof = np.array([str(x) for x in d["variants"]], dtype=object)
    out = {}
    for p in sorted(set(prof.tolist())):
        rows = np.where(prof == p)[0][::max(1, stride)]
        out[p] = stats_of([np.asarray(imgs[i][:, :int(widths[i])], np.uint8)
                           for i in rows])
        out[p]["width_median"] = float(np.median(widths[prof == p]))
    return out


def app_stats(dump=APP_DUMP):
    """{profile: {'n': crops, stats...}} from the app's own crop dump.

    The `enhanced` arm's numbers are the served crop; `crops[i]["sharp"]` etc. are
    the same three statistics computed inside the browser by the app's own code.
    """
    if not Path(dump).exists():
        return None
    a = json.loads(Path(dump).read_text())
    by = {}
    for s in a["samples"]:
        by.setdefault(s["profile"], []).extend(s["crops"])
    out = {"__dump__": {k: v for k, v in a.items() if k != "samples"}}
    for p, cs in by.items():
        out[p] = {"n": len(cs)}
        for k in STATS:
            out[p][k + "_median"] = float(np.median([c[k] for c in cs]))
            out[p][k + "_mean"] = float(np.mean([c[k] for c in cs]))
        out[p]["stats_source"] = "app crops (browser: preprocess+detector+crop rule)"
    return out


def nearest_and_gaps(query, cs, profile_medians=True):
    """(nearest corpus profile, standardised distance, per-stat gap in train-IQRs).

    `query` is a dict with ink/contrast/sharp medians; `cs` the corpus's per-profile
    stats. Distance is real_degradation_cache.nearest_profile's (the repo's own
    standardised 3-vector); the gaps are dist_compare.eff's |d| / train IQR, so
    every number here is in the units of the distribution report already on file.
    """
    vecs = {p: (v["ink_mean"], v["contrast_mean"], v["sharp_mean"])
            for p, v in cs.items()}
    nearest, dist = RDC.nearest_profile(
        (query["ink_mean"], query["contrast_mean"], query["sharp_mean"]), vecs)
    t = cs[nearest]
    gaps = {}
    for k in STATS:
        iqr = t[k + "_p75"] - t[k + "_p25"]
        gaps[k] = {"app": query[k + "_median"], "train": t[k + "_median"],
                   "xIQR": DC.eff(query[k + "_median"], t[k + "_median"], iqr)}
    return nearest, dist, gaps


# ── the sampled render: the existing seed convention, on the proposed list ────

def render_pages(profiles, pages_per_profile=3, seed=0, width=900,
                 scales=(0.85, 1.0, 1.15), page_offset=9000):
    """A SMALL deterministic render of one page per (profile, rep).

    Mirrors render_corpus.build's conventions exactly — page id -> font via
    zlib.crc32, page_seed for both the document and the profile, the same scale
    ladder — but pins the profile per page instead of drawing it at random, so a
    profile that fails to render or produces nothing cannot hide behind the mix.

    Returns [(profile, page_id, [crop arrays], label count, seconds)].
    """
    import random
    out = []
    for p_i, profile in enumerate(profiles):
        for r in range(pages_per_profile):
            dtype_lines = C.DOCS[(p_i + r) % len(C.DOCS)]
            lines, dtype = dtype_lines
            pid = "%s-%04d" % (dtype, page_offset + p_i * 100 + r)
            rng = random.Random(RC.page_seed(pid))
            scale = rng.choice(scales)
            t0 = time.time()
            clean = C.render_document(lines, width=width, seed=RC.page_seed(pid),
                                      scale=scale, font=RC.font_for_page(pid))
            deg = C.apply_profile(clean, seed=RC.page_seed(pid, 1),
                                  **C.CAPTURE_PROFILES[profile])
            gray = np.asarray(deg.convert("L"), dtype=np.uint8)
            crops = [c for c, _ in RC.page_crops(gray, lines, scale)]
            out.append((profile, pid, crops, len(lines), time.time() - t0))
    return out


def check_schema(crop):
    """Every crop must survive the trainer's transform: [CROP_H, W] within bounds."""
    a = B.crop_to_array(Image.fromarray(np.asarray(crop, np.uint8)))
    h, w = a.shape
    return h, w, (h == B.CROP_H and B.MIN_W <= w <= B.MAX_W)


def validate_sample(profiles, pages_per_profile=3, seed=0):
    """Render, re-render for determinism, and check the trainer's schema.

    The statistics are computed on the PACKED crop — after `crop_to_array`, the
    same transform the corpus cache and the app's crop rule both feed the model
    with — so they are comparable with the corpus's and the app's numbers.
    """
    runs = render_pages(profiles, pages_per_profile, seed)
    again = render_pages(profiles, pages_per_profile, seed)
    rows, bad = [], []
    for (profile, pid, crops, nlines, secs), (_, _, crops2, _, _) in zip(runs, again):
        same = (len(crops) == len(crops2)
                and all(a.tobytes() == b.tobytes() for a, b in zip(crops, crops2)))
        packed = [B.crop_to_array(Image.fromarray(np.asarray(c, np.uint8)))
                  for c in crops]
        shapes = [(p.shape[0], p.shape[1],
                   p.shape[0] == B.CROP_H and B.MIN_W <= p.shape[1] <= B.MAX_W)
                  for p in packed]
        ok = bool(crops) and all(s[2] for s in shapes) and same
        st = stats_of(packed) if packed else {}
        rows.append({"profile": profile, "page": pid, "lines": nlines,
                     "crops": len(crops), "seconds": round(secs, 2),
                     "content_size": [c.shape for c in crops],
                     "packed": [[s[0], s[1]] for s in shapes],
                     "deterministic": same, "ok": ok,
                     "stats": {k: round(st.get(k + "_median", float("nan")), 4)
                               for k in STATS} if crops else {}})
        if not ok:
            bad.append(profile + "/" + pid + (" (determinism)" if not same
                                              else " (empty or schema)"))
    return rows, bad


def render_side_gaps(rows, cs):
    """{profile: (nearest corpus profile, distance, per-stat xIQR gaps)} from a
    sampled render — for the profiles the app's own crop dump does not cover.

    CPU-frugal on purpose: the render was already done for determinism/schema, so
    a profile with no app measurement still gets a numeric 'how far is it from the
    nearest corpus profile' instead of a blank cell.
    """
    by = {}
    for r in rows:
        by.setdefault(r["profile"], []).append(r)
    out = {}
    for p, rs in by.items():
        # re-render is not needed: the row carries the medians of its packed crops
        q = {"ink_mean": float(np.mean([r["stats"]["ink"] for r in rs])),
             "contrast_mean": float(np.mean([r["stats"]["contrast"] for r in rs])),
             "sharp_mean": float(np.mean([r["stats"]["sharp"] for r in rs])),
             "ink_median": float(np.median([r["stats"]["ink"] for r in rs])),
             "contrast_median": float(np.median([r["stats"]["contrast"] for r in rs])),
             "sharp_median": float(np.median([r["stats"]["sharp"] for r in rs]))}
        out[p] = nearest_and_gaps(q, cs)
    return out


# ── the plan for the next full build ─────────────────────────────────────────

def build_plan(profiles, pages_per_doc, crops_per_page, secs_per_page):
    """Expected pages/crops/CPU for the next build, from the measured rate."""
    pages = pages_per_doc * len(C.DOCS)
    crops = int(round(pages * crops_per_page))
    return {"profiles": list(profiles), "pages_per_doc": pages_per_doc,
            "pages": pages, "crops": crops,
            "crops_per_profile": int(round(crops / max(1, len(profiles)))),
            "seconds": pages * secs_per_page,
            "minutes": pages * secs_per_page / 60.0,
            "secs_per_page": secs_per_page}


def merge_commands(profiles, pages_per_doc, corpus_dir="data/render_corpus"):
    """The ordered command list for the next build. Nothing here runs it."""
    plist = ",".join(profiles)
    return [
        "# 0. WAIT for tmux `retrain3` to exit: it reads data/render_corpus/crops_train.npz"
        " and holds every core; render and train must not contend.",
        "tmux list-sessions | grep retrain3   # must be empty before step 1",
        "# 1. keep the corpus the running experiment trained on (it is deterministic,"
        " but attribution is cheap to buy)",
        "cp data/render_corpus/crops_train.npz data/render_corpus/crops_train_4profiles.npz",
        "cp models/ocr/crnn.pt models/ocr/crnn_before_coverage.pt",
        "# 2. rebuild the corpus with EVERY profile the app can serve (was PROFILES[:4])",
        "bash run.sh render-corpus --pages %d --seed 0 --profiles %s"
        % (pages_per_doc, plist),
        "#    -> writes %s/{crops.npz,crops_train.npz,index.json}" % corpus_dir,
        "# 3. merge SROIE + the rendered corpus + the real-photo cache and train",
        "bash run.sh py ml/src/train_ocr.py --data artifacts/ocr_train.npz,"
        "%s/crops_train.npz,data/real_corpus/crops_rectified.npz --epochs 30 --threads 4"
        % corpus_dir,
        "# 4. re-measure, in this order",
        "bash run.sh eval-render --split test",
        "bash run.sh eval-ocr",
        "bash run.sh py ml/src/eval_field_truth.py --split test --limit-pages 120",
        "# 5. re-measure the coverage itself (per-profile table, no render)",
        "bash run.sh py ml/src/profile_coverage.py --json /tmp/pc_after.json",
        "bash run.sh py ml/src/dist_compare.py --npz %s/crops_train.npz" % corpus_dir,
    ]


# ── report ──────────────────────────────────────────────────────────────────

def f(v, n=2):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"


def coverage_table(cs, as_, full=None):
    """Per app-servable profile: in corpus? nearest corpus profile, statistics gap."""
    rows = {}
    for p in full or app_servable_profiles():
        row = {"in_corpus": p in cs, "app_crops": 0, "nearest": None,
               "nearest_distance": None, "gaps": {}, "gate": ""}
        if as_ and p in as_:
            q = as_[p]
            row["app_crops"] = q["n"]
            near, dist, gaps = nearest_and_gaps(q, cs)
            row.update(nearest=near, nearest_distance=dist, gaps=gaps)
            row["gate"] = max(gaps, key=lambda k: gaps[k]["xIQR"])
        rows[p] = row
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--npz", default=str(SHIPPED_CACHE),
                    help="the corpus to check coverage against (default: the cache "
                         "the running retrain is built on)")
    ap.add_argument("--app", default=str(APP_DUMP))
    ap.add_argument("--profiles", default="",
                    help="proposed profile list (default: every profile the app can "
                         "serve); 'bench' = the app's bench fixture profiles")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--sample-pages", type=int, default=0,
                    help="pages PER newly-added profile to render as validation (0 = skip)")
    ap.add_argument("--pages", type=int, default=500, help="pages per document type")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    cs = corpus_stats(a.npz, a.stride)
    as_ = app_stats(a.app)
    props = resolve_profiles(a.profiles)

    print("=" * 78)
    print("CORPUS  " + a.npz)
    print("  profiles present: " + ", ".join("%s(%d)" % (p, cs[p]["n"])
                                             for p in sorted(cs)))
    print("  the app can be served: " + ", ".join(app_servable_profiles()))
    print("  the app's bench fixtures cover: " + ", ".join(app_bench_profiles()))
    miss = [p for p in app_servable_profiles() if p not in cs]
    print("  PROFILES MISSING FROM THE CORPUS: " + (", ".join(miss) or "none"))

    new = [p for p in props if p not in cs]
    rows_sample, bad, secs_per_page, crops_per_page, rs = [], [], 0.0, 0.0, {}
    if a.sample_pages and new:
        rows_sample, bad = validate_sample(new, a.sample_pages)
        secs_per_page = float(np.mean([r["seconds"] for r in rows_sample]))
        crops_per_page = float(np.mean([r["crops"] for r in rows_sample]))
        rs = render_side_gaps(rows_sample, cs)

    print("\n" + "=" * 78)
    print("COVERAGE — per profile the app can serve")
    print("  'gap' = |app - corpus| / corpus IQR (dist_compare.eff units); "
          "for profiles the app's dump\ndoes not cover, the same gap is computed "
          "on the sampled render (marked R).")
    print("  " + "profile".ljust(16) + "in corpus".rjust(10) + "app crops".rjust(10)
          + "   nearest corpus profile (std dist)".rjust(34)
          + "   gap x corpus IQR (ink / contrast / sharp)".rjust(46))
    rows = coverage_table(cs, as_)
    for p in app_servable_profiles():
        r = rows[p]
        near, gap = "n/a (nothing measured for this profile)", "n/a"
        if r["nearest"]:
            near = "%s (%.2f)" % (r["nearest"], r["nearest_distance"])
            gap = " / ".join("%s %.1fx" % (k, r["gaps"][k]["xIQR"]) for k in STATS)
        elif p in rs:
            nr, dist, gaps = rs[p]
            near = "%s (%.2f)  R" % (nr, dist)
            gap = " / ".join("%s %.1fx" % (k, gaps[k]["xIQR"]) for k in STATS)
            r["render_side"] = {"nearest": nr, "distance": dist, "gaps": gaps}
        print("  " + p.ljust(16) + ("yes" if r["in_corpus"] else "NO").rjust(10)
              + str(r["app_crops"]).rjust(10) + "   " + near.rjust(32) + "   " + gap)
    if as_ is None:
        print("  (no app dump at " + a.app + " — run: node app/tools/crop_distribution.js"
              " --json=" + a.app + ")")
    else:
        n_app = sum(v["n"] for k, v in as_.items() if k != "__dump__")
        n_missing = sum(v["n"] for k, v in as_.items() if k != "__dump__" and k not in cs)
        print("\n  app crop population: " + str(n_app) + " crops, "
              + str(n_missing) + " (" + f(100.0 * n_missing / max(1, n_app), 1)
              + "%) from profiles the corpus does not contain")

    print("\n  corpus per-profile statistics (medians; the reference for the gaps):")
    print("    " + "profile".ljust(16) + "crops".rjust(6) + "width med".rjust(11)
          + "ink".rjust(9) + "contrast".rjust(10) + "sharp".rjust(10))
    for p in sorted(cs):
        v = cs[p]
        print("    " + p.ljust(16) + str(v["n"]).rjust(6)
              + f(v["width_median"], 0).rjust(11) + f(v["ink_median"], 3).rjust(9)
              + f(v["contrast_median"], 1).rjust(10) + f(v["sharp_median"], 1).rjust(10))

    # ── the proposed list, and the sampled validation of it ────────────────
    print("\n" + "=" * 78)
    print("PROPOSED PROFILE LIST (" + str(len(props)) + " profiles, "
          + str(len(new)) + " not in the corpus): " + ",".join(props))

    if a.sample_pages and new:
        crops = [r["crops"] for r in rows_sample]
        print("\n  SAMPLED RENDER — " + str(len(rows_sample)) + " pages, "
              + str(a.sample_pages) + " per newly-added profile (seed convention:"
              " render_corpus.page_seed / zlib.crc32)")
        print("    " + "profile".ljust(16) + "page".ljust(18) + "lines".rjust(5)
              + "crops".rjust(6) + "secs".rjust(7) + "  ink/contrast/sharp"
              + "   schema (h x w, in bounds)   deterministic")
        for r in rows_sample:
            pk = r["packed"]
            inb = all(h == B.CROP_H and B.MIN_W <= w <= B.MAX_W for h, w in pk)
            print("    " + r["profile"].ljust(16) + r["page"].ljust(18)
                  + str(r["lines"]).rjust(5) + str(r["crops"]).rjust(6)
                  + f(r["seconds"], 2).rjust(7) + "  "
                  + (f(r["stats"]["ink"], 3) + "/" + f(r["stats"]["contrast"], 2)
                     + "/" + f(r["stats"]["sharp"], 1)).ljust(20)
                  + ("  %dx%d  %s" % (pk[0][0], pk[0][1], "in bounds" if inb else "OUT OF BOUNDS")).ljust(30)
                  + "   " + ("yes" if r["deterministic"] else "NO"))
        print("    -> rendered %d pages, %d crops, mean %.2fs/page, %.1f crops/page"
              % (len(rows_sample), sum(crops), secs_per_page, crops_per_page))
        print("    -> (a) renders at all: " + ("OK" if not bad else "FAIL " + "; ".join(bad))
              + "   (b) trainer schema after crop_to_array: "
              + ("OK" if all(r["ok"] for r in rows_sample) else "FAIL")
              + "   (c) deterministic under the same seed: "
              + ("OK" if all(r["deterministic"] for r in rows_sample) else "FAIL"))
        # (d) sensible relative to the app's fixtures for the same profile
        if as_:
            by_prof = {}
            for r in rows_sample:
                by_prof.setdefault(r["profile"], []).append(r["stats"])
            print("    -> (d) sampled render vs the app's own crops for the same profile"
                  " (app dump only covers the bench profiles):")
            for p, sts in by_prof.items():
                q = as_.get(p)
                if not q:
                    print("      " + p.ljust(16) + "render ink %.3f / contrast %.1f /"
                          " sharp %.0f   (no app fixture for this profile:"
                          " render_samples.py covers only the bench set)"
                          % (float(np.median([s["ink"] for s in sts])),
                             float(np.median([s["contrast"] for s in sts])),
                             float(np.median([s["sharp"] for s in sts]))))
                    continue
                print("      " + p.ljust(16) + "render ink %.3f vs app %.3f | contrast"
                      " %.1f vs %.1f | sharp %.0f vs %.0f"
                      % (float(np.median([s["ink"] for s in sts])), q["ink_median"],
                         float(np.median([s["contrast"] for s in sts])), q["contrast_median"],
                         float(np.median([s["sharp"] for s in sts])), q["sharp_median"]))

    # ── the plan ───────────────────────────────────────────────────────────
    if not secs_per_page:
        note = ("CPU cost unknown: re-run with --sample-pages 3 to measure the"
                " per-page rate under the current load.")
    else:
        note = None
    if crops_per_page <= 0:
        crops_per_page = 15.67        # measured: 23500 crops / 1500 pages, 4 profiles
        note = (note + " " if note else "") + \
            "(crops/page from the 4-profile corpus: 23500/1500)"
    plan = build_plan(props, a.pages, crops_per_page, secs_per_page)
    print("\n" + "=" * 78)
    print("NEXT BUILD — ordered commands (nothing here is run by this tool)")
    for line in merge_commands(props, a.pages):
        print("  " + line)
    print("\n  expected: %d pages (%d per document type x %d doc types), ~%d crops,"
          " ~%d crops per profile"
          % (plan["pages"], plan["pages_per_doc"], len(C.DOCS), plan["crops"],
             plan["crops_per_profile"]))
    if secs_per_page:
        print("  measured rate: %.2fs/page under the current load -> ~%.1f min for the"
              " render alone; the merge/train step then adds the crops to the existing"
              " SROIE set" % (secs_per_page, plan["minutes"]))
        alt = build_plan(props, 1125, crops_per_page, secs_per_page)
        print("  where the crops come from: %d pages x %.1f crops/page = %d crops, i.e."
              " ~%d per profile at the same per-page density as today's corpus"
              % (plan["pages"], crops_per_page, plan["crops"], plan["crops_per_profile"]))
        print("  merged training set: %d SROIE + %d rendered + 672 rectified = ~%d crops,"
              " ~the size retrain3 measured at 1796.6 s/epoch (30 epochs ~ 15 h) — so the"
              " default build changes the COMPOSITION, not the budget"
              % (52554, plan["crops"], 52554 + plan["crops"] + 672))
        print("  if per-profile parity is wanted instead (was ~5875 crops/profile):"
              " --pages 1125 -> %d pages, ~%d crops, ~%.1f min render, and the merged"
              " set grows ~%d crops (~+%.0f%% per-epoch cost)"
              % (alt["pages"], alt["crops"], alt["minutes"],
                 alt["crops"] - plan["crops"],
                 100.0 * (alt["crops"] - plan["crops"]) / max(1, 52554 + plan["crops"] + 672)))
    if note:
        print("  " + note.strip())

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"npz": a.npz, "coverage": rows, "corpus_stats": cs,
             "app_dump": None if as_ is None else {k: v for k, v in as_.items()
                                                   if k != "__dump__"},
             "proposed_profiles": props, "missing_from_corpus": new,
             "sample": rows_sample, "plan": plan}, indent=1) + "\n")
        print("\n  wrote " + a.json)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
