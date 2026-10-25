"""
fetch_dose_labels.py — REAL medication dose-panel text, at volume, from openFDA.

    bash run.sh py ml/tools/fetch_dose_labels.py --n 1200
    bash run.sh py ml/tools/fetch_dose_labels.py --report

WHY REAL TEXT MATTERS HERE
--------------------------
The renderer can draw any string, so a corpus of invented lines teaches the model
our own idea of what a dose panel says. It has no idea that real panels print
"DOSAGE Adults- Take 4 or 6 Pellets by mouth, three times daily" or that the
ingredient line leads with "ACTIVE INGREDIENT". openFDA returns the STRUCTURED
label sections of real FDA-approved products — active_ingredient,
dosage_and_administration, purpose, indications_and_usage, warnings, directions,
storage_and_handling, keep_out_of_reach_of_children — which is exactly the text a
doses tab contains, in the wording and punctuation the world actually uses.

Public domain (US government work), no key, no auth. The API is rate-limited, so
this pages politely and keeps a local cache; re-running costs nothing.

Output `data/dose_labels.jsonl`: one product per line with its sections, and
`data/dose_panels.txt`: the flattened panel lines, which is what the bottle
renderer lays out.
"""

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "dose_labels.jsonl"
PANELS = ROOT / "data" / "dose_panels.txt"

API = "https://api.fda.gov/drug/label.json"
UA = {"User-Agent": "sightline-research/1.0 (offline OCR corpus; https://github.com/adeviwon/SightLine)"}

# The sections a DOSES TAB actually carries. Order matters: it is the order a
# patient reads, and the renderer lays lines out in this order.
SECTIONS = [
    ("active_ingredient", "ACTIVE INGREDIENT"),
    ("purpose", "PURPOSE"),
    ("dosage_and_administration", "DOSAGE AND ADMINISTRATION"),
    ("directions", "DIRECTIONS"),
    ("indications_and_usage", "INDICATIONS AND USAGE"),
    ("warnings", "WARNINGS"),
    ("storage_and_handling", "STORAGE AND HANDLING"),
    ("keep_out_of_reach_of_children", "KEEP OUT OF REACH OF CHILDREN"),
    ("inactive_ingredient", "INACTIVE INGREDIENTS"),
]

QUERY = ('openfda.product_type:"HUMAN OTC DRUG" OR '
         'openfda.product_type:"HUMAN PRESCRIPTION DRUG"')


def fetch(skip, limit=100, tries=4):
    q = urllib.parse.urlencode({"search": QUERY, "limit": limit, "skip": skip})
    delay = 2.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(API + "?" + q, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:                     # openFDA's "no more results"
                return {"results": []}
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return {"results": []}


def panel_lines(sections):
    """The flattened panel: real label text, one printed line per entry."""
    out = []
    for key, heading in SECTIONS:
        v = sections.get(key)
        if not v:
            continue
        out.append(heading)
        text = v[0] if isinstance(v, list) else str(v)
        for raw in text.replace("\r", "\n").split("\n"):
            line = " ".join(raw.split())
            if len(line) >= 4:
                out.append(line)
            if len(out) >= 60:                    # a panel, not a whole leaflet
                break
        if len(out) >= 60:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="products to keep")
    ap.add_argument("--page", type=int, default=100)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    if a.report:
        if not OUT.exists():
            print("nothing yet at %s" % OUT)
            return 1
        rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
        lines = sum(len(r["panel"]) for r in rows)
        lens = sorted(len(l) for r in rows for l in r["panel"])
        print("%d real products, %d panel lines (%d unique)"
              % (len(rows), lines, len(set(l for r in rows for l in r["panel"]))))
        print("line length: min %d, median %d, p90 %d, max %d chars"
              % (lens[0], lens[len(lens)//2], lens[int(0.9*len(lens))], lens[-1]))
        refs = sorted({k for r in rows for k in r["sections"]})
        print("sections present: " + ", ".join(refs))
        return 0

    kept, skip, seen = [], 0, set()
    while len(kept) < a.n:
        d = fetch(skip, a.page)
        res = d.get("results") or []
        if not res:
            print("openFDA returned no more results at skip=%d" % skip)
            break
        for r in res:
            sid = r.get("set_id") or r.get("id")
            if not sid or sid in seen:
                continue
            panel = panel_lines(r)
            if len(panel) < 6:                    # not enough to be a doses tab
                continue
            seen.add(sid)
            kept.append({"set_id": sid,
                         "sections": {k: (r[k][0] if isinstance(r[k], list) else r[k])
                                      for k, _ in SECTIONS if k in r},
                         "panel": panel})
        skip += a.page
        print("  kept %d products (skip=%d)" % (len(kept), skip), flush=True)
        time.sleep(0.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(k) for k in kept))
    lines = [l for k in kept for l in k["panel"]]
    PANELS.write_text("\n".join(lines))
    print("wrote %s (%d products) and %s (%d panel lines, %d unique)"
          % (OUT.name, len(kept), PANELS.name, len(lines), len(set(lines))))
    assert all(len(k["panel"]) >= 6 for k in kept)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
