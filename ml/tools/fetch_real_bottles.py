"""
fetch_real_bottles.py — acquire REAL-WORLD photographs of medicine bottles and
packaging, with their licence metadata, as the raw material for a doses-tab model.

    bash run.sh py ml/tools/fetch_real_bottles.py --limit 220
    bash run.sh py ml/tools/fetch_real_bottles.py --report      # what is on disk

WHY COMMONS, AND WHY THIS IS NOT A SHORTCUT AROUND REAL DATA
------------------------------------------------------------
The product reads dosage panels off curved, glossy medicine packaging. Every
number this repo has published for the doses path was measured on OUR OWN
RENDERER (1,500 synthetic pages, labels = the drawn string), and the honest
domain gap is already recorded: photographed receipts score ~25% exact against
~70% on renders. There is no public corpus of medication labels with text
transcriptions (checked: HuggingFace's "medicine bottle" datasets are robot-arm
manipulation sets; the prescription sets are text-only), so the real photos have
to be assembled.

Wikimedia Commons is the one large pool of genuinely REAL photographs that is
licensed for reuse, and its API returns the licence and author with each file —
so attribution travels with the data instead of being reconstructed later. This
script only ACQUIRES; transcription (the ground truth) is a separate, human-checked
step, and every downstream number must name which of the two it was measured on.

SAFETY NOTE FOR ANYONE CHANGING THIS: photos chosen for a TRAINING set are fine
as search results; a photo chosen for a published number is not, because the
transcript is read off the pixels by a person and a wrong transcript silently
caps the metric (§17/§18 of sightline-ml-validation-gates).
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "real_bottles"
RAW = OUT / "raw"
MANIFEST = OUT / "manifest.csv"

API = "https://commons.wikimedia.org/w/api.php"
# A contact URL is not decoration: without one the API answers 403 "Too Many Reqs"
# to a burst from a shared IP, which reads as "Commons has no data" if you believe
# the first error. Wikimedia's UA policy asks for a real identifier.
UA = "sightline-research/1.0 (offline OCR corpus; https://github.com/adeviwon/SightLine)"
# Politeness between API calls, and retries that back off — the first run of this
# script lost ALL ELEVEN sources to a 403 burst and reported 0 candidates.
API_SLEEP = 1.2
RETRIES = 4

# Categories and searches that actually contain PHOTOGRAPHS OF LABELLED PACKAGING.
# A category name is not evidence of content — the counts (and the triage step)
# are what decide whether a source is worth keeping.
SOURCES = [
    ("cat", "Medicine bottles"),
    ("cat", "Blister packs"),
    ("cat", "Pharmaceutical tablets"),
    ("cat", "Pill bottles"),
    ("cat", "Medicine boxes"),
    ("cat", "Drug packaging"),
    ("cat", "Pharmacy shelves"),
    ("search", 'filetype:bitmap "medicine bottle" label'),
    ("search", 'filetype:bitmap "prescription label"'),
    ("search", 'filetype:bitmap "medicine box" dosage'),
    ("search", 'filetype:bitmap "tablets" package instructions'),
]

# Licences that make reuse safe here. Anything unclear is DROPPED and counted,
# not smuggled in — and the manifest keeps the reason.
BAD_LICENCE = re.compile(r"\b(nc|nd|non-?commercial|no-?deriv|fair use)\b", re.I)


def api(params):
    params = dict(params, format="json", formatversion=2)
    url = API + "?" + urllib.parse.urlencode(params)
    delay = 2.0
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if attempt == RETRIES - 1:
                raise
            if e.code in (403, 429, 503):
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return {}


def candidates(limit_per_source=60):
    """(title, url, licence, artist, w, h, source) for every usable file."""
    out, seen = [], set()
    for kind, name in SOURCES:
        try:
            if kind == "cat":
                data = api({"action": "query", "generator": "categorymembers",
                            "gcmtitle": "Category:" + name.replace(" ", "_"),
                            "gcmtype": "file", "gcmlimit": limit_per_source,
                            "prop": "imageinfo",
                            "iiprop": "url|size|mime|extmetadata"})
            else:
                data = api({"action": "query", "generator": "search",
                            "gsrsearch": name, "gsrnamespace": 6,
                            "gsrlimit": limit_per_source, "prop": "imageinfo",
                            "iiprop": "url|size|mime|extmetadata"})
        except Exception as e:                                    # pragma: no cover
            print("  source failed (%s / %s): %s" % (kind, name, e), file=sys.stderr)
            continue
        time.sleep(API_SLEEP)
        pages = (data.get("query") or {}).get("pages") or []
        kept = 0
        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            title, url = p.get("title", ""), ii.get("url", "")
            if not url or title in seen:
                continue
            meta = ii.get("extmetadata") or {}
            lic = (meta.get("LicenseShortName") or {}).get("value", "") or ""
            if BAD_LICENCE.search(lic) or not lic:
                continue
            if ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if (ii.get("width") or 0) < 700:            # too small to read a panel
                continue
            seen.add(title)
            out.append({"title": title, "url": url, "licence": lic,
                        "artist": (meta.get("Artist") or {}).get("value", "")[:120],
                        "w": ii.get("width"), "h": ii.get("height"),
                        "source": ("%s:%s" % (kind, name))})
            kept += 1
        print("  %-45s %3d usable" % (name[:45], kept), flush=True)
    return out


def download(items, limit):
    RAW.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for it in items:
        if n >= limit:
            break
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", it["title"].split(":", 1)[-1])[:80]
        dest = RAW / name
        if not dest.exists():
            try:
                req = urllib.request.Request(it["url"], headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as r:
                    blob = r.read()
                dest.write_bytes(blob)
                time.sleep(0.2)                       # be a decent API citizen
            except Exception as e:
                print("  skip %s (%s)" % (name[:40], e), file=sys.stderr)
                continue
        it = dict(it, file=name, bytes=dest.stat().st_size)
        rows.append(it)
        n += 1
        if n % 25 == 0:
            print("  downloaded %d" % n, flush=True)
    with MANIFEST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "title", "url", "licence",
                                           "artist", "w", "h", "source", "bytes"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    return rows


def report():
    if not MANIFEST.exists():
        print("no manifest at %s — run the fetcher first" % MANIFEST)
        return 1
    rows = list(csv.DictReader(MANIFEST.open()))
    print("%d photographs, %.1f MB" % (len(rows),
                                       sum(int(r["bytes"] or 0) for r in rows) / 1e6))
    by_lic, by_src = {}, {}
    for r in rows:
        by_lic[r["licence"]] = by_lic.get(r["licence"], 0) + 1
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print("licences: " + ", ".join("%s=%d" % kv for kv in sorted(by_lic.items(),
                                                                key=lambda k: -k[1])))
    print("sources:  " + ", ".join("%s=%d" % kv for kv in sorted(by_src.items(),
                                                                key=lambda k: -k[1])[:6]))
    print("manifest: %s (attribution travels with the data)" % MANIFEST)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=220)
    ap.add_argument("--per-source", type=int, default=60)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    print("querying Commons sources ...")
    items = candidates(a.per_source)
    print("candidates: %d (licence-clean, big enough to read a panel)" % len(items))
    rows = download(items, a.limit)
    print("downloaded %d to %s" % (len(rows), RAW))
    assert len(rows) == len(set(r["file"] for r in rows)), "duplicate filenames in the manifest"
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
