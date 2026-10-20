#!/usr/bin/env python3
"""
fetch_text_data.py — fetch a candidate real-text corpus, LICENCE FIRST.

The project's only real-text corpus today is ICDAR2019-SROIE (CC-BY-4.0), plus the
CC-BY-4.0 hand-photographed receipts from Zenodo. Licensing is a hard gate here, so
this script does the safe thing by default:

    * `--dry-run` is the DEFAULT. It prints the URL, expected size, published
      checksum and licence for one dataset, then exits 0 without touching the
      network.
    * `--yes` is the only way anything is downloaded. It streams to disk
      resumably (HTTP Range), verifies the size and the checksum, and exits
      non-zero on any mismatch.
    * A dataset whose licence could NOT be confirmed is refused outright (exit 2),
      dry-run included. Guessing a licence is how a demo ships something it cannot
      legally ship.

    bash run.sh py ml/src/fetch_text_data.py --dry-run --dataset mjsynth
    bash run.sh py ml/src/fetch_text_data.py --list
    bash run.sh py ml/src/fetch_text_data.py --yes --dataset sroie --out data/sroie

The table, the reasoning and the citations live in docs/17_OCR_ACCURACY_RESEARCH.md
(§2). Sizes and checksums here were read from each host's own metadata at the time of
writing; a size that drifts is a signal the host re-cut the release, so both size and
checksum are verified on download.
"""

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "text"

CHUNK = 1 << 20
UA = {"User-Agent": "sightline-fetch_text_data/1"}

# ── the table. licence=None means "could not confirm" -> refused. ─────────────
# `files` is the download list: (relative path, url, bytes, sha256|None, md5|None).
DATASETS = {
    "sroie": dict(
        name="ICDAR2019-SROIE (Huang et al., ICDAR 2019 Robust Reading Challenge)",
        url="https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE",
        licence="CC-BY-4.0 (HF mirror jsdnrs/ICDAR2019-SROIE; the original "
                "competition release carries its own terms)",
        licence_ok=True,
        contains="626 train + 361 test real scanned receipts, per-WORD boxes, "
                 "transcriptions and key entities (this is what we train on)",
        files=[
            ("train-00000.parquet",
             "https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE/resolve/"
             "refs%2Fconvert%2Fparquet/default/train/0000.parquet",
             318620215,
             "b18c16b4d8481e5e4537a1700e4616907fe4acd92d6362a7e430b0e866213887",
             None),
            ("test-00000.parquet",
             "https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE/resolve/"
             "refs%2Fconvert%2Fparquet/default/test/0000.parquet",
             191045976,
             "04f8f31b45944cc6e6459a7a95c851a721fc93ffec0a5c29ece9ded734a684c2",
             None),
        ],
    ),
    "handheld": dict(
        name="Express Expense — hand-photographed restaurant receipts "
             "(Auad et al., SIBGRAPI 2024)",
        url="https://zenodo.org/records/13688441",
        licence="CC-BY-4.0 (Zenodo record 13688441)",
        licence_ok=True,
        contains="198 hand-photographed receipts at night (plastic sleeve, glare, "
                 "tilt), whitespace-aligned page transcripts, 4-corner homographies. "
                 "Page transcripts only — no line boxes.",
        files=[
            ("Dataset.zip",
             "https://zenodo.org/api/records/13688441/files/Dataset.zip/content",
             18998248, None, "b7a29ba1cfab963714782e1752ce8e62"),
        ],
    ),
    "render_corpus": dict(
        name="in-repo renderer (ml/src/capture.py + ml/src/render_corpus.py)",
        url="(none — built locally)",
        licence="our own code, no external licence",
        licence_ok=True,
        contains="pages rendered by capture.render_document, 9 capture profiles, "
                 "labels drawn from the layout (no detector error can leak in)",
        files=[],   # generated, not fetched
    ),
    "textocr": dict(
        name="TextOCR (Singh et al., CVPR 2021)",
        url="https://textvqa.org/textocr/dataset/",
        licence="CC-BY-4.0",
        licence_ok=True,
        contains="714,770 train + 107,802 val word annotations on 24,902 OpenImages "
                 "photos; case-sensitive, punctuation included, illegible words "
                 "annotated as '.' (needs case-fold + punctuation strip for us)",
        files=[
            ("TextOCR_0.1_train.json",
             "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json",
             272000000, None, None),
            ("TextOCR_0.1_val.json",
             "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json",
             39000000, None, None),
        ],
    ),
    # ── licence-unknown: present so the refusal is explicit, not silent ───────
    "iiit5k": dict(
        name="IIIT 5K-Words (Mishra et al., BMVC 2012)",
        url="https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset",
        licence=None, licence_ok=False,
        contains="5,000 scene/born-digital word crops + 50-word and 0.5M-word lexicons",
        files=[("IIIT5K-Word_V3.0.tar.gz",
                "https://cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/"
                "IIIT5K-Word_V3.0.tar.gz", 111149056, None, None)],
    ),
    "iam": dict(
        name="IAM Handwriting Database 3.0",
        url="https://fki.tic.heia-fr.ch/databases/iam-handwriting-database",
        licence=None, licence_ok=False,
        contains="1,539 handwritten English pages / 115,320 word crops, 300 dpi PNG",
        files=[("iamdb.tgz", "https://fki.tic.heia-fr.ch/DBs/iamDB/data/",
                60000000000, None, None)],
    ),
    "svt": dict(
        name="Street View Text (Wang et al., 2011)",
        url="https://vision.ucsd.edu/index.php/datasets/street-view-text",
        licence=None, licence_ok=False,
        contains="349 Google Street View images, ~725 signage words",
        files=[],
    ),
    "cute80": dict(
        name="CUTE80 (Risnumawan et al., ESWA 2014)",
        url="https://github.com/mohtashim-nawaz/Cute80-Dataset",
        licence=None, licence_ok=False,
        contains="80 images / 288 curved scene-text word crops; the mirror states no "
                 "licence, only a citation request",
        files=[],
    ),
    "union14m": dict(
        name="Union14M (Jiang et al., ICCV 2023)",
        url="https://github.com/Mountchicken/Union14M",
        licence=None, licence_ok=False,
        contains="4M labelled + 10M unlabelled scene-text crops consolidated from 17 "
                 "public datasets whose licences differ per source",
        licence_note="code is MIT; per-crop provenance/licence is not traceable",
        files=[],
    ),
    "hiertext": dict(
        name="HierText (Long et al., CVPR 2022)",
        url="https://github.com/google-research-datasets/hiertext",
        licence="CC-BY-SA-4.0 annotations (share-alike), images from OpenImages",
        licence_ok=False,
        contains="11,639 images, ~1.2M word annotations plus line and paragraph levels",
        files=[],
    ),
    "mjsynth": dict(
        name="MJSynth / Synthetic Word Dataset (Jaderberg et al., 2014)",
        url="https://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz",
        licence="Non-commercial research only — Oxford VGG dataset Terms of Access "
                "(group 2): 12-month licence, artefacts must not be incorporated in "
                "commercial products",
        licence_ok=False,
        contains="9M synthetic scene-text word-box images covering 90k English words",
        files=[("mjsynth.tar.gz",
                "https://thor.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz",
                10 * 1024**3, None, None)],
    ),
    "synthtext": dict(
        name="SynthText in the Wild (Gupta et al., CVPR 2016)",
        url="https://www.robots.ox.ac.uk/~vgg/data/scenetext/",
        licence="Non-commercial research only — Oxford VGG dataset Terms of Access "
                "(group 2)",
        licence_ok=False,
        contains="800k images with ~8M synthetic word instances placed into scenes "
                 "(download links withdrawn from the VGG site; mirrors only)",
        files=[],
    ),
    "synthdog": dict(
        name="SynthDoG (Donut, Kim et al., ECCV 2022)",
        url="https://github.com/clovaai/donut",
        licence="MIT (generator — nothing to download)",
        licence_ok=True,
        contains="generates synthetic DOCUMENTS (forms, invoices) with a configurable "
                 "alphabet, including English; runs on CPU",
        files=[],
    ),
    "trdg": dict(
        name="TextRecognitionDataGenerator (TRDG)",
        url="https://github.com/Belval/TextRecognitionDataGenerator",
        licence="MIT (generator — pip install trdg)",
        licence_ok=True,
        contains="word/line crop generator with your own fonts, lexicon and distortion "
                 "stack",
        files=[],
    ),
}


def human(n):
    if n is None:
        return "(unknown)"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0


def describe(key):
    d = DATASETS[key]
    print("dataset : %s  (--dataset %s)" % (d["name"], key))
    print("source  : %s" % d["url"])
    print("licence : %s" % (d["licence"] or "UNCONFIRMED — could not be confirmed"))
    if d.get("licence_note"):
        print("          note: %s" % d["licence_note"])
    print("contains: %s" % d["contains"])
    if not d["files"]:
        print("size    : 0 B — nothing to download "
              "(generated locally, or no published URL)")
        print("sha256  : none published — nothing to verify (no download)")
    for name, url, size, sha256, md5 in d["files"]:
        print("files   : %s" % name)
        print("  url   : %s" % url)
        print("  size  : %s (%d bytes)" % (human(size), size or 0))
        print("  sha256: %s" % (sha256 or "none published by the host"))
        print("  md5   : %s" % (md5 or "none published by the host"))
    if not d["licence_ok"]:
        print("verdict : NOT USABLE for this product — see docs/17_OCR_ACCURACY_RESEARCH.md §2")
    return d


def download(url, dest, expected, sha256=None, md5=None):
    """Resumable stream to `dest`, then verify size and checksum."""
    have = dest.stat().st_size if dest.exists() else 0
    if expected and have > expected:
        print("  local file larger than expected (%d > %d) — restarting"
              % (have, expected))
        dest.unlink()
        have = 0
    if have and expected and have == expected:
        print("  already complete, skipping download")
    else:
        headers = dict(UA)
        if have:
            headers["Range"] = "bytes=%d-" % have
        print("  GET %s%s" % (url, " (resuming at %d B)" % have if have else ""))
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            if have and getattr(r, "status", 200) == 200:
                # host ignored our Range -> start over rather than append garbage
                print("  host ignored Range; restarting from 0")
                have = 0
            mode = "ab" if have else "wb"
            done = have
            with dest.open(mode) as f:
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    f.write(b)
                    done += len(b)
                    if expected and done % (64 * CHUNK) < CHUNK:
                        print("    %s / %s" % (human(done), human(expected)))

    size = dest.stat().st_size
    if expected and size != expected:
        print("  SIZE MISMATCH: got %d, expected %d" % (size, expected))
        return False
    if sha256 or md5:
        h256, h5 = hashlib.sha256(), hashlib.md5()
        with dest.open("rb") as f:
            for b in iter(lambda: f.read(CHUNK), b""):
                h256.update(b)
                h5.update(b)
        if sha256 and h256.hexdigest() != sha256:
            print("  SHA256 MISMATCH: got %s" % h256.hexdigest())
            return False
        if md5 and h5.hexdigest() != md5:
            print("  MD5 MISMATCH: got %s" % h5.hexdigest())
            return False
        print("  checksum ok (%s)" % ("sha256" if sha256 else "md5"))
    print("  ok: %s (%s)" % (dest, human(size)))
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Fetch a candidate corpus, licence first. Dry-run by default.")
    ap.add_argument("--dataset", default="", help="one of: " + ", ".join(sorted(DATASETS)))
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be fetched and exit 0 (the default)")
    ap.add_argument("--yes", action="store_true", help="actually download")
    ap.add_argument("--out", default="", help="output directory (default data/text/<dataset>)")
    ap.add_argument("--list", action="store_true", help="list the dataset keys")
    a = ap.parse_args()

    if a.list or not a.dataset:
        for k in sorted(DATASETS):
            d = DATASETS[k]
            print("%-14s %-4s %s" % (k, "ok" if d["licence_ok"] else "GATE",
                                     d["name"]))
        if not a.dataset:
            print("\nnothing fetched: pass --dataset <key> (dry-run unless --yes)")
        return 0

    key = a.dataset
    if key not in DATASETS:
        print("unknown dataset %r. known keys: %s"
              % (key, ", ".join(sorted(DATASETS))), file=sys.stderr)
        return 1

    d = describe(key)

    if d["licence"] is None:
        print("\nREFUSING: licence unconfirmed for %r. Confirm the terms first, or "
              "pick a licence-clean corpus (sroie, handheld, render_corpus, textocr, "
              "synthdog, trdg) — docs/17_OCR_ACCURACY_RESEARCH.md §2." % key,
              file=sys.stderr)
        return 2

    if not d["files"]:
        print("\nnothing to download for %r — it is generated locally." % key)
        return 0

    if not a.yes:
        print("\nDRY RUN (default): nothing downloaded. Add --yes to fetch.")
        return 0

    if not d["licence_ok"]:
        print("\nWARNING: %s forbids this data in a shipped product. Downloading "
              "anyway because --yes was passed — local experiments only." % key)

    out = Path(a.out) if a.out else DEFAULT_OUT / key
    out.mkdir(parents=True, exist_ok=True)
    for name, url, size, sha256, md5 in d["files"]:
        print("\n-> %s" % name)
        try:
            if not download(url, out / name, size, sha256, md5):
                return 3
        except KeyboardInterrupt:
            print("\ninterrupted — the partial file is kept; re-run the same command "
                  "to resume.")
            return 130
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            print("  download failed: %s" % e, file=sys.stderr)
            return 4
    print("\nfetched into %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
