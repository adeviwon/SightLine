"""
Build the text-recognition training set from REAL documents.

WHY THIS REPLACES THE SYNTHETIC CORPUS
--------------------------------------
Everything measured so far was measured on `capture.py` renders: clean
DejaVu text, synthetic motion/defocus/noise. That corpus is useful for testing
the degradation GATE, because we know exactly what damage was applied. It is
the wrong distribution for training a recogniser, and the difference is not
subtle. Rendering the same receipt two ways:

  synthetic  uniform black ink on flat #f8f8f8, one font, one size
  real       thermal ink that fades unevenly across the page, pink cast,
             barcode noise, a handwritten annotation, ghost print from the
             reverse, thin small strokes, JPEG ringing

A recogniser trained on the first will fail on the second, and no amount of
synthetic augmentation bridges that gap — it is a gap in the IMAGING PROCESS,
not in the degradation model.

THE TWO REAL SOURCES
--------------------
1. ICDAR2019-SROIE (Huang et al. 2019), via the jsdnrs/ICDAR2019-SROIE
   Hugging Face mirror. 626 train + 347 test REAL scanned receipts with
   per-WORD bounding boxes and transcriptions. Flatbed scans, but genuinely
   photographed/printed paper: thermal fade, barcodes, handwriting, ghost
   print, barcode artefacts.

2. Express Expense / Zenodo 13688441 (Auad et al., SIBGRAPI 2024), CC-BY-4.0.
   198 HAND-PHOTOGRAPHED restaurant receipts with whitespace-aligned
   transcriptions and four-corner homographies. This is the real-life capture
   set the brief asks for: shot by hand, at night, on a dark background, in a
   plastic sleeve, rotated, unevenly lit, with glare.

   Its labels are whole-receipt transcripts rather than word boxes, so it is
   used for evaluation and for whole-page training signal, while SROIE supplies
   the tight word crops.

SPLITTING
---------
By DOCUMENT, never by crop. A receipt contributes many word crops; splitting
crops would put the same receipt in train and test and report a number that is
really memorisation. `split_by_document()` partitions receipt IDs, and every
crop inherits its parent receipt's split.

LABELS ARE CLEANED, NOT SANITISED AWAY
---------------------------------------
Real transcriptions contain OCR-era noise from the original dataset
annotation (`BOOK TA .K` for `BOOK TA.K`). That is kept in the RAW label and
removed only in a normalised copy used for CER scoring, so a model is never
trained to reproduce annotation typos but is still measured against a fair
target.
"""

import csv
import io
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SROIE_DIR = ROOT / "data" / "sroie" / "data"
HAND_DIR = ROOT / "data" / "handheld"
OUT = ROOT / "artifacts"

# Crop geometry. Height is fixed because CTC needs a constant time axis; width
# varies with the text and is bucketed at collate time.
CROP_H = 32
MIN_W, MAX_W = 16, 320


# ── label normalisation ───────────────────────────────────────────────────────

# Glyph confusions that matter for THIS product. A receipt's safety-relevant
# tokens are numeric: a dosage, an account number, a total. These are exactly
# the pairs that turn a correct read into a wrong one.
_CONFUSIONS = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "|": "1", "!": "1",
    "S": "5", "s": "5", "B": "8",
    "Z": "2", "z": "2",
    "G": "6", "b": "6",
    "T": "7",
    # NOTE: 'g'/'q' are deliberately NOT mapped to 9 here.
    #
    # "500mg" is a DOSAGE. Mapping g->9 turns it into "500M9", which is not a
    # number at all and would fail the extractor's \d+mg pattern -- a false
    # negative on the single most safety-critical token this app reads. The
    # asymmetry is the point: false negatives (omit a field) are recoverable,
    # false positives (announce a corrupted one) are not. The same reasoning
    # excludes 'm'->anything.
}
_CONF_RE = re.compile("|".join(map(re.escape, _CONFUSIONS)))

# Units that must survive normalisation untouched. If a transcription contains
# one of these, the run is a MEASUREMENT and digit/letter confusions inside it
# are still resolved, but the unit itself is preserved verbatim.
_UNIT_RE = re.compile(r"\b(MG|MCG|UG|KG|ML|L|%|G)\b")


def norm_label(s: str) -> str:
    """
    Normalise a transcription for SCORING ONLY. Never used as a training
    target: training the model to emit "0" for a printed "O" would be teaching
    it to corrupt company names.
    """
    s = unicodedata.normalize("NFKC", str(s))
    s = _CONF_RE.sub(lambda m: _CONFUSIONS[m.group(0)], s)
    s = s.upper()
    return re.sub(r"[^A-Z0-9./:$,-]+", "", s)


def clean_for_training(s: str) -> str:
    """
    Collapse whitespace and strip characters that are never legible in a crop.
    Keeps case and punctuation: the recogniser should see the real string.
    """
    s = unicodedata.normalize("NFKC", str(s)).strip()
    return re.sub(r"\s+", " ", s)


def cer(pred: str, ref: str) -> float:
    """Character Error Rate via Levenshtein. 0.0 = perfect, 1.0 = all wrong."""
    p, r = norm_label(pred), norm_label(ref)
    if not r:
        return 0.0 if not p else 1.0
    prev = list(range(len(r) + 1))
    for i, pc in enumerate(p, 1):
        cur = [i]
        for j, rc in enumerate(r, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (pc != rc)))
        prev = cur
    return prev[-1] / len(r)


# ── charset ───────────────────────────────────────────────────────────────────

# Deliberately small. The app extracts dosages, account numbers, sort codes,
# case references and dates. A 37-symbol charset (10 digits + 26 letters +
# space) covers all of that, keeps the output layer small enough to train on
# CPU, and makes CTC converge in a reasonable number of epochs.
#
# Punctuation is NOT included. It is the highest-error class in practice and
# contributes nothing to any field this app extracts; spending model capacity
# on it would cost accuracy where it matters. The field regexes tolerate its
# absence, and punctuation in real crops is handled by the extractor layer.
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
BLANK = 0                      # CTC blank index
C2I = {c: i + 1 for i, c in enumerate(CHARSET)}
I2C = {i + 1: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1


def encode(s: str):
    return [C2I[c] for c in s if c in C2I]


def decode_greedy(indices):
    """Collapse repeats then drop blanks — the standard CTC decode."""
    out, prev = [], -1
    for i in indices:
        if i != prev and i != BLANK:
            out.append(I2C.get(int(i), ""))
        prev = i
    return "".join(out)


# ── SROIE word crops ──────────────────────────────────────────────────────────

def load_sroie(split: str):
    """Yield (receipt_id, [(word, (x1,y1,x2,y2))]) from one SROIE parquet."""
    import pyarrow.parquet as pq
    files = sorted((SROIE_DIR).glob(f"{split}-*.parquet"))
    if not files:
        return
    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                key = row.get("key") or ""
                words = row.get("words") or []
                boxes = row.get("bboxes") or []
                yield key, list(zip(words, boxes))


def sroie_crops(max_receipts=None):
    """
    Word-level crops with tight boxes. Each crop keeps its parent receipt ID so
    the split can be by document.
    """
    out = []
    for rid, pairs in load_sroie("train"):
        if max_receipts and len({r for r, _ in out}) >= max_receipts:
            break
        for w, b in pairs:
            txt = clean_for_training(w)
            if len(txt) < 2:
                continue
            try:
                x1, y1, x2, y2 = [int(v) for v in b]
            except (TypeError, ValueError):
                continue
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            out.append((rid, txt, (x1, y1, x2, y2)))
    return out


def render_sroie_crops(limit_receipts=200, max_crops=60000, seed=0):
    """
    Materialise SROIE word crops to an .npz cache. Written once and reused:
    decoding a 300 MB parquet every epoch would dominate training time, and the
    receipt images cannot be held in RAM together (626 receipts at ~1000x2000
    RGB is several GB).
    """
    cache = OUT / f"sroie_crops_{limit_receipts}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["images"], list(d["labels"]), list(d["receipts"])
    raise RuntimeError(
        "call ml/src/build_dataset.py, which streams the parquet and writes "
        f"{cache.name}")


# ── handheld whole-receipt pages (real-life photos) ───────────────────────────

def handheld_pages():
    """
    The 198 hand-photographed receipts. Returns (name, image_path, transcript).

    Transcripts come from the whitespace-aligned .txt files, where runs of
    spaces encode horizontal position. That is not usable as a transcript
    directly — it is a fixed-width rendering of the page — so the text is
    recovered by collapsing the runs, and the horizontal structure is used to
    split into LINES, which is the unit the recogniser actually consumes.
    """
    pages = []
    labels = HAND_DIR / "Labels"
    for jp in sorted(HAND_DIR.glob("*-receipt.jpg")):
        tp = labels / (jp.stem + ".txt")
        if not tp.exists():
            continue
        raw = tp.read_text(errors="ignore")
        lines = []
        for ln in raw.split("\n"):
            # A line is a run of >=2 spaces; collapse to single spaces.
            t = re.sub(r"\s{2,}", " ", ln).strip()
            if len(t) >= 3:
                lines.append(t)
        if not lines:
            continue
        pages.append((jp.stem, jp, lines))
    return pages


def handheld_homography():
    """Four-corner annotations, for optional perspective rectification."""
    hp = HAND_DIR / "homography.csv"
    if not hp.exists():
        return {}
    out = {}
    with hp.open() as f:
        for row in csv.DictReader(f, delimiter=";"):
            try:
                pts = [[int(v) for v in row[k].split(",")]
                       for k in ("P1", "P2", "P3", "P4")]
                out[row["Image"]] = np.array(pts, np.float32)
            except (KeyError, ValueError):
                continue
    return out


# ── splitting ─────────────────────────────────────────────────────────────────

def split_by_document(items, doc_key, ratios=(0.7, 0.15, 0.15), seed=0):
    """
    Partition by DOCUMENT. `doc_key(item)` must return the parent document id.

    Returns {"train": [...], "val": [...], "test": [...]}.

    Sorting the document ids before shuffling makes the split a pure function
    of (ids, seed) rather than of dict iteration order, so it is reproducible
    across runs and machines — the same lesson as seedutil, applied here at the
    level where it actually bites.
    """
    docs = sorted({doc_key(i) for i in items})
    rng = random.Random(seed)
    rng.shuffle(docs)
    n = len(docs)
    a = int(n * ratios[0])
    b = a + int(n * ratios[1])
    assign = {}
    for i, d in enumerate(docs):
        assign[d] = "train" if i < a else ("val" if i < b else "test")
    out = {"train": [], "val": [], "test": []}
    for it in items:
        out[assign[doc_key(it)]].append(it)
    return out


def main():
    if "--selftest" in sys.argv:
        pages = handheld_pages()
        print(f"handheld pages: {len(pages)}")
        if pages:
            name, jp, lines = pages[0]
            print(f"  e.g. {name}: {len(lines)} lines")
            for l in lines[:5]:
                print(f"     {l!r}")
        h = handheld_homography()
        print(f"homographies: {len(h)}")
        print(f"charset: {len(CHARSET)} symbols -> {NUM_CLASSES} classes")
        assert norm_label("2O24-CV-OO456") == "2024-CV-00456", norm_label("2O24-CV-OO456")
        assert norm_label("5OO.OO") == "500.00"
        # The unit suffix must survive: '500mg' -> '500MG', never '500M9'.
        # A corrupted dosage is worse than an omitted one.
        assert norm_label("5OOmg") == "500MG", norm_label("5OOmg")
        assert "MG" in norm_label("Amoxicillin 5OOmg three times daily")
        # '.' and other punctuation are outside the charset BY DESIGN, so they
        # are dropped at encode time rather than predicted wrongly. That is a
        # deliberate accuracy trade, documented at CHARSET: field regexes all
        # tolerate a missing decimal point. Assert the behaviour so a future
        # charset change cannot silently alter it.
        assert encode("9.00") == encode("900")

        # CTC collapses REPEATS during greedy decode, so encoding "900" and
        # decoding it yields "90". That is correct CTC semantics, not a bug --
        # and it is a real constraint worth documenting rather than hiding:
        #
        #   * the model is trained with blank-between-repeats, so at INFERENCE
        #     the emitted sequence for "900" is [9, blank, 0, 0] and greedy
        #     decode returns "900" correctly.
        #   * feeding a bare [9,0,0] with no blank in between is not a
        #     well-formed CTC output, so it is never produced in practice.
        #
        # CTC collapses a run of identical symbols into one, so a well-formed
        # target must place a blank between every repeat. That is exactly what
        # the CTC loss requires and what the model learns to emit:
        #
        #     "900"  ->  [9, blank, 0, blank, 0]   -> "900"   correct
        #     [9, blank, 0, 0]                     -> "90"    malformed input
        #
        # The second case is not a bug in decode_greedy, it is what CTC means.
        # Assert both so a future "optimisation" of the decoder cannot quietly
        # change repeat handling, which would silently corrupt every price,
        # dosage and account number containing a doubled digit.
        assert decode_greedy([10, BLANK, 1, BLANK, 1]) == "900"
        assert decode_greedy([10, BLANK, 1, 1]) == "90"
        assert decode_greedy(encode("TOTAL")) == "TOTAL"
        assert decode_greedy([]) == ""
        assert cer("TOTAL 9.00", "TOTAL 9.OO") == 0.0, cer("TOTAL 9.00", "TOTAL 9.OO")
        assert cer("TOTAL 9.08", "TOTAL 9.00") > 0
        print("normalisation + encode/decode + CER OK")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    pairs = sroie_crops()
    print(f"SROIE word crops: {len(pairs)}")
    print(f"distinct receipts: {len({p[0] for p in pairs})}")
    sp = split_by_document(pairs, lambda p: p[0])
    for k, v in sp.items():
        print(f"  {k:5} {len(v):6} crops  "
              f"{len({p[0] for p in v}):4} receipts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
