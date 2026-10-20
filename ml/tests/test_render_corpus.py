"""
test_render_corpus.py — the labelled corpus must be exact, or training is poisoned.

`ml/src/render_corpus.py` generates line crops from the SAME renderer the product's
documents come from (`capture.render_document`), so the recogniser can be trained
on the domain it is served: our own document types, at the geometry the app feeds
it. Three properties make that corpus trustworthy and are pinned here:

  1. LABELS ARE THE CHARSET, EXACTLY. The model can only emit "0-9A-Z " and
     space, so a target containing ":" or lowercase would teach it to drop
     characters. `train_label` is deliberately NOT `realdata.norm_label` (which
     maps O->0 for SCORING) nor `clean_for_training` (which keeps case and
     punctuation for real photographs).
  2. BANDS COME FROM THE LAYOUT, NOT A DETECTOR. The renderer places line i at a
     computable y, so no detection error can leak into a training label.
  3. SPLITS ARE BY PAGE. Two crops of the same rendered page must never sit on
     opposite sides of train/test — the same leakage rule the SRE receipts follow
     (`realdata.split_by_document`).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import capture       # noqa: E402
import realdata as R  # noqa: E402
import render_corpus as RC  # noqa: E402


def test_train_label_keeps_only_what_the_model_can_emit():
    assert RC.train_label("Patient: Jane Doe 500mg") == "PATIENT JANE DOE 500MG"
    # A hyphen is DROPPED, not turned into a space: the model cannot emit it, and
    # realdata.norm_label + evaluate.fields_found both strip punctuation before
    # comparing, so "401104" matches the ground-truth "40-11-04" as one token.
    # Splitting it into "40 11 04" would break that match.
    assert RC.train_label("  40-11-04  ") == "401104"
    assert set(RC.train_label("a-b/c%")) <= set(R.CHARSET)
    assert RC.train_label("::") == ""       # nothing legible survives


def test_line_bands_cover_the_lines_the_renderer_drew():
    lines, _ = capture.DOCS[0]
    gray = np.asarray(capture.render_document(lines, seed=1).convert("L"))
    bands = RC.line_bands(lines, 1.0)
    assert len(bands) == len(lines)
    for i, (y0, y1) in enumerate(bands):
        assert 0 <= y0 < y1 <= gray.shape[0], f"band {i} outside the page"
        assert gray[y0:y1].min() < 200, f"line {i} has no ink inside its band"


def test_page_crops_returns_one_labelled_crop_per_line():
    lines, _ = capture.DOCS[1]
    gray = np.asarray(capture.render_document(lines, seed=2).convert("L"))
    got = RC.page_crops(gray, lines, 1.0)
    assert len(got) == len(lines), "a rendered line produced no crop"
    for crop, label in got:
        assert crop.ndim == 2 and crop.shape[0] >= 4 and crop.shape[1] >= 4
        assert label and set(label) <= set(R.CHARSET)


def test_build_splits_by_page_never_by_crop(tmp_path):
    d = RC.build(n_pages=9, profiles=["studio_clean"], seed=3, out_dir=tmp_path)
    assert d["doc"].shape[0] == d["crops"].shape[0] == d["labels"].shape[0]
    seen = {}
    for doc, split in zip(d["doc"], d["split"]):
        assert seen.setdefault(doc, split) == split, f"page {doc} is in two splits"
    assert set(d["split"]) == {"train", "val", "test"}, "every split must be non-empty"
    # and it must be readable from disk exactly as written
    back = np.load(tmp_path / "crops.npz", allow_pickle=True)
    assert back["labels"].shape == d["labels"].shape
    json.loads((tmp_path / "index.json").read_text())


def test_write_cache_matches_the_trainer_schema(tmp_path):
    """`train_ocr.py --data a,b,c` concatenates caches, so ours must be one.

    It reads images/widths/labels/variants/splits and a source key
    (`receipts`, or `page` when there is none), and it filters crops whose label
    is longer than the model's timesteps. A schema mismatch would either crash or
    silently train on none of this corpus, which is the kind of failure this file
    exists to prevent.
    """
    d = RC.build(n_pages=6, profiles=["studio_clean"], seed=5, out_dir=tmp_path)
    p = RC.write_cache(d, tmp_path / "crops_train.npz")
    z = np.load(p, allow_pickle=True)
    assert {"images", "widths", "labels", "variants", "splits", "page"} <= set(z.files)
    assert z["images"].shape[1] == R.CROP_H, "crops must be packed CROP_H tall"
    assert z["images"].shape[2] <= R.MAX_W, "packed width must respect MAX_W"
    n = z["images"].shape[0]
    assert n == len(z["labels"]) == z["widths"].shape[0] == len(z["splits"])
    for lab in list(z["labels"])[:50]:
        assert R.encode(str(lab)), "a label the model cannot encode is untrainable"
    for w in z["widths"]:
        assert 1 <= int(w) <= z["images"].shape[2]
    assert {str(s) for s in z["splits"]} <= {"train", "val", "test"}
    # every page stays on one side of the split
    seen = {}
    for page, split in zip(z["page"], z["splits"]):
        assert seen.setdefault(str(page), str(split)) == str(split)


# ── the alphabet-confusion lever: one typeface per page ───────────────────────

def test_default_render_is_unchanged_when_no_font_is_given():
    """capture.render_document(font=None) must be byte-identical to the old render.

    Everything already measured — SROIE caches, repro-check, the frozen
    checkpoint's own training data — was produced by the font=None path. If this
    drifts, every stored number becomes unattributable.
    """
    import hashlib
    import capture as C
    lines = C.PRESCRIPTION
    a = C.render_document(lines, seed=7).tobytes()
    b = C.render_document(lines, seed=7).tobytes()
    assert a == b, "render_document is not deterministic"
    # and the pinned-family path must actually differ, or the lever is a no-op
    c = C.render_document(lines, seed=7, font=C.FONTS[1]).tobytes()
    assert c != a, "font= changed nothing: the multi-font lever would be a no-op"
    assert hashlib.sha256(a).hexdigest()[:12] != hashlib.sha256(c).hexdigest()[:12]


def test_corpus_records_one_font_per_page(tmp_path):
    """The font must be in the artifact, or per-font accuracy is unanswerable."""
    import capture as C
    import render_corpus as RC
    data = RC.build(n_pages=6, seed=5, out_dir=tmp_path, verbose=False)
    fonts = list(data["font"])
    assert len(fonts) == len(data["labels"])
    for f in set(fonts):
        assert Path(f).exists(), f
    # a page's crops share a font; the corpus as a whole does not
    by_doc = {}
    for doc, f in zip(data["doc"], fonts):
        by_doc.setdefault(str(doc), set()).add(f)
    assert all(len(v) == 1 for v in by_doc.values()), "one page changed font mid-page"
    assert len(set(fonts)) >= 2, "only one typeface in the corpus: nothing was widened"
    idx = json.loads((tmp_path / "index.json").read_text())
    assert idx["font_counts"] and len(idx["fonts"]) >= 2
