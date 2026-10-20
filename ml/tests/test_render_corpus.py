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
