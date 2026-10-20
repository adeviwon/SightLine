"""
Word segmentation of a detected line — Tesseract's first structural advantage.

Our recogniser reads ONE squeezed crop per line. Exact-match on an 18-character
line needs 98.8% character accuracy; on an 8-character word it needs only 97.2%
((1-CER)^L = 0.8). Cutting the line at the ink gaps BETWEEN words produces
shorter crops, so the same character error rate buys more whole tokens.

This is NOT the chunking experiment already measured here as a LOSS. That one
cut a 228x15 line into contiguous fixed-width chunks and scored 54% against 98%
for one squeezed crop (app/tools/crop_contract.js). This splits at whitespace
gaps and labels each crop with the individual word — a different experiment
whose failure mode is over/under-splitting, not a cut through a glyph.

Pure function over a uint8 array. No PIL, no I/O, deterministic.

MEASURED VERDICT: DO NOT WIRE THIS INTO THE PRODUCT (yet).
----------------------------------------------------------
A/B on 200 held-out line crops, frozen checkpoint, calibrated gap threshold
(0.42 of line height, 6px floor — the sweep is at MIN_GAP_FRAC), same crops:

    words found / label words      1,007 / 965   (1.04x overall)
      per profile                  studio_clean 0.82x (under-splits, 24 lines)
                                   handheld_heavy 1.64x, handheld_light 1.43x
                                   low_light 0.34x (38 under-split lines)
    paired, "all words of the line read exactly" vs "the line read exactly":
      both correct            2
      word ok / line bad      0    <- word reading rescued NOTHING
      line ok / word bad      5    <- word reading LOST five lines
      both wrong            193
    McNemar chi2 (continuity-corrected) 3.20, b=0 c=5

So splitting a line into words does not buy accuracy here — it costs a little.
That is consistent with the chunking result already recorded in
app/tools/crop_contract.js, and the mechanism is the same: when the recogniser
cannot read the glyphs, shorter crops do not help it read them. Tesseract's
advantage is not "it works on words", it is a font-agnostic feature set plus a
WORD lexicon pruning the output; segmentation alone copies the cheap half.

Honest sample caveat: 5 discordant pairs, p ~ 0.07 — not significant. It is
"no gain, directionally worse", not a proven loss. Revisit only if a retrained
model reads clean text at high accuracy, where segmentation might matter again;
and revisit the LEXICON (field_lexicon.py) as a decoder prior rather than as a
crop generator, which is where the published gains actually are.
"""

import numpy as np

# ── parameters, named so the sweep is reproducible ────────────────────────────
#
# Ink is anything darker than this in the line crop. 170 matches the renderer's
# own ink test (render_corpus.page_crops uses `band < 170`).
INK_THRESHOLD = 170

# A run of blank columns this wide or wider is a word break. Expressed as BOTH
# an absolute floor and a fraction of the line height, because the same absolute
# gap is a break on a 20px line and intra-word kerning on a 60px line. The
# effective threshold is the larger of the two.
MIN_GAP_PX = 6
MIN_GAP_FRAC = 0.42
#
# CALIBRATED, not guessed. Swept min_gap_px 2-8 x min_gap_frac 0.12-0.90 over the
# whole held-out render split (3,270 lines, 15,855 label words) and scored each
# setting by |found - expected| / expected. The first guess (0.12 / 3) found
# 1.73x too many segments: a 3px threshold on a 13px-tall tight crop cuts through
# intra-word kerning, and an over-segmented line can never read correctly.
#   0.12 / 3  -> 27,300 found (1.73x)   <- the guess
#   0.42 / 6  -> 16,085 found (1.02x, err 1.5%)   <- shipped
#   0.90 / 8  ->  8,993 found (0.57x)   <- under-splits, merges words
# The blank-run distribution inside these line crops has p50 0.08 and p90 0.42 of
# the line height, which is why the fraction dominates the absolute floor.

# A run narrower than this cannot be a word; it is speckle or a punctuation
# mark the charset does not even contain.
MIN_WORD_WIDTH = 2

# Columns carrying fewer than this many ink pixels are treated as white margin
# and trimmed, so a word's box is tight rather than padded.
TRIM_INK_PX = 1


def _runs(mask_1d):
    """Contiguous True runs as (start, end_exclusive) in a 1-D boolean array."""
    out, start = [], None
    for i, v in enumerate(mask_1d):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask_1d)))
    return out


def gap_threshold(line_h, min_gap_px=None, min_gap_frac=None):
    """Effective blank-run width that separates words for a line of this height."""
    g = MIN_GAP_PX if min_gap_px is None else min_gap_px
    f = MIN_GAP_FRAC if min_gap_frac is None else min_gap_frac
    return max(int(g), int(round(f * line_h)))


def segment_words(gray, w, box, ink_threshold=None, min_gap_px=None,
                  min_gap_frac=None, min_word_width=None, trim_ink_px=None):
    """Split a line crop into word boxes by column-ink-gap splitting.

    `gray`  uint8 2-D line crop
    `w`     its width (accepted for call compatibility; `box` governs extent)
    `box`   (x0, y0, x1, y1) the line box inside `gray`; returned boxes are in
            the SAME coordinate frame and never exceed it

    Returns a left-to-right list of tight (x0, y0, x1, y1) word boxes. An
    all-paper region returns [].
    """
    it = INK_THRESHOLD if ink_threshold is None else ink_threshold
    mw = MIN_WORD_WIDTH if min_word_width is None else min_word_width
    tp = TRIM_INK_PX if trim_ink_px is None else trim_ink_px

    H, W = gray.shape
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return []

    region = np.asarray(gray[y0:y1, x0:x1])
    ink = region < it                       # dark = ink
    col_ink = ink.sum(axis=0)
    thr = gap_threshold(y1 - y0, min_gap_px, min_gap_frac)

    # Split the column profile into runs of inked columns, breaking wherever a
    # blank run of >= thr columns intervenes. Trailing/leading blanks are margin,
    # not a break, so they are simply not part of any run.
    inked = col_ink >= tp
    boxes = []
    for s, e in _runs(inked):
        # Merge runs separated by fewer than `thr` blank columns: a short gap is
        # intra-word kerning. `_runs` already breaks on every blank, so re-merge.
        if boxes and s - boxes[-1][1] < thr:
            boxes[-1] = (boxes[-1][0], e)
        else:
            boxes.append((s, e))
    # boxes now hold (col_start, col_end_exclusive) merged word runs

    out = []
    for cs, ce in boxes:
        if ce - cs < mw:
            continue                        # speckle, not a word
        sub = ink[:, cs:ce]
        rows = np.where(sub.any(axis=1))[0]
        if len(rows) < 2:
            continue
        cols = np.where(sub.any(axis=0))[0]
        bx0 = x0 + cs + int(cols[0])
        bx1 = x0 + cs + int(cols[-1]) + 1
        by0 = y0 + int(rows[0])
        by1 = y0 + int(rows[-1]) + 1
        out.append((bx0, by0, bx1, by1))
    return out


def _selfcheck():
    """One runnable check: three words -> three boxes; blank -> none."""
    h, w = 32, 120
    img = np.full((h, w), 255, np.uint8)
    for s in (10, 40, 80):
        img[6:26, s:s + 12] = 0
    boxes = segment_words(img, w, (0, 0, w, h))
    assert len(boxes) == 3, boxes
    assert segment_words(np.full((h, w), 255, np.uint8), w, (0, 0, w, h)) == []
    print("word_segment self-check ok:", boxes)


if __name__ == "__main__":
    _selfcheck()
