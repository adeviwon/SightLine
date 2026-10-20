"""
Word-level segmentation of a detected line — Tesseract's first structural trick.

The recogniser is asked to read one squeezed crop per LINE. Exact-match on an
18-char line needs 98.8% char accuracy; on an 8-char word it needs only 97.2%
((1-CER)^L = 0.8). Split the line at INK GAPS BETWEEN WORDS and each crop is
short, so the error budget moves in our favour.

This is NOT the chunking experiment that already failed here (fixed-width
contiguous chunks measured 54% vs 98% for one squeezed crop,
app/tools/crop_contract.js). Word segmentation splits at whitespace gaps and
labels each crop with the individual word, which is a different experiment with
a different failure mode (over/under-splitting, not mid-word cuts).

Pure function over a uint8 array: no PIL, no I/O, deterministic.
"""

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import word_segment as WS  # noqa: E402


def line(widths, gaps, h=32, pad=0):
    """A synthetic line crop: dark word bars on white paper, in a list order.

    widths  word widths in px
    gaps    gap widths in px BETWEEN words (len(gaps) == len(widths) - 1)
    pad     white margin added on the left and right
    """
    assert len(gaps) == len(widths) - 1
    total = pad + sum(widths) + sum(gaps) + pad
    img = np.full((h, total), 255, np.uint8)
    x = pad
    for i, wd in enumerate(widths):
        img[6:h - 6, x:x + wd] = 0
        x += wd
        if i < len(gaps):
            x += gaps[i]
    return img


def seg(img, box=None):
    h, w = img.shape
    return WS.segment_words(img, w, box or (0, 0, w, h))


def _split_gap(h=32):
    """A gap that must split, for a line of this height: just over the threshold."""
    return WS.gap_threshold(h) + 2


def test_three_well_separated_words_segment_to_three():
    img = line([10, 12, 8], [_split_gap(), _split_gap()])
    boxes = seg(img)
    assert len(boxes) == 3
    # left to right, non-overlapping, each covering its word
    xs = [b[0] for b in boxes]
    assert xs == sorted(xs)
    assert boxes[0][0] <= 1 and boxes[-1][2] >= img.shape[1] - 2


def test_single_word_line_segments_to_one():
    img = line([20], [])
    boxes = seg(img)
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert x1 - x0 >= 18 and y1 - y0 >= 16


def test_gap_below_the_threshold_stays_one_word_and_above_it_splits():
    # The threshold is calibrated (0.42 of the line height, 6px floor -- see
    # word_segment.MIN_GAP_FRAC). Assert the RULE against the function that owns
    # it, not a literal: when the threshold was recalibrated from 0.12/3 to
    # 0.42/6, the literal fixtures silently stopped exercising a split at all.
    th = WS.gap_threshold(32)
    assert len(seg(line([10, 10], [1]))) == 1, "1px is kerning, not a word break"
    assert len(seg(line([10, 10], [max(2, th - 1)]))) == 1, "just under must not split"
    assert len(seg(line([10, 10], [th + 1]))) == 2, "just over must split"


def test_boxes_never_exceed_the_line_box():
    img = line([10, 10], [_split_gap()], pad=40)
    box = (20, 4, img.shape[1] - 20, img.shape[0] - 4)
    boxes = seg(img, box)
    assert boxes, "words inside the box must be found"
    for x0, y0, x1, y1 in boxes:
        assert x0 >= box[0] and y0 >= box[1]
        assert x1 <= box[2] and y1 <= box[3]


def test_all_paper_segments_to_zero():
    img = np.full((32, 120), 255, np.uint8)
    assert seg(img) == []


def test_determinism():
    img = line([10, 12, 8], [12, 12])
    a = seg(img)
    b = seg(img)
    assert a == b


def test_tiny_speckle_is_not_a_word():
    # A 1px-wide blob is noise, not a word. MIN_WORD_WIDTH is the floor.
    img = np.full((32, 80), 255, np.uint8)
    img[10:20, 10:11] = 0            # 1px speckle -> dropped
    img[6:26, 30:40] = 0             # real word
    img[6:26, 60:70] = 0             # real word
    boxes = seg(img)
    assert len(boxes) == 2
    assert WS.MIN_WORD_WIDTH >= 2


def test_white_margin_is_trimmed():
    # A word padded with white on both sides must come back tight, not full-width.
    img = np.full((32, 120), 255, np.uint8)
    img[6:26, 50:70] = 0
    boxes = seg(img)
    assert len(boxes) == 1
    x0, _, x1, _ = boxes[0]
    assert x0 >= 45 and x1 <= 75, "white margin must be trimmed"
    assert x1 - x0 <= 25
