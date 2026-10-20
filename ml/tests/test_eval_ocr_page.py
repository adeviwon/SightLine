"""
test_eval_ocr_page.py — the page reader: WHERE it crops, and what it returns.

`eval_handheld` is the number the repo quotes for real photographs (line recall,
line-exact accuracy, CER). It works by projecting ink rows to find text lines and
cropping each band out of the page. Two things about that are worth pinning:

  1. COORDINATE SPACE. The projection is computed after the image is trimmed to
     the ink extent, but the crop is taken from the ORIGINAL array. Band indices
     from the trimmed projection are therefore only valid if the trim offset is
     added back. Without it the recogniser is handed a window of blank paper
     shifted up by however many rows the trim removed — silent, and fatal to the
     score.
  2. WHAT IT RETURNS. One (text, confidence) pair per detected line, not a joined
     string: the confidence is what the product's gate needs, and the caller
     cannot recover it from a string.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eval_ocr as EO  # noqa: E402


def _page_with_line_at(y, h=400, w=640, line_h=30):
    """Paper with ONE dark text bar at row `y`, so any trim offset is knowable."""
    a = np.full((h, w), 245, np.uint8)
    a[y:y + line_h, 60:w - 60] = 20
    return a


def test_read_page_crops_the_rows_the_line_is_actually_on(monkeypatch):
    """The fallback path: no document could be located, so the bands came from a
    ROW-TRIMMED projection. Those indices are only valid on the original array
    once the trim offset is added back — otherwise the crop lands `off` rows too
    high, which on a photo with 150px of table above the receipt means the
    recogniser is fed blank paper and every line reads as nothing."""
    monkeypatch.setattr(EO, "locate_document", lambda a: None)
    a = _page_with_line_at(300)          # ink lives only in the bottom third
    seen = []

    def fake_recognise(model, crop, device):
        seen.append(crop)
        return "LINE", 0.9

    monkeypatch.setattr(EO, "recognise", fake_recognise)
    out = EO.read_page(None, "cpu", a)

    assert out, "the page reader found no line at all"
    assert seen, "recognise() was never called"
    for crop in seen:
        # A crop of blank paper (mean ~245) means the band coordinates were in
        # the wrong coordinate space. The bar itself is ~20.
        assert crop.mean() < 200, (
            "handed the recogniser a cage of blank paper: crop mean %.1f" % crop.mean())


def test_read_page_returns_one_text_and_confidence_per_line(monkeypatch):
    a = _page_with_line_at(100)
    monkeypatch.setattr(EO, "recognise", lambda m, c, d: ("HELLO", 0.75))
    out = EO.read_page(None, "cpu", a)
    assert out, "no line was read"
    assert all(pair == ("HELLO", 0.75) for pair in out), out


def test_read_page_is_empty_on_a_blank_page(monkeypatch):
    monkeypatch.setattr(EO, "recognise",
                        lambda m, c, d: (_ for _ in ()).throw(AssertionError("called")))
    assert EO.read_page(None, "cpu", np.full((200, 300), 245, np.uint8)) == []
