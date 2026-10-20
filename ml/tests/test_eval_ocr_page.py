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


def test_read_page_does_not_consult_the_locate_override(monkeypatch):
    """A MEASURED DELETION, pinned so it cannot come back unmeasured.

    Against an exact drawn ground truth (120 re-rendered test pages, 1800 lines)
    the locate_document override costs 56 lines: line recall 1744/1800 = 0.969
    with it against 1800/1800 without, exact McNemar p=2.776e-17, pairs (b=0,
    c=56) -- it is never better on a single page. The loss is concentrated on
    the low_light profile (recall 0.867, 14 of 28 pages lose lines) and it wins
    nothing back on the 193 real hand-photographed receipts (transcript
    count-match 16/193 with it, 20/193 without, p=0.125). locate_document() and
    bands_in_document() stay in the module -- they are still tested, and still
    the right tool for finding a receipt on a table -- they are just no longer
    in the bander's path.

    If this test fails, someone put the override back: re-run
    `bash run.sh py ml/src/line_detect.py --split test --limit-pages 120` and
    check that the paired verdict still favours dropping it before believing it.
    """
    called = []
    monkeypatch.setattr(EO, "locate_document",
                        lambda a: called.append(a) or (0, 0, a.shape[1], a.shape[0]))
    monkeypatch.setattr(EO, "bands_in_document", lambda a, q: [(0, 5)])   # poison
    monkeypatch.setattr(EO, "recognise",
                        lambda model, crop, device: ("LINE", 0.9))

    a = np.full((400, 640), 245, np.uint8)
    for y in (60, 160, 260):
        a[y:y + 30, 60:580] = 20

    out = EO.read_page(None, None, a)
    assert called == [], ("read_page consulted locate_document: the override is "
                          "back, and it is worth 56 lines over 120 pages")
    # The poison override returns a SINGLE 5-row band of blank paper, whose ink
    # column count is far below read_page's 4-column floor, so it yields no
    # reads at all. A non-empty result therefore proves the bander's own bands
    # were the ones used. The exact band count is deliberately NOT asserted: how
    # many bands the bander finds on a synthetic page is the bander's own tests'
    # business, and pinning it here would fail this test for reasons that have
    # nothing to do with the override.
    assert out, ("read_page produced nothing -- the poison override's single blank "
                 "band was used, so the override is back in the path")
