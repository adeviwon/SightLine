"""
SightLine — Real-world test suite.
Tests the pipeline on messy, realistic document images that simulate
real webcam photos: blur, noise, rotation, shadows, glare, low contrast.

These tests verify the pipeline works in production conditions, not just
on perfect synthetic images.
"""

import os
import sys
import unittest
import random
import numpy as np

_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from offscan.pipeline import SightLinePipeline
from offscan.preprocess import preprocess


def _get_font(size=18):
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def _make_messy_image(lines, rotation=0, blur=0, noise=0, shadow=False, 
                      low_contrast=False, glare=False):
    """Create a realistic messy document photo."""
    random.seed(42)  # Deterministic for reproducible tests
    width, height = 800, 100 + len(lines) * 28 + 50
    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    font = _get_font(18)
    y = 30
    for line in lines:
        draw.text((50 + random.randint(-2, 2), y), line, fill=(30, 30, 30), font=font)
        y += 28
    if shadow:
        overlay = Image.new("RGB", (width, height), (0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for x in range(width):
            op = int(80 * (x / width))
            od.line([(x, 0), (x, height)], fill=(op, op, op))
        img = Image.blend(img, overlay, 0.3)
    if glare:
        overlay = Image.new("RGB", (width, height), (0, 0, 0))
        od = ImageDraw.Draw(overlay)
        cx, cy = width // 3, height // 3
        for r in range(200, 0, -5):
            od.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(min(200,r*2),)*3)
        img = Image.blend(img, overlay, 0.25)
    if noise > 0:
        arr = np.array(img)
        arr = np.clip(arr.astype(np.int16) + np.random.normal(0, noise, arr.shape).astype(np.int16), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    if low_contrast:
        arr = np.array(img)
        arr = np.clip((arr.astype(float) - 128) * 0.5 + 160, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    if rotation != 0:
        img = img.rotate(rotation, fillcolor=(200, 200, 200), expand=False)
    return np.array(img)


PRESCRIPTION = [
    "PRESCRIPTION", "Patient: Jane Doe", "Date: 01/10/2026", "",
    "Rx: Amoxicillin 500mg", "Take one capsule three times daily", "for 7 days", "",
    "Rx: Ibuprofen 400mg", "Every 6-8 hours as needed", "Maximum: 1200mg per day", "",
    "WARNING: May cause drowsiness", "Keep out of reach of children",
]

BANKING = [
    "HSBC BANK STATEMENT", "Account Holder: John Smith", "Account Number: 12345678",
    "Sort Code: 40-12-19", "", "Opening Balance: GBP 1,234.56", "Deposits: GBP 3,200.00",
    "Withdrawals: GBP 1,890.50", "Closing Balance: GBP 2,544.06", "", "Card ending 4521",
]


class TestRealWorldConditions(unittest.TestCase):
    """Test pipeline on messy, realistic document photos."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = SightLinePipeline(speak=False)

    def _scan(self, lines, **kwargs):
        img = _make_messy_image(lines, **kwargs)
        return self.pipeline.scan(img)

    def test_perfect_image(self):
        """Perfect conditions — should work flawlessly."""
        result = self._scan(PRESCRIPTION)
        self.assertGreater(result.ocr_confidence, 0.85)
        self.assertEqual(result.document_category, "medical")
        self.assertGreater(len(result.extracted_fields), 5)

    def test_slight_blur(self):
        """Slight blur (camera not perfectly still) — should still work."""
        result = self._scan(PRESCRIPTION, blur=1)
        self.assertGreater(result.ocr_confidence, 0.7)
        self.assertEqual(result.document_category, "medical")
        self.assertGreater(len(result.extracted_fields), 3)

    def test_rotation_5_degrees(self):
        """Document slightly rotated — should deskew and read."""
        result = self._scan(PRESCRIPTION, rotation=5)
        self.assertGreater(result.ocr_confidence, 0.8)
        self.assertEqual(result.document_category, "medical")

    def test_rotation_10_degrees(self):
        """Document significantly rotated — should still deskew."""
        result = self._scan(PRESCRIPTION, rotation=10)
        self.assertGreater(result.ocr_confidence, 0.75)
        self.assertEqual(result.document_category, "medical")

    def test_camera_noise(self):
        """Camera sensor noise — should filter garbage and extract real text."""
        result = self._scan(PRESCRIPTION, noise=25)
        self.assertGreater(result.ocr_confidence, 0.5)
        self.assertLess(result.word_count, 100, "Should not produce hundreds of garbage words")
        self.assertEqual(result.document_category, "medical")

    def test_low_contrast(self):
        """Poor lighting / low contrast — should enhance and read."""
        result = self._scan(PRESCRIPTION, low_contrast=True)
        self.assertGreater(result.ocr_confidence, 0.8)
        self.assertEqual(result.document_category, "medical")

    def test_shadow(self):
        """Uneven lighting / shadow on document — should handle with CLAHE."""
        result = self._scan(PRESCRIPTION, shadow=True)
        self.assertGreater(result.ocr_confidence, 0.85)
        self.assertEqual(result.document_category, "medical")

    def test_glare(self):
        """Glare / reflection on document — should still extract key fields."""
        result = self._scan(PRESCRIPTION, glare=True)
        self.assertGreater(result.ocr_confidence, 0.5)
        self.assertGreater(len(result.extracted_fields), 1)

    def test_everything_bad(self):
        """Worst case: blur + noise + shadow + rotation + low contrast.
        In extreme conditions, the pipeline should still extract SOME text
        and classify it as a known document type (not crash or produce garbage)."""
        result = self._scan(PRESCRIPTION, blur=2, noise=20, shadow=True, rotation=5, low_contrast=True)
        # Should extract some text even in worst case
        self.assertGreater(len(result.ocr_text), 10, "Should extract some text even in worst case")
        # Should classify as a known category (not crash)
        self.assertIn(result.document_category, ["medical", "banking", "legal", "general"])
        # Should have been warned about image quality
        self.assertTrue(len(getattr(result, 'image_warnings', [])) > 0 or result.ocr_confidence < 0.6)

    def test_banking_document_messy(self):
        """Banking document with real-world photo conditions."""
        result = self._scan(BANKING, blur=2, noise=15, shadow=True, rotation=3)
        self.assertEqual(result.document_category, "banking")
        self.assertGreater(result.ocr_confidence, 0.5)

    def test_banking_document_perfect(self):
        """Banking document in perfect conditions."""
        result = self._scan(BANKING)
        self.assertGreater(result.ocr_confidence, 0.9)
        self.assertEqual(result.document_category, "banking")
        self.assertGreater(len(result.extracted_fields), 2)

    def test_offline_guarantee_messy(self):
        """Pipeline must make zero network calls even on bad images."""
        result = self._scan(PRESCRIPTION, blur=3, noise=30)
        self.assertEqual(result.network_calls_made, 0)
        self.assertTrue(result.fully_offline)


class TestQualityAssessment(unittest.TestCase):
    """Test image quality detection."""

    def test_detects_blurry_image(self):
        """Should detect and warn about blurry images."""
        img = _make_messy_image(PRESCRIPTION, blur=3)
        result = preprocess(img)
        self.assertTrue(result.is_blurry)
        self.assertGreater(len(result.warnings), 0)

    def test_detects_dark_image(self):
        """Should detect and warn about dark images."""
        img = _make_messy_image(PRESCRIPTION, noise=0)
        arr = np.array(img)
        arr = (arr * 0.2).astype(np.uint8)  # Very dark
        result = preprocess(arr)
        self.assertTrue(result.is_dark)

    def test_no_warnings_for_good_image(self):
        """Should not warn about good quality images."""
        img = _make_messy_image(PRESCRIPTION)
        result = preprocess(img)
        self.assertFalse(result.is_blurry)
        self.assertEqual(len(result.warnings), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
