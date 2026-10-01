"""
SightLine — Test suite.
Tests every component of the pipeline with synthetic data.
All tests run fully offline — no network access required.
"""

import json
import os
import sys
import tempfile
import unittest

# Add src to path
_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from offscan.preprocess import preprocess
from offscan.ocr import run_ocr, OCRResult
from offscan.classifier import DocumentClassifier, CATEGORIES
from offscan.ner import NERExtractor
from offscan.tts import TextToSpeech
from offscan.pipeline import SightLinePipeline, ScanResult
from offscan.demo import generate_demo_images


class TestPreprocess(unittest.TestCase):
    """Test image preprocessing."""

    def test_preprocess_synthetic_image(self):
        """Preprocessing should run on a synthetic document image."""
        images = generate_demo_images()
        label, path = images[0]
        result = preprocess(path)
        self.assertIsNotNone(result.original)
        self.assertIsNotNone(result.final)
        self.assertEqual(result.final.ndim, 2)  # Grayscale thresholded

    def test_preprocess_numpy_input(self):
        """Preprocessing should accept numpy arrays."""
        import numpy as np
        img = np.zeros((200, 400), dtype=np.uint8)
        img[50:150, 50:350] = 255
        result = preprocess(img)
        self.assertEqual(result.final.shape[0], 200)

    def test_skew_correction(self):
        """Skew angle should be a float."""
        images = generate_demo_images()
        _, path = images[0]
        result = preprocess(path)
        self.assertIsInstance(result.skew_angle, float)


class TestOCR(unittest.TestCase):
    """Test OCR engine."""

    def test_ocr_on_synthetic_document(self):
        """OCR should extract text from a synthetic document."""
        images = generate_demo_images()
        # Use the banking statement (clear text)
        label, path = images[0]  # banking
        pp = preprocess(path)
        result = run_ocr(pp.final)
        self.assertIsInstance(result.text, str)
        self.assertGreater(len(result.text), 0)
        self.assertGreater(result.mean_confidence, 0.0)

    def test_ocr_returns_words(self):
        """OCR should return word-level data."""
        images = generate_demo_images()
        _, path = images[0]
        pp = preprocess(path)
        result = run_ocr(pp.final)
        self.assertIsInstance(result.words, list)
        if result.words:
            self.assertTrue(hasattr(result.words[0], "text"))
            self.assertTrue(hasattr(result.words[0], "confidence"))


class TestClassifier(unittest.TestCase):
    """Test document classifier."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = DocumentClassifier()

    def test_classify_banking(self):
        """Classifier should identify banking documents."""
        text = "Account Number: 12345678 Sort Code: 40-12-19 Balance: £2,544.06"
        result = self.classifier.classify(text)
        self.assertEqual(result.category, "banking")
        self.assertGreater(result.confidence, 0.35)

    def test_classify_medical(self):
        """Classifier should identify medical documents."""
        text = "Rx: Amoxicillin 500mg Take one capsule three times daily Warning: May cause drowsiness"
        result = self.classifier.classify(text)
        self.assertEqual(result.category, "medical")
        self.assertGreater(result.confidence, 0.35)

    def test_classify_legal(self):
        """Classifier should identify legal documents."""
        text = "Contract Agreement between Party A and Party B Clause 3.2: Terms of Service binding under laws"
        result = self.classifier.classify(text)
        self.assertEqual(result.category, "legal")
        self.assertGreater(result.confidence, 0.35)

    def test_classify_general(self):
        """Classifier should identify general documents."""
        text = "Hello your appointment is confirmed for Tuesday at 2pm please arrive early"
        result = self.classifier.classify(text)
        self.assertEqual(result.category, "general")

    def test_classify_empty_text(self):
        """Classifier should handle empty text gracefully."""
        result = self.classifier.classify("")
        self.assertEqual(result.category, "general")
        self.assertEqual(result.confidence, 0.0)

    def test_field_extraction_banking(self):
        """Banking field extraction should find account numbers and amounts."""
        text = "Account Number: 12345678 Sort Code: 40-12-19 Balance: £2,544.06"
        result = self.classifier.classify(text)
        field_labels = [f.label for f in result.extracted_fields]
        self.assertIn("Account Number", field_labels)

    def test_field_extraction_medical(self):
        """Medical field extraction should find dosages and warnings."""
        text = "Rx: Amoxicillin 500mg three times daily Warning: May cause drowsiness"
        result = self.classifier.classify(text)
        field_labels = [f.label for f in result.extracted_fields]
        # At least some fields should be extracted
        self.assertGreater(len(result.extracted_fields), 0)


class TestNER(unittest.TestCase):
    """Test named entity recognition."""

    @classmethod
    def setUpClass(cls):
        cls.ner = NERExtractor()

    def test_extract_drugs(self):
        """NER should find drug names."""
        text = "Prescription: Amoxicillin 500mg and Metformin 1000mg"
        result = self.ner.extract(text)
        drug_entities = [e for e in result.entities if e.label == "DRUG"]
        self.assertGreater(len(drug_entities), 0)

    def test_extract_dosages(self):
        """NER should find dosage information."""
        text = "Take 500mg twice daily and 10ml once daily"
        result = self.ner.extract(text)
        dosage_entities = [e for e in result.entities if e.label == "DOSAGE"]
        self.assertGreater(len(dosage_entities), 0)

    def test_extract_money(self):
        """NER should find monetary amounts."""
        text = "Balance: £2,544.06 Payment: $1,200.00"
        result = self.ner.extract(text)
        money_entities = [e for e in result.entities if e.label == "MONEY"]
        self.assertGreater(len(money_entities), 0)

    def test_extract_dates(self):
        """NER should find dates."""
        text = "Appointment on 15/03/2024 and 20th January 2024"
        result = self.ner.extract(text)
        date_entities = [e for e in result.entities if e.label == "DATE"]
        self.assertGreater(len(date_entities), 0)


class TestTTS(unittest.TestCase):
    """Test text-to-speech."""

    def test_tts_engine_detection(self):
        """TTS should detect an available engine."""
        tts = TextToSpeech()
        self.assertIn(tts.engine, ["espeak-ng", "espeak", "pyttsx3", "none"])

    def test_tts_to_file(self):
        """TTS should be able to output to a file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        try:
            tts = TextToSpeech(output_file=output_path)
            engine = tts.speak("Test message")
            self.assertNotEqual(engine, "none")
            # File may or may not be created depending on engine, just check no crash
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestPipeline(unittest.TestCase):
    """Test the full pipeline end-to-end."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = SightLinePipeline(speak=False)

    def test_scan_banking_document(self):
        """Full pipeline should scan a banking document."""
        images = generate_demo_images()
        label, path = images[0]  # banking
        result = self.pipeline.scan(path)
        self.assertIsInstance(result, ScanResult)
        self.assertGreater(result.total_time_ms, 0)
        self.assertGreater(len(result.ocr_text), 0)
        self.assertEqual(result.fully_offline, True)
        self.assertEqual(result.network_calls_made, 0)

    def test_scan_medical_document(self):
        """Full pipeline should scan a medical document."""
        images = generate_demo_images()
        label, path = images[1]  # medical
        result = self.pipeline.scan(path)
        self.assertIsInstance(result, ScanResult)
        self.assertGreater(len(result.ocr_text), 0)

    def test_scan_result_to_json(self):
        """ScanResult should serialize to JSON."""
        images = generate_demo_images()
        _, path = images[0]
        result = self.pipeline.scan(path)
        json_str = result.to_json()
        parsed = json.loads(json_str)
        self.assertIn("ocr_text", parsed)
        self.assertIn("document_category", parsed)
        self.assertIn("fully_offline", parsed)

    def test_offline_guarantee(self):
        """Pipeline must make zero network calls."""
        images = generate_demo_images()
        _, path = images[0]
        result = self.pipeline.scan(path)
        self.assertEqual(result.network_calls_made, 0)
        self.assertTrue(result.fully_offline)


if __name__ == "__main__":
    unittest.main(verbosity=2)
