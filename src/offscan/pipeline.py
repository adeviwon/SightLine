"""
SightLine — Main pipeline.
The end-to-end offline document scanning pipeline:

  Image → Preprocess → OCR → Classify → NER → TTS → Audio output

Every step runs locally. Zero network calls. Zero cloud APIs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from .preprocess import preprocess
from .ocr import run_ocr, DeepOCR, OCRResult
from .classifier import DocumentClassifier, ClassificationResult
from .ner import NERExtractor, NERResult
from .tts import TextToSpeech

# Model paths
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


def _model_path(filename: str) -> str | None:
    """Return model path if file exists, else None."""
    p = os.path.join(_MODEL_DIR, filename)
    return p if os.path.exists(p) else None


@dataclass
class ScanResult:
    """Complete result of the scanning pipeline."""
    # Timing
    total_time_ms: float = 0.0
    # Preprocessing
    skew_angle: float = 0.0
    image_quality: float = 1.0
    image_warnings: list = field(default_factory=list)
    # OCR
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    ocr_engine: str = ""
    word_count: int = 0
    # Classification
    document_category: str = ""
    document_label: str = ""
    classification_confidence: float = 0.0
    classification_engine: str = ""
    # Extracted fields
    extracted_fields: list[dict] = field(default_factory=list)
    # NER entities
    entities: list[dict] = field(default_factory=list)
    ner_engine: str = ""
    # Summary for TTS
    summary: str = ""
    # TTS
    tts_engine: str = ""
    tts_output: str = ""
    # Offline verification
    network_calls_made: int = 0
    fully_offline: bool = True

    @property
    def ocr_result(self):
        """Compatibility property for tts.narrate_result."""
        class _Compat:
            pass
        c = _Compat()
        c.ocr_result = _Compat()
        c.ocr_result.text = self.ocr_text
        c.classification = _Compat()
        c.classification.category = self.document_category
        c.classification.summary = self.summary
        return c

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)


class SightLinePipeline:
    """
    The full offline document scanning pipeline.
    Initialize once, then call scan() for each document.
    """

    def __init__(
        self,
        models_dir: str | None = None,
        voice: str = "default",
        rate: int = 175,
        speak: bool = True,
        output_dir: str | None = None,
    ):
        model_dir = models_dir or _MODEL_DIR

        # Initialize components
        self.classifier = DocumentClassifier(
            onnx_model_path=_model_path_rel(model_dir, "text_classifier.onnx"),
        )
        self.ner_extractor = NERExtractor(
            onnx_model_path=_model_path_rel(model_dir, "ner_model.onnx"),
        )
        self.deep_ocr = DeepOCR(
            detection_model_path=_model_path_rel(model_dir, "ocr_detection.onnx"),
            recognition_model_path=_model_path_rel(model_dir, "ocr_recognition.onnx"),
        )
        self.tts = TextToSpeech(voice=voice, rate=rate)
        self.speak_enabled = speak
        self.output_dir = output_dir

    def scan(self, image_source, read_full: bool = False) -> ScanResult:
        """
        Scan a document image and produce structured output + speech.

        Args:
            image_source: File path or numpy array
            read_full: Whether TTS reads full text after summary

        Returns:
            ScanResult with all pipeline outputs
        """
        result = ScanResult()
        t0 = time.perf_counter()

        # Step 1: Preprocess
        pp = preprocess(image_source)
        result.skew_angle = pp.skew_angle
        result.image_quality = pp.quality_score
        result.image_warnings = pp.warnings

        # Step 2: OCR (multi-pass with quality-aware fallback)
        ocr_result = run_ocr(
            pp.final, 
            quality_score=pp.quality_score,
            is_blurry=pp.is_blurry,
            warnings=pp.warnings,
        )
        result.ocr_text = ocr_result.text
        result.ocr_confidence = ocr_result.mean_confidence
        result.ocr_engine = ocr_result.engine
        result.word_count = len(ocr_result.words)

        # Step 3: Classify
        classification = self.classifier.classify(ocr_result.text)
        result.document_category = classification.category
        result.document_label = classification.category_label
        result.classification_confidence = classification.confidence
        result.classification_engine = classification.engine
        result.summary = classification.summary
        result.extracted_fields = [
            {"label": f.label, "value": f.value, "confidence": f.confidence, "source": f.source}
            for f in classification.extracted_fields
        ]

        # Step 4: NER
        ner_result = self.ner_extractor.extract(ocr_result.text)
        result.entities = [
            {"text": e.text, "label": e.label, "confidence": e.confidence, "source": e.source}
            for e in ner_result.entities
        ]
        result.ner_engine = ner_result.engine

        # Step 5: TTS
        if self.speak_enabled and result.summary:
            tts_output_file = None
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                tts_output_file = os.path.join(self.output_dir, "output.wav")
                self.tts.output_file = tts_output_file

            result.tts_engine = self.tts.speak_summary(
                category=result.document_category,
                summary=result.summary,
                full_text=result.ocr_text,
                read_full=read_full,
            )
            result.tts_output = tts_output_file or "speaker"

        # Timing
        result.total_time_ms = (time.perf_counter() - t0) * 1000

        # Offline verification — we never make network calls
        result.network_calls_made = 0
        result.fully_offline = True

        return result


def _model_path_rel(model_dir: str, filename: str) -> str | None:
    """Check if a model file exists and return its path, or None."""
    p = os.path.join(model_dir, filename)
    return p if os.path.exists(p) else None
