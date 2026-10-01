"""
SightLine — OCR module.
Multi-pass OCR with quality fallback for real-world conditions.

Strategy:
  1. Assess image quality from preprocessing
  2. Try Tesseract with PSM 6 (uniform block) first
  3. If confidence < 50%, retry with PSM 3 (auto) and PSM 11 (sparse text)
  4. Pick the result with highest confidence
  5. Filter garbage (too many short words, non-text characters)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytesseract
from PIL import Image as PILImage

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


@dataclass
class OCRWord:
    """A single recognized word with bounding box and confidence."""
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


@dataclass 
class OCRResult:
    """Full OCR output — raw text, word-level data, and confidence stats."""
    text: str
    words: list[OCRWord] = field(default_factory=list)
    mean_confidence: float = 0.0
    engine: str = "tesseract"
    line_count: int = 0
    psm_mode: int = 6
    quality_warning: str = ""


def _tesseract_pass(image: np.ndarray, psm: int) -> OCRResult:
    """Run a single Tesseract OCR pass with given PSM mode."""
    config = f"--psm {psm}"
    
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config=config,
    )

    words = []
    confidences = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if text and conf > 0:
            words.append(OCRWord(
                text=text,
                confidence=conf / 100.0,
                bbox=(data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
            ))
            confidences.append(conf / 100.0)

    full_text = pytesseract.image_to_string(image, config=config)
    
    # Clean text — remove excessive whitespace and non-printable chars
    full_text = re.sub(r'\n{3,}', '\n\n', full_text.strip())
    
    line_count = full_text.strip().count("\n") + 1 if full_text.strip() else 0
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return OCRResult(
        text=full_text,
        words=words,
        mean_confidence=mean_conf,
        engine="tesseract",
        line_count=line_count,
        psm_mode=psm,
    )


def _is_garbage(result: OCRResult) -> bool:
    """Detect garbage OCR output (too many single-char words or non-text)."""
    if not result.words:
        return True
    
    # If most words are 1-2 characters, likely garbage
    short_words = sum(1 for w in result.words if len(w.text) <= 2)
    if short_words > len(result.words) * 0.6:
        return True
    
    # If there are too many words for a document (likely noise)
    if len(result.words) > 500:
        return True
    
    return False


def _filter_words(result: OCRResult) -> OCRResult:
    """Filter out low-confidence and garbage words."""
    filtered_words = [w for w in result.words if w.confidence > 0.3 and len(w.text) >= 2]
    
    if not filtered_words:
        # Keep at least something
        filtered_words = [w for w in result.words if w.confidence > 0.1]
    
    result.words = filtered_words
    if result.words:
        result.mean_confidence = sum(w.confidence for w in result.words) / len(result.words)
    
    return result


def run_ocr(image: np.ndarray, quality_score: float = 1.0, 
            is_blurry: bool = False, warnings: list = None) -> OCRResult:
    """
    Run multi-pass OCR on a preprocessed image.
    
    Args:
        image: Preprocessed binary image
        quality_score: Quality score from preprocessing (0-1)
        is_blurry: Whether the image was detected as blurry
        warnings: Preprocessing warnings to pass through
    
    Returns:
        Best OCRResult from multiple passes.
    """
    warnings = warnings or []
    
    # Pass 1: PSM 6 (assume uniform block of text) — best for documents
    result = _tesseract_pass(image, psm=6)
    result = _filter_words(result)
    
    # If first pass is good enough, return it
    if result.mean_confidence > 0.6 and not _is_garbage(result) and len(result.text) > 20:
        if warnings:
            result.quality_warning = "; ".join(warnings)
        return result
    
    # Pass 2: Try PSM 3 (fully automatic page segmentation)
    result_psm3 = _tesseract_pass(image, psm=3)
    result_psm3 = _filter_words(result_psm3)
    
    # Pass 3: Try PSM 11 (sparse text — good for noisy images)
    result_psm11 = _tesseract_pass(image, psm=11)
    result_psm11 = _filter_words(result_psm11)
    
    # Pick the best result
    candidates = [r for r in [result, result_psm3, result_psm11] 
                  if not _is_garbage(r) and len(r.text) > 10]
    
    if candidates:
        best = max(candidates, key=lambda r: (r.mean_confidence, len(r.text)))
    else:
        # All passes failed — return the least bad result
        best = max([result, result_psm3, result_psm11], key=lambda r: len(r.text))
        best.quality_warning = "Low confidence OCR. Image may be too blurry or dark. Please retake."
    
    if warnings:
        best.quality_warning = (best.quality_warning + "; " if best.quality_warning else "") + "; ".join(warnings)
    
    return best


# ── Deep OCR (ONNX) — kept for future enhancement ──

class DeepOCR:
    """ONNX-based deep learning OCR engine (placeholder for future enhancement)."""
    
    def __init__(self, detection_model_path: str | None = None,
                 recognition_model_path: str | None = None):
        self.detection_session = None
        self.recognition_session = None
        self.available = False
        
        if not HAS_ONNX:
            return
            
        if detection_model_path:
            try:
                self.detection_session = ort.InferenceSession(
                    detection_model_path,
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                pass
                
        if recognition_model_path:
            try:
                self.recognition_session = ort.InferenceSession(
                    recognition_model_path,
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                pass
                
        self.available = self.detection_session is not None and self.recognition_session is not None

    def run(self, image: np.ndarray) -> OCRResult:
        """Run deep OCR (falls back to Tesseract if unavailable)."""
        return run_ocr(image)
