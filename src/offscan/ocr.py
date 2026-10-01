"""
SightLine — OCR module.
Dual-engine OCR: Tesseract baseline + ONNX deep learning OCR path.

Tesseract runs first as the fast baseline. If ONNX models are available
(see setup_models.py), a deep learning OCR path runs for higher accuracy.
Results are merged with confidence scoring.
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


def _tesseract_ocr(image: np.ndarray) -> OCRResult:
    """Run Tesseract OCR with word-level bounding boxes and confidence."""
    # Get detailed word data: text, confidence, bounding box
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--psm 6",  # Assume a uniform block of text
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

    # Build full text from word sequences, preserving line breaks
    full_text = pytesseract.image_to_string(image, config="--psm 6")
    line_count = full_text.strip().count("\n") + 1 if full_text.strip() else 0

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return OCRResult(
        text=full_text.strip(),
        words=words,
        mean_confidence=mean_conf,
        engine="tesseract",
        line_count=line_count,
    )


class DeepOCR:
    """
    ONNX-based deep learning OCR engine.
    Uses a text detection model (CRAFT-style) + recognition model (CRNN-style)
    loaded as ONNX. Falls back gracefully to Tesseract if models unavailable.
    """

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

    def detect_text_regions(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect text bounding boxes using ONNX text detection model.
        Falls back to OpenCV contour-based detection if model unavailable.
        """
        if self.detection_session is None:
            return self._fallback_detect(image)

        # ONNX inference: input is a normalized, resized grayscale image
        h, w = image.shape[:2]
        input_size = (320, 320)  # Standard detection input
        resized = _resize_and_pad(image, input_size)
        normalized = resized.astype(np.float32) / 255.0
        normalized = np.expand_dims(normalized, axis=(0, 1))  # NCHW

        try:
            outputs = self.detection_session.run(None, {self.detection_session.get_inputs()[0].name: normalized})
            score_map = outputs[0][0, 0]

            # Threshold the score map to get text regions
            mask = (score_map > 0.5).astype(np.uint8) * 255
            mask = _resize_back(mask, (w, h))

            # Find bounding boxes from the mask
            contours, _ = cv2_find_contours(mask)
            boxes = []
            for c in contours:
                x, y, cw, ch = cv2_bounding_rect(c)
                if cw * ch > 100:  # Filter tiny noise
                    boxes.append((x, y, cw, ch))
            return boxes if boxes else self._fallback_detect(image)
        except Exception:
            return self._fallback_detect(image)

    def recognize_text(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Recognize text in a cropped region using ONNX recognition model.
        Falls back to Tesseract for recognition if model unavailable.
        """
        if self.recognition_session is None:
            # Use Tesseract on the crop
            text = pytesseract.image_to_string(crop, config="--psm 7").strip()
            return text, 0.85

        # ONNX recognition: CRNN-style, input is (1, 1, 32, W)
        h, w = crop.shape[:2]
        target_h = 32
        target_w = max(32, int(w * target_h / h))
        resized = _resize_and_pad(crop, (target_w, target_h))
        normalized = resized.astype(np.float32) / 127.5 - 1.0
        normalized = np.expand_dims(normalized, axis=(0, 1))

        try:
            outputs = self.recognition_session.run(None, {self.recognition_session.get_inputs()[0].name: normalized})
            # Decode CTC output (simplified — real CRNN uses a character dictionary)
            pred = outputs[0]
            text, conf = _decode_ctc(pred)
            return text, conf
        except Exception:
            text = pytesseract.image_to_string(crop, config="--psm 7").strip()
            return text, 0.80

    def _fallback_detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """OpenCV-based text region detection using MSER or morphological approach."""
        if len(image.shape) == 3:
            gray = cv2_cvtColor(image)
        else:
            gray = image

        # MSER text detection
        mser = cv2_mser_create()
        regions, _ = mser.detectRegions(gray)

        if regions is None or len(regions) == 0:
            # Morphological fallback
            _, binary = cv2_threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            dilated = cv2.dilate(binary, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                if cw * ch > 200 and 0.1 < ch / cw < 10:
                    boxes.append((x, y, cw, ch))
            return boxes

        # Convert MSER regions to bounding boxes
        boxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2).astype(np.int32))
            if w * h > 100:
                boxes.append((x, y, w, h))
        return boxes

    def run(self, image: np.ndarray) -> OCRResult:
        """Full deep OCR: detect regions, recognize text in each, merge results."""
        regions = self.detect_text_regions(image)
        words = []
        texts = []
        confidences = []

        for x, y, w, h in regions:
            crop = image[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            text, conf = self.recognize_text(crop)
            if text:
                words.append(OCRWord(text=text, confidence=conf, bbox=(x, y, w, h)))
                texts.append(text)
                confidences.append(conf)

        full_text = " ".join(texts)
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            words=words,
            mean_confidence=mean_conf,
            engine="onnx-deep-ocr" if self.available else "opencv-tesseract-hybrid",
            line_count=len(texts),
        )


def run_ocr(image: np.ndarray, deep_ocr: DeepOCR | None = None) -> OCRResult:
    """
    Run OCR on a preprocessed image.
    If deep_ocr is provided and available, use it; otherwise Tesseract only.
    """
    # Always run Tesseract as baseline
    tess_result = _tesseract_ocr(image)

    if deep_ocr and deep_ocr.available:
        deep_result = deep_ocr.run(image)
        # Merge: if deep OCR found more text with higher confidence, prefer it
        if len(deep_result.text) > len(tess_result.text) and deep_result.mean_confidence > tess_result.mean_confidence:
            return deep_result
        # Otherwise return Tesseract but note deep was attempted
        tess_result.engine = "tesseract+onnx-fallback"
        return tess_result

    return tess_result


# ── Helper functions (abstracted for cleaner testing) ──

def _resize_and_pad(image, target_size):
    h, w = image.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    import cv2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Pad to target
    top = (th - new_h) // 2
    bottom = th - new_h - top
    left = (tw - new_w) // 2
    right = tw - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded


def _resize_back(image, target_size):
    import cv2
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def cv2_find_contours(binary):
    import cv2
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, _


def cv2_bounding_rect(contour):
    import cv2
    return cv2.boundingRect(contour)


def cv2_cvtColor(image):
    import cv2
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def cv2_threshold(gray, low, high, flag):
    import cv2
    return cv2.threshold(gray, low, high, flag)


def cv2_mser_create():
    import cv2
    return cv2.MSER_create()


def _decode_ctc(pred):
    """Simplified CTC decoding — collapses repeated characters and removes blanks."""
    # pred shape: (1, T, C) where T=timesteps, C=character classes
    if pred.ndim == 3:
        pred = pred[0]
    # Greedy decoding
    raw = pred.argmax(axis=-1)
    # CTC blank is index 0
    decoded = []
    prev = 0
    for idx in raw:
        if idx != 0 and idx != prev:
            decoded.append(chr(idx + 31) if idx < 128 else '?')  # ASCII offset
        prev = idx

    text = ''.join(decoded).strip()
    confidence = float(pred.max(axis=-1).mean()) if len(pred) > 0 else 0.0
    return text, confidence
