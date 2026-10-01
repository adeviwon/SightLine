"""
SightLine — Image preprocessing module.
Enhances document images for better OCR accuracy using OpenCV.

Steps:
  1. Grayscale conversion
  2. Deskew (correct document rotation)
  3. Denoise (remove scan artifacts)
  4. Contrast enhancement (CLAHE)
  5. Adaptive thresholding (binarize text)
  6. Border cleanup
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    """Result of preprocessing — carries every intermediate image."""
    original: np.ndarray
    grayscale: np.ndarray
    deskewed: np.ndarray
    denoised: np.ndarray
    enhanced: np.ndarray
    thresholded: np.ndarray
    final: np.ndarray
    skew_angle: float


def _load_image(image_source) -> np.ndarray:
    """Load image from file path or accept numpy array directly."""
    if isinstance(image_source, np.ndarray):
        return image_source
    img = cv2.imread(image_source)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_source}")
    return img


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate document skew angle in degrees using minAreaRect on text blobs."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 100]
    if not contours:
        return 0.0
    angles = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        rect = cv2.minAreaRect(c)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        angles.append(angle)
    angles.sort()
    median_angle = angles[len(angles) // 2]
    return median_angle


def _deskew(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image to correct skew."""
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _denoise(gray: np.ndarray) -> np.ndarray:
    """Remove scan noise while preserving text edges."""
    return cv2.fastNlMeansDenoising(gray, h=10)


def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE — adaptive histogram equalization for uneven lighting."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Binarize using adaptive thresholding — handles uneven illumination."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=10,
    )


def preprocess(image_source) -> PreprocessResult:
    """
    Full preprocessing pipeline for a document image.
    Args:
        image_source: File path or numpy array.
    Returns:
        PreprocessResult with all intermediate images.
    """
    original = _load_image(image_source)
    gray = _to_grayscale(original)
    skew = _estimate_skew_angle(gray)
    deskewed = _deskew(gray, skew)
    denoised = _denoise(deskewed)
    enhanced = _enhance_contrast(denoised)
    thresholded = _adaptive_threshold(enhanced)
    return PreprocessResult(
        original=original,
        grayscale=gray,
        deskewed=deskewed,
        denoised=denoised,
        enhanced=enhanced,
        thresholded=thresholded,
        final=thresholded,
        skew_angle=skew,
    )
