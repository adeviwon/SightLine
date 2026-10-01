"""
SightLine — Image preprocessing module.
Enhances document images for better OCR accuracy using OpenCV.

Handles real-world conditions: blur, noise, shadows, glare, rotation, low contrast.

Steps:
  1. Grayscale conversion
  2. Quality assessment (blur detection, brightness check)
  3. Perspective correction (deskew)
  4. Adaptive denoising (stronger for noisy images)
  5. Contrast enhancement (CLAHE — adaptive to lighting)
  6. Binarization (adaptive threshold + Otsu fallback)
  7. Border cleanup
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
    quality_score: float
    is_blurry: bool
    is_dark: bool
    warnings: list


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


def _assess_quality(gray: np.ndarray) -> tuple[float, bool, bool]:
    """
    Assess image quality: blur score and brightness.
    Returns (quality_score, is_blurry, is_dark).
    """
    # Blur detection using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = laplacian_var < 100  # Threshold for blur
    blur_score = min(1.0, laplacian_var / 500)

    # Brightness check
    mean_brightness = np.mean(gray)
    is_dark = mean_brightness < 50

    quality_score = (blur_score + (1.0 if not is_dark else 0.0)) / 2
    return quality_score, is_blurry, is_dark


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


def _adaptive_denoise(gray: np.ndarray, is_blurry: bool) -> np.ndarray:
    """
    Adaptive denoising — stronger for noisy/blurry images.
    Uses bilateral filter (preserves edges) + non-local means.
    """
    # Bilateral filter — removes noise while keeping text edges sharp
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    
    if is_blurry:
        # For blurry images, try sharpening with unsharp mask
        gaussian = cv2.GaussianBlur(denoised, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        return sharpened
    
    return denoised


def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE — adaptive histogram equalization for uneven lighting."""
    # Clip limit adapts to image — higher for low contrast images
    clip_limit = 3.0 if np.std(gray) < 50 else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """
    Multi-strategy binarization — tries adaptive threshold first,
    falls back to Otsu if adaptive produces too much noise.
    """
    # Strategy 1: Adaptive Gaussian threshold
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=10,
    )

    # Strategy 2: Otsu's method
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Check which produces better results (less noise = fewer small contours)
    adaptive_noise = cv2.countNonZero(cv2.Canny(adaptive, 50, 150))
    otsu_noise = cv2.countNonZero(cv2.Canny(otsu, 50, 150))

    # Pick the cleaner one
    return adaptive if adaptive_noise <= otsu_noise else otsu


def preprocess(image_source) -> PreprocessResult:
    """
    Full preprocessing pipeline for a document image.
    Handles real-world conditions: blur, noise, shadows, glare, rotation.

    Args:
        image_source: File path or numpy array.

    Returns:
        PreprocessResult with all intermediate images and quality metrics.
    """
    original = _load_image(image_source)
    gray = _to_grayscale(original)

    # Quality assessment
    quality_score, is_blurry, is_dark = _assess_quality(gray)
    warnings = []
    if is_blurry:
        warnings.append("Image appears blurry. Hold the camera steadier or move closer.")
    if is_dark:
        warnings.append("Image appears dark. Move to better lighting.")

    # Deskew
    skew = _estimate_skew_angle(gray)
    deskewed = _deskew(gray, skew)

    # Denoise (adaptive strength)
    denoised = _adaptive_denoise(deskewed, is_blurry)

    # Enhance contrast
    enhanced = _enhance_contrast(denoised)

    # Binarize
    thresholded = _binarize(enhanced)

    return PreprocessResult(
        original=original,
        grayscale=gray,
        deskewed=deskewed,
        denoised=denoised,
        enhanced=enhanced,
        thresholded=thresholded,
        final=thresholded,
        skew_angle=skew,
        quality_score=quality_score,
        is_blurry=is_blurry,
        is_dark=is_dark,
        warnings=warnings,
    )
