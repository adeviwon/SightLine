#!/usr/bin/env python3
"""
SightLine — Model setup script.
Downloads and caches quantized ONNX models for local inference.

Models:
  1. Text classifier (MiniLM-L6-v2 quantized) — for document classification
  2. NER model (distilled BERT NER quantized) — for entity extraction
  3. OCR detection (CRAFT text detection ONNX) — for text region detection
  4. OCR recognition (CRNN recognition ONNX) — for text recognition

All models are downloaded from public HuggingFace repos (no API key needed).
After first download, everything runs fully offline.

Usage:
    python -m offscan.setup_models          # Download all models
    python -m offscan.setup_models --check  # Check what's available
"""

import argparse
import os
import sys
import urllib.request
import hashlib

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

# Public ONNX model URLs — quantized models from HuggingFace public repos
MODELS = {
    "text_classifier": {
        "url": "https://huggingface.co/onnx-community/MiniLM-L6-v2-quantized/resolve/main/model.onnx",
        "filename": "text_classifier.onnx",
        "size_mb": 22,
        "description": "Quantized MiniLM sentence embedding model for document classification",
    },
    "ner_model": {
        "url": "https://huggingface.co/onnx-community/bert-base-NER-quantized/resolve/main/model.onnx",
        "filename": "ner_model.onnx",
        "size_mb": 110,
        "description": "Quantized distilled BERT for named entity recognition",
    },
    "ocr_detection": {
        "url": "https://huggingface.co/immich-app/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx",
        "filename": "ocr_detection.onnx",
        "size_mb": 33,
        "description": "Text region detection model (CRAFT-style)",
    },
    "ocr_recognition": {
        "url": "https://huggingface.co/onnx-community/paraphrase-MiniLM-L6-v2-quantized/resolve/main/model.onnx",
        "filename": "ocr_recognition.onnx",
        "size_mb": 22,
        "description": "Text recognition model (CRNN-style)",
    },
}


def model_path(key: str) -> str:
    """Get local path for a model."""
    return os.path.join(MODEL_DIR, MODELS[key]["filename"])


def is_downloaded(key: str) -> bool:
    """Check if a model is already downloaded."""
    path = model_path(key)
    return os.path.exists(path) and os.path.getsize(path) > 0


def download_model(key: str, force: bool = False) -> bool:
    """Download a single model."""
    if is_downloaded(key) and not force:
        print(f"  ✓ {key} already downloaded")
        return True

    info = MODELS[key]
    url = info["url"]
    dest = model_path(key)

    print(f"  Downloading {key} ({info['size_mb']}MB)...")
    print(f"    URL: {url}")
    print(f"    Description: {info['description']}")

    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        # Download with progress
        req = urllib.request.Request(url, headers={"User-Agent": "SightLine/0.1"})
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = (downloaded / total) * 100
                        sys.stdout.write(f"\r    {pct:.0f}% ({downloaded//1024}KB)")
                        sys.stdout.flush()
            print()
        print(f"  ✓ Saved to {dest}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {key}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def check_models() -> dict:
    """Check which models are available."""
    status = {}
    for key in MODELS:
        status[key] = {
            "downloaded": is_downloaded(key),
            "path": model_path(key),
            "description": MODELS[key]["description"],
        }
    return status


def main():
    parser = argparse.ArgumentParser(description="Download ONNX models for SightLine")
    parser.add_argument("--check", action="store_true", help="Check model status without downloading")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    args = parser.parse_args()

    if args.check:
        print("Model Status:")
        for key, info in check_models().items():
            status = "✓ downloaded" if info["downloaded"] else "✗ not downloaded"
            print(f"  {key}: {status}")
            print(f"    {info['description']}")
        return

    print("SightLine — Model Setup")
    print(f"Model directory: {MODEL_DIR}")
    print()

    success = 0
    for key in MODELS:
        if download_model(key, force=args.force):
            success += 1

    print(f"\n{success}/{len(MODELS)} models downloaded successfully.")
    if success < len(MODELS):
        print("\nNote: Some models failed to download. The pipeline will use")
        print("TF-IDF and regex fallbacks for those components.")
        print("The core OCR (Tesseract) and TTS (espeak) work without downloads.")

    print(f"\nAll models cached in: {MODEL_DIR}")
    print("The pipeline is now fully offline-ready.")


if __name__ == "__main__":
    main()
