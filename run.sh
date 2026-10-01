#!/bin/bash
# SightLine — One-command setup & run (FULLY OFFLINE)
# Uses your Mac's built-in webcam. No WiFi, no internet, no network.

set -e

echo "╔══════════════════════════════════════════════╗"
echo "║  SightLine — Fully Offline Setup            ║"
echo "║  Your eyes, offline. Nothing leaves device.  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Clone if not in the repo
if [ ! -f "src/offscan/pipeline.py" ]; then
    echo "→ Cloning SightLine..."
    git clone https://github.com/adeviwon/SightLine.git /tmp/sightline-run
    cd /tmp/sightline-run
fi

# Install system dependencies (macOS)
echo "→ Checking system dependencies..."
if ! command -v tesseract &>/dev/null; then
    echo "  Installing Tesseract OCR..."
    brew install tesseract
fi
if ! command -v espeak-ng &>/dev/null; then
    echo "  Installing espeak-ng..."
    brew install espeak-ng
fi

# Install Python dependencies
echo "→ Installing Python packages..."
pip3 install --quiet opencv-python pytesseract pyttsx3 onnxruntime scikit-learn numpy Pillow flask psutil 2>/dev/null || pip install --quiet opencv-python pytesseract pyttsx3 onnxruntime scikit-learn numpy Pillow flask psutil

# Run — localhost only, no external network
echo ""
echo "──────────────────────────────────────────────"
echo "  SightLine is starting on YOUR MAC."
echo "  Your webcam is the camera."
echo ""
echo "  Open this in your browser:"
echo "    http://localhost:5000"
echo ""
echo "  Point your Mac at a document → tap Scan."
echo "  It reads it aloud. Zero network."
echo "──────────────────────────────────────────────"
echo ""

export PYTHONPATH=src
python3 -m offscan.webapp --host 127.0.0.1 --port 5000
