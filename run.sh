#!/bin/bash
# SightLine — One-command setup & run
# Works on macOS. Uses your Mac's webcam or your phone camera (same WiFi).

set -e

echo "╔══════════════════════════════════════════════╗"
echo "║  SightLine — Setup & Run                    ║"
echo "║  Your eyes, offline. Nothing leaves device.  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Clone if not already in the directory
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

# Run the web app
echo ""
echo "──────────────────────────────────────────────"
echo "  SightLine is starting..."
echo ""
echo "  On your Mac webcam:"
echo "    http://localhost:5000"
echo ""

# Get local IP for phone access
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "your-mac-ip")
echo "  On your phone (same WiFi):"
echo "    http://$LOCAL_IP:5000"
echo ""
echo "  Open either URL in your browser."
echo "  Point camera at a document → tap Scan Document"
echo "  It reads the document aloud. Fully offline."
echo "──────────────────────────────────────────────"
echo ""

export PYTHONPATH=src
python3 -m offscan.webapp --host 0.0.0.0 --port 5000
