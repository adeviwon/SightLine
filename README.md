# SightLine — Offline Document Scanner for the Visually Impaired

> **Your eyes, offline. Nothing leaves your phone.**

SightLine is a fully offline document scanning pipeline that reads banking statements,
medical prescriptions, and legal documents aloud for blind and visually impaired users.
Every component runs on-device — OCR, document classification, named entity recognition,
and text-to-speech. **Zero network calls. Zero cloud APIs. Zero data leaving the device.**

Built for [ICHack](https://ichack.org/) (Imperial College London Hackathon).

---

## Why This Matters

43 million people worldwide are blind. 2 million in the UK live with sight loss.
Every day, they face a choice between **independence and privacy** when reading
sensitive documents — bank statements, prescriptions, legal contracts.

Existing solutions (Google Lens, Seeing AI, Be My Eyes) send document images to
cloud servers. SightLine eliminates that trade-off.

---

## Pipeline Architecture

```
Camera/Image → OpenCV Preprocessing → OCR (Tesseract + ONNX) → Document Classification → NER → TTS → Audio
```

### Components

| Stage | Technology | Purpose |
|---|---|---|
| **Preprocessing** | OpenCV (deskew, denoise, CLAHE, adaptive threshold) | Clean up image for better OCR |
| **OCR** | Tesseract + ONNX deep OCR (CRAFT detection + CRNN recognition) | Extract text from image |
| **Classification** | ONNX model + TF-IDF/LogisticRegression fallback | Categorize: banking, medical, legal, general |
| **Field Extraction** | Regex + ONNX NER model | Extract dosages, account numbers, dates, amounts |
| **TTS** | espeak-ng / pyttsx3 | Read document aloud — fully offline |

### ML Models (ONNX Runtime)

SightLine uses quantized ONNX models for on-device inference:

1. **Text classifier** — MiniLM-L6-v2 quantized for document classification
2. **NER model** — Distilled BERT for entity extraction (drugs, dosages, money, dates)
3. **OCR detection** — CRAFT-style text region detection
4. **OCR recognition** — CRNN-style text recognition

All models run through ONNX Runtime with CPUExecutionProvider. If models aren't
downloaded, the pipeline falls back to Tesseract + TF-IDF + regex — still fully
functional, just less accurate.

---

## Quick Start

```bash
# Clone
git clone https://github.com/adeviwon/SightLine.git
cd SightLine

# Install system dependencies (macOS)
brew install tesseract espeak-ng

# Install Python dependencies
pip3 install opencv-python pytesseract pyttsx3 onnxruntime scikit-learn numpy Pillow click rich psutil

# Run the demo — generates 4 synthetic documents and scans them
PYTHONPATH=src python3 -m offscan.cli demo

# Scan a real document image (photograph it with your phone, pass the file)
PYTHONPATH=src python3 -m offscan.cli scan /path/to/document.png

# Get JSON output instead of speech
PYTHONPATH=src python3 -m offscan.cli scan /path/to/document.png --json

# Read the full document aloud (not just the summary)
PYTHONPATH=src python3 -m offscan.cli scan /path/to/document.png --full

# Run the test suite (22 tests, all offline)
pip3 install pytest
PYTHONPATH=src python3 -m pytest tests/ -v
```

### Phone Camera Mode (Web App)

Run a local web server on your laptop and scan documents from your phone's camera:

```bash
# Install Flask
pip3 install flask

# Start the web server (accessible from any device on your WiFi)
PYTHONPATH=src python3 -m offscan.webapp

# Then open the printed URL on your phone browser
# (e.g. http://192.168.1.x:5000)
# Point your phone camera at a document and tap "Scan Document"
```

The web app uses your phone's rear camera via the browser's `getUserMedia` API.
It captures the image, sends it to your laptop for offline processing, and plays
the audio result back on your phone. All processing is on your laptop — no cloud.

You can also upload existing photos via the "Upload" tab.

### Demo

The built-in demo generates four synthetic documents (banking statement, medical prescription,
legal contract, appointment letter), runs the full pipeline on each, and reads them aloud.
No real documents needed.

---

## Testing

```bash
python -m pytest tests/ -v
```

All tests run offline with synthetic data — no real documents needed.

---

## Project Structure

```
OffScan/
├── src/offscan/
│   ├── __init__.py          # Package init
│   ├── preprocess.py        # OpenCV image preprocessing
│   ├── ocr.py               # Tesseract + ONNX deep OCR
│   ├── classifier.py        # Document classification (ONNX + TF-IDF)
│   ├── ner.py               # Named entity recognition (ONNX + regex)
│   ├── tts.py               # Offline TTS (espeak-ng / pyttsx3)
│   ├── pipeline.py          # End-to-end pipeline orchestration
│   ├── cli.py               # Command-line interface
│   ├── demo.py              # Synthetic document generation
│   └── setup_models.py      # ONNX model download/caching
├── tests/
│   └── test_pipeline.py     # Full test suite
├── docs/
│   ├── HACKATHON_GUIDE.md   # Winning strategy guide
│   ├── architecture.md      # Pipeline architecture with diagrams
│   ├── PRIVACY.md           # Privacy compliance documentation
│   └── ROADMAP.md           # Post-hackathon roadmap
├── scripts/
│   └── install.sh           # Cross-platform setup script
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI
├── models/                  # Cached ONNX models (downloaded separately)
├── pyproject.toml           # Package config
├── Dockerfile               # Offline Docker image
├── docker-compose.yml       # Docker with network_mode: "none"
├── Makefile                 # install, test, demo, lint, clean
└── requirements.txt         # Python dependencies
```

---

## Privacy Guarantee

**SightLine makes zero network calls.** This is verifiable:

1. The `docker-compose.yml` sets `network_mode: "none"` — the container physically
   cannot access the network.
2. Every ML inference runs through ONNX Runtime or Tesseract — both run locally.
3. The pipeline outputs `network_calls_made: 0` in every scan result.
4. All code is open-source — a security researcher can audit the zero-network claim.

See [docs/PRIVACY.md](docs/PRIVACY.md) for full compliance documentation.

---

## License

MIT — see [LICENSE](LICENSE).
