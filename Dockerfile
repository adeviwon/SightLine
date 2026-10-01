# ── OffScan Dockerfile — fully offline document scanning pipeline ──
# All system dependencies (Tesseract, OpenCV libs) are baked into the image.
# No network calls are made at runtime.

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# ── System dependencies: Tesseract OCR, OpenCV runtime libs, audio ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libopencv-dev \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        espeak-ng \
        alsa-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies first (better layer caching) ──
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

# ── Copy the rest of the project ──
COPY scripts/ ./scripts/
COPY docs/ ./docs/

# ── Non-root user for security ──
RUN useradd --create-home --shell /bin/bash offscan
USER offscan

# ── Offline guarantee: no ENTRYPOINT that fetches anything ──
ENTRYPOINT ["offscan"]
CMD ["--help"]
