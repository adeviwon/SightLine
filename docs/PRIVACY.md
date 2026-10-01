# OffScan — Privacy Policy

> **Zero data leaves your device.** This document explains how OffScan is
> designed and verified to protect user privacy.

## Core Principle

OffScan is a **100% offline** application. No image, OCR text, audio output,
telemetry, usage statistics, or any other user data is ever transmitted over
a network. The application functions completely without an internet
connection.

## Data Handling

### What we collect: **Nothing**

| Data type | Collected | Stored locally | Transmitted |
|-----------|-----------|---------------|-------------|
| Document images | ❌ No (user-provided) | ✅ User-controlled | ❌ Never |
| OCR text output | ❌ No (generated on-device) | ✅ User-controlled | ❌ Never |
| Audio output | ❌ No (generated on-device) | Ephemeral | ❌ Never |
| Usage analytics | ❌ No | ❌ No | ❌ Never |
| Crash reports | ❌ No | ❌ No | ❌ Never |
| IP address | ❌ No | ❌ No | ❌ Never |
| Device fingerprints | ❌ No | ❌ No | ❌ Never |

### Data lifecycle

1. **Input:** The user provides an image file from local storage.
2. **Processing:** All OCR, classification, and text-to-speech happen on-device.
3. **Output:** Text and audio files are written to a user-specified local
   directory. The user retains full control and may delete them at any time.
4. **Teardown:** No residual data is retained by the application beyond what
   the user explicitly saves.

## Network Access

OffScan does **not**:

- Make HTTP/HTTPS requests to any server.
- Upload images or text to cloud OCR services.
- Send analytics, telemetry, or crash reports.
- Download models or language packs at runtime (all are bundled).
- Use any third-party SDK that communicates over a network.

### Verification

- The Docker Compose configuration runs with `network_mode: "none"`,
  physically preventing any network access.
- The CI pipeline builds and tests in an isolated environment.
- The dependency manifest (`pyproject.toml`) is auditable: every dependency
  is a known local-computation library (OpenCV, Tesseract, ONNX Runtime,
  scikit-learn, pyttsx3). No networking libraries are included.

### Dependency audit

| Dependency | Purpose | Network calls? |
|------------|---------|----------------|
| `opencv-python` | Image preprocessing | ❌ No |
| `pytesseract` | Tesseract wrapper | ❌ No (subprocess only) |
| `tesseract-ocr` (system) | OCR engine | ❌ No |
| `pyttsx3` | Text-to-speech | ❌ No (local speech engine) |
| `onnxruntime` | Model inference | ❌ No (local CPU) |
| `scikit-learn` | Feature extraction | ❌ No |
| `numpy` | Array operations | ❌ No |
| `Pillow` | Image I/O | ❌ No |
| `click` | CLI framework | ❌ No |
| `rich` | Terminal formatting | ❌ No |

## ONNX Models

All ONNX models used for classification are:

- Downloaded **at build time** (during Docker image construction or
  `scripts/install.sh`) and bundled locally.
- Loaded from the local filesystem at runtime.
- Never fetched from a remote endpoint during normal operation.

## Compliance

### GDPR

- No personal data is processed by OffScan infrastructure.
- All processing occurs on the user's own device (data controller = user).
- No data controller / processor relationship exists between the user and
  OffScan maintainers.

### CCPA

- No consumer data is collected, sold, or shared.
- No "do not sell my personal information" mechanism is needed because no
  information is collected.

### HIPAA (for medical document scanning)

- When used in an offline environment, OffScan does not transmit PHI.
- Users in healthcare settings should ensure local storage is encrypted and
  access-controlled per their organization's policies.

## Source Code Transparency

OffScan is open-source under the MIT License. Every line of code is
auditable at [github.com/mrsmallflame-ai/OffScan](https://github.com/mrsmallflame-ai/OffScan).
No obfuscated or proprietary binary is required for the core pipeline.

## Summary

```
┌──────────────────────────────────────────────────────────┐
│                    Your Device                            │
│                                                          │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌───────────┐ │
│  │  Image   │─▶│ Preprocess│─▶│   OCR   │─▶│  Output   │ │
│  │  (local) │  └──────────┘  └─────────┘  └─────┬─────┘ │
│  └─────────┘                                     │       │
│                                                  ▼       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │          Local filesystem (user-controlled)          │ │
│  │              .txt  /  .json  /  audio                 │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ═════════════════ NO NETWORK TRAFFIC ═════════════════  │
└──────────────────────────────────────────────────────────┘
```

**Questions?** Open an issue at
[github.com/mrsmallflame-ai/OffScan/issues](https://github.com/mrsmallflame-ai/OffScan/issues)
or email contact@mrsmallflame.ai.
