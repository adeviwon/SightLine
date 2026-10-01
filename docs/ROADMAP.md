# OffScan — Post-Hackathon Roadmap

> Where OffScan is headed after the initial hackathon release.

## Current State (v0.1.0 — Hackathon MVP)

- ✅ Single-image OCR pipeline (OpenCV → Tesseract → output)
- ✅ Document classification (ONNX + scikit-learn)
- ✅ Text-to-speech playback (pyttsx3)
- ✅ Fully offline operation (verified via `network_mode: none`)
- ✅ CLI interface (Click)
- ✅ Docker support

---

## Phase 1 — Stability & Polish (v0.2.0)

**Target: 4–6 weeks post-hackathon**

- [ ] Comprehensive test suite (target 90%+ coverage)
- [ ] CI matrix for macOS and Windows runners
- [ ] Automated performance benchmarks (CI badge)
- [ ] Proper logging with structured output (JSON lines)
- [ ] Graceful degradation when Tesseract is not installed
- [ ] Configuration file support (`.offscan.toml`)
- [ ] Internationalization: pluggable Tesseract language packs (fra, deu, spa, chi_sim)

## Phase 2 — Batch & Multi-Page (v0.3.0)

**Target: 8–10 weeks post-hackathon**

- [ ] Multi-page PDF input support (pdf2image / PyMuPDF)
- [ ] Directory batch mode with parallel processing (multiprocessing)
- [ ] Progress bars for batch runs (rich)
- [ ] Output archive: structured `.zip` with text, JSON, and metadata
- [ ] Diff mode: compare two scans of the same document and highlight changes
- [ ] OCR confidence heatmap overlay (debug visualization)

## Phase 3 — Smarter Classification (v0.4.0)

**Target: 12–16 weeks post-hackathon**

- [ ] Custom ONNX layout-analysis model ( LayoutLMv3 distilled, CPU-only)
- [ ] Table extraction (structure recognition → CSV/Excel)
- [ ] Key-value pair extraction (receipts, invoices, forms)
- [ ] Multi-language OCR with automatic language detection
- [ ] Confidence-weighted field extraction (reject low-confidence matches)
- [ ] Active learning loop: save uncertain samples (locally) for model improvement

## Phase 4 — Accessibility & UX (v0.5.0)

**Target: 16–20 weeks post-hackathon**

- [ ] GUI front-end (PySide6 or Tauri): drag-and-drop image → output
- [ ] Real-time camera capture mode (live preview with edge detection)
- [ ] Voice command interface ("OffScan, read the last page")
- [ ] Configurable TTS voices and speed
- [ ] High-contrast and large-text accessibility modes
- [ ] Screen reader compatibility for the CLI

## Phase 5 — Distribution & Ecosystem (v1.0.0)

**Target: 6 months post-hackathon**

- [ ] PyPI publication with signed releases (sigstore)
- [ ] Homebrew formula and AUR package
- [ ] Snap / Flatpak for Linux desktop
- [ ] Standalone executables (PyInstaller / Nuitka) — no Python required
- [ ] Plugin system with documented SDK
- [ ] MkDocs documentation site (hosted on GitHub Pages, statically generated)
- [ ] Benchmarks published on the docs site (auto-updated from CI)

## Long-Term Vision (v1.x+)

- [ ] Mobile companion app (Flutter / React Native) sharing the ONNX model
- [ ] Embedded deployment on Raspberry Pi 5 (optimized ONNX quantization)
- [ ] Edge TPU support (Coral USB Accelerator) for accelerated inference
- [ ] Community model zoo: shareable, signed ONNX classifiers for new document types
- [ ] Federation protocol: optionally sync learned model weights across devices
  (still offline by default; opt-in only)

## Non-Goals

To keep the project focused and its privacy guarantee intact, the following
are explicitly **out of scope**:

- ❌ Cloud-hosted OCR or storage
- ❌ Telemetry, analytics, or usage tracking of any kind
- ❌ Account / login system
- ❌ Online model serving

## Release Cadence

| Phase | Version | Cadence |
|-------|---------|---------|
| MVP | v0.1.0 | Hackathon |
| Stability | v0.2.0 | Monthly patch |
| Batch | v0.3.0 | Monthly patch |
| Smart classification | v0.4.0 | Monthly patch |
| Accessibility | v0.5.0 | Bi-weekly |
| Stable release | v1.0.0 | Quarterly minor |

All releases remain MIT-licensed and 100% offline.
