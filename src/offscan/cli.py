#!/usr/bin/env python3
"""
SightLine — CLI interface.
Usage:
    sightline scan document.png                    # Scan and speak
    sightline scan document.png --json             # Output JSON instead of speech
    sightline scan document.png --full             # Read full text aloud
    sightline scan document.png --output out/      # Save audio to file
    sightline demo                                  # Run built-in demo
    sightline models --check                        # Check ONNX model status
    sightline models                                # Download ONNX models
"""

import argparse
import json
import os
import sys
import time

# Add src to path for direct execution
_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from offscan.pipeline import SightLinePipeline, ScanResult


def cmd_scan(args):
    """Scan a document image."""
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    pipeline = SightLinePipeline(
        voice=args.voice,
        rate=args.rate,
        speak=not args.json,
        output_dir=args.output_dir,
    )

    result = pipeline.scan(args.image, read_full=args.full)

    if args.json:
        print(result.to_json())
    else:
        print(f"\n{'='*60}")
        print(f"SightLine — Scan Complete")
        print(f"{'='*60}")
        print(f"Time: {result.total_time_ms:.0f}ms")
        print(f"Skew correction: {result.skew_angle:.1f}°")
        print(f"OCR engine: {result.ocr_engine}")
        print(f"OCR confidence: {result.ocr_confidence:.1%}")
        print(f"Words detected: {result.word_count}")
        print(f"\nDocument type: {result.document_label}")
        print(f"Classification confidence: {result.classification_confidence:.1%}")
        print(f"Classification engine: {result.classification_engine}")
        print(f"\nExtracted fields:")
        for f in result.extracted_fields:
            print(f"  {f['label']}: {f['value']} (conf: {f['confidence']:.0%}, src: {f['source']})")
        print(f"\nNER entities ({result.ner_engine}):")
        for e in result.entities:
            print(f"  {e['label']}: {e['text']} (conf: {e['confidence']:.0%})")
        print(f"\nSummary: {result.summary}")
        print(f"\nTTS engine: {result.tts_engine}")
        print(f"Fully offline: {result.fully_offline}")
        print(f"Network calls: {result.network_calls_made}")
        print(f"{'='*60}")


def cmd_demo(args):
    """Run a built-in demo with synthetic document images."""
    print("SightLine — Demo Mode")
    print("Generating synthetic document images...\n")

    from offscan.demo import generate_demo_images
    images = generate_demo_images()

    pipeline = SightLinePipeline(speak=not args.json)

    for label, image_path in images:
        print(f"\n{'─'*40}")
        print(f"Scanning: {label}")
        print(f"{'─'*40}")
        result = pipeline.scan(image_path, read_full=False)

        if args.json:
            print(result.to_json())
        else:
            print(f"  Document type: {result.document_label}")
            print(f"  OCR confidence: {result.ocr_confidence:.1%}")
            print(f"  Fields found: {len(result.extracted_fields)}")
            print(f"  Entities found: {len(result.entities)}")
            print(f"  Time: {result.total_time_ms:.0f}ms")
            print(f"  Summary: {result.summary[:120]}")
            if result.tts_engine != "none":
                print(f"  TTS: {result.tts_engine}")


def cmd_models(args):
    """Check or download ONNX models."""
    from offscan.setup_models import check_models, main as setup_main

    if args.check:
        print("SightLine — ONNX Model Status\n")
        for key, info in check_models().items():
            status = "✓ downloaded" if info["downloaded"] else "✗ not downloaded"
            print(f"  {key}: {status}")
            print(f"    {info['description']}")
        return

    setup_main()


def main():
    parser = argparse.ArgumentParser(
        prog="sightline",
        description="SightLine — Offline document scanner for the visually impaired",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # scan
    scan_parser = subparsers.add_parser("scan", help="Scan a document image")
    scan_parser.add_argument("image", help="Path to the document image")
    scan_parser.add_argument("--full", action="store_true", help="Read full text aloud (not just summary)")
    scan_parser.add_argument("--json", action="store_true", help="Output JSON instead of speech")
    scan_parser.add_argument("--voice", default="default", choices=["default", "male", "female"], help="TTS voice")
    scan_parser.add_argument("--rate", type=int, default=175, help="TTS speed (words per minute)")
    scan_parser.add_argument("--output-dir", default=None, help="Directory for audio output files")
    scan_parser.set_defaults(func=cmd_scan)

    # demo
    demo_parser = subparsers.add_parser("demo", help="Run built-in demo with synthetic documents")
    demo_parser.add_argument("--json", action="store_true", help="Output JSON instead of speech")
    demo_parser.set_defaults(func=cmd_demo)

    # models
    models_parser = subparsers.add_parser("models", help="Check or download ONNX models")
    models_parser.add_argument("--check", action="store_true", help="Check model status")
    models_parser.set_defaults(func=cmd_models)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
