"""
SightLine — Demo module.
Generates synthetic document images (banking, medical, legal, general)
using PIL/Pillow, then runs the full pipeline on them.

This lets you demo the entire system without needing real documents.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Sample document contents ──
SAMPLES = {
    "banking": {
        "title": "HSBC BANK STATEMENT",
        "lines": [
            "Account Holder: John Smith",
            "Account Number: 12345678",
            "Sort Code: 40-12-19",
            "",
            "Statement Period: January 2024",
            "Opening Balance:    £1,234.56",
            "Deposits:           £3,200.00",
            "Withdrawals:        £1,890.50",
            "Closing Balance:    £2,544.06",
            "",
            "Card ending 4521",
            "Available: £2,544.06",
        ],
    },
    "medical": {
        "title": "PRESCRIPTION",
        "lines": [
            "Patient: Jane Doe",
            "Date of Birth: 15/03/1985",
            "Patient ID: 1234567",
            "",
            "Rx: Amoxicillin 500mg",
            "Take one capsule three times daily",
            "for 7 days",
            "",
            "Rx: Ibuprofen 400mg",
            "Every 6-8 hours as needed",
            "Maximum: 1200mg per day",
            "",
            "WARNING: May cause drowsiness.",
            "Do not operate machinery.",
            "Keep out of reach of children.",
        ],
    },
    "legal": {
        "title": "LEASE AGREEMENT",
        "lines": [
            "Contract between Party A and Party B",
            "Dated 15th January 2024",
            "",
            "Property: 123 Baker Street, London NW1",
            "Tenant shall pay monthly rent of £2,500",
            "",
            "Clause 3.2: Terms of Service",
            "Clause 5.1: Termination Notice",
            "",
            "This contract is binding under the",
            "laws of England and Wales.",
            "",
            "Case Reference: 2024-CV-00456",
        ],
    },
    "general": {
        "title": "APPOINTMENT CONFIRMATION",
        "lines": [
            "Dear John,",
            "",
            "Your appointment is confirmed for",
            "Tuesday 15th at 2:00pm.",
            "Please arrive 10 minutes early.",
            "",
            "Location: Room 302, Building B",
            "Phone: +44 20 7946 0958",
            "Email: appointments@clinic.com",
            "",
            "Thank you.",
        ],
    },
}


def _get_font(size: int = 20):
    """Try to load a TrueType font, fall back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_document(title: str, lines: list[str], width: int = 800) -> Image.Image:
    """Render a document as an image using PIL."""
    # Calculate height based on number of lines
    line_height = 30
    height = 100 + len(lines) * line_height + 50

    # White background
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Title (bold, centered)
    title_font = _get_font(24)
    draw.text((width // 2 - 200, 30), title, fill="black", font=title_font)

    # Horizontal line under title
    draw.line([(50, 70), (width - 50, 70)], fill="black", width=2)

    # Body text
    body_font = _get_font(20)
    y = 90
    for line in lines:
        draw.text((50, y), line, fill="black", font=body_font)
        y += line_height

    return img


def generate_demo_images(output_dir: str | None = None) -> list[tuple[str, str]]:
    """
    Generate synthetic document images and return (label, path) pairs.
    """
    if output_dir is None:
        output_dir = os.path.join(tempfile.gettempdir(), "sightline_demo")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for category, content in SAMPLES.items():
        img = render_document(content["title"], content["lines"])
        path = os.path.join(output_dir, f"{category}.png")
        img.save(path)
        results.append((content["title"], path))

    return results
