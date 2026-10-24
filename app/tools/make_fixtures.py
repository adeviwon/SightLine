"""Fixtures for the browser suites — the ones a *product* path needs.

    bash run.sh py app/tools/make_fixtures.py

Writes:
  /tmp/blank.png    a blank white page            the refusal case
  /tmp/noise.png    speckle                       the garbage-OCR case
  /tmp/letter.pdf   a letter-sized PDF with text  the pdf.js render path

NOTE ON THE PDF FIXTURE'S GEOMETRY. The line detector's band limits scale with
page height (`layout.scaleFor`: a band must be at least 1.2% of the page height),
which is correct for a photographed page — text on a receipt or a letter is 2-4%
of page height. A PDF whose MediaBox is huge relative to its text (e.g. a
screenshot pasted into a 1240x1754-point page: 14px lines on a 1684px canvas =
0.8%) legitimately yields ZERO lines, and that is the detector working as
designed, not the PDF path failing. So this writes a real letter page: 2400x3100
px at 300 dpi = 576x744 points, whose 26px text is 1.5% of the height.
"""
from PIL import Image, ImageDraw
import random

Image.new("RGB", (1200, 900), (255, 255, 255)).save("/tmp/blank.png")

im = Image.new("RGB", (1200, 900), (250, 250, 250))
d = ImageDraw.Draw(im)
random.seed(7)
for _ in range(6000):
    x, y = random.randrange(1200), random.randrange(900)
    g = random.randrange(0, 255)
    d.rectangle([x, y, x + 1, y + 1], fill=(g, g, g))
im.save("/tmp/noise.png")

# A letter page: 8.5x11in at 300dpi. Saved with resolution=300 so the MediaBox
# is 612x792 points (a normal page), not 2400x3100.
page = Image.new("RGB", (2400, 3100), (255, 255, 255))
dd = ImageDraw.Draw(page)
from PIL import ImageFont
from pathlib import Path
font_pairs = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
]
pair = next((pair for pair in font_pairs if all(Path(f).exists() for f in pair)), None)
if not pair:
    raise RuntimeError("Install DejaVu Sans, Arial or Liberation Sans to render the documented 46px PDF fixture; a tiny bitmap fallback is not a valid test.")
font = ImageFont.truetype(pair[0], 46)
bold = ImageFont.truetype(pair[1], 54)
print("PDF fixture font:", pair[0])

lines = [
    (True,  "NORTHGATE LEGAL LLP"),
    (False, "Case No. 2024-CV-00456"),
    (False, "Claim amount: GBP 2,500.00"),
    (False, "Hearing date: 14 March 2025"),
    (False, "Reference: NG/2024/0122"),
    (False, "Please quote the reference above in all correspondence."),
]
y = 260
for is_head, text in lines:
    dd.text((200, y), text, fill=(20, 20, 20), font=(bold if is_head else font))
    y += 110 if is_head else 95
page.save("/tmp/letter.pdf", "PDF", resolution=300.0)

print("ok: /tmp/blank.png /tmp/noise.png /tmp/letter.pdf (letter 612x792pt, 46px text)")
