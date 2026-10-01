"""
SightLine ML — Degradation simulation matched to real phone-camera capture.

The old synthdata used a fixed 8-condition gate. This module is built around
what actually happens when a blind or low-vision user holds a phone over a
document: shaky hands cause MOTION blur (directional, not isotropic), the
lens sits close so DEPTH OF FIELD is shallow, and the document is lit by
whatever room light exists — which produces a hard-edged shadow of the hand
and phone across the page.

Every degradation here has a physical analogue:

  motion      directional line kernel, angle = wrist angle
  defocus     circular bokeh kernel (real lens, not Gaussian)
  hand shadow hard-edged polygon from a virtual hand position
  perspective homography from an off-axis camera pose
  sensor noise   Poisson-Gaussian, ISO-dependent (luminance-dependent)
  jpeg        real JPEG round-trip at a quality the share-sheet would pick

`CAPTURE_PROFILES` are the conditions a judge will actually reproduce when
they hold their own phone up. The eval harness runs every profile.
"""

import io
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seedutil  # noqa: E402  — stable cross-process seed derivation

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


def _font(size=20, bold=False):
    paths = [FONT_CANDIDATES[1]] + FONT_CANDIDATES if bold else FONT_CANDIDATES
    for fp in paths:
        try:
            return ImageFont.truetype(fp, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ── Document content ────────────────────────────────────────────────────

PRESCRIPTION = [
    "PRESCRIPTION", "Patient: Jane Doe   DOB: 04/02/1979", "Date: 01/10/2026",
    "PRESCRIBER: Dr A. Ahmed, MBBS", "Pharmacy: Northgate Pharmacy",
    "Rx  Amoxicillin 500mg capsules",
    "Take one capsule three times daily with food",
    "Complete the full 7 day course",
    "Rx  Ibuprofen 400mg tablets",
    "Every 6 to 8 hours as required for pain",
    "Maximum 1200mg in 24 hours",
    "WARNING: may cause drowsiness",
    "Do not drive or operate machinery",
    "Keep out of reach of children",
    "Pharmacist: R. Silva  GPhC 20441",
]

BANK_STATEMENT = [
    "HSBC UK BANK PLC", "Statement of Account", "Account Holder: John Smith",
    "Account Number: 40218877", "Sort Code: 40-11-04", "Period: 01/03 - 31/03/2026",
    "Opening Balance        GBP 1,204.33",
    "01/03 SALARY CREDIT   GBP 2,900.00",
    "01/03 RENT DD 1100     GBP -1,100.00",
    "08/03 TESCO STORES     GBP -44.19",
    "14/03 BRITISH GAS DD   GBP -89.99",
    "19/03 ATM WITHDRAWAL   GBP -200.00",
    "22/03 CASHBACK         GBP +12.40",
    "Deposits total         GBP 2,912.40",
    "Withdrawals total      GBP -1,434.18",
    "Closing Balance        GBP 2,682.55",
    "Card ending 4521 available funds 2,100.00",
]

LEASE_AGREEMENT = [
    "LEASE AGREEMENT", "This agreement is made on 15 January 2024",
    "BETWEEN: The Landlord (1) and The Tenant (2)",
    "PROPERTY: 123 Baker Street, London NW1 6XE",
    "TERM: 24 months from 1 February 2024",
    "Clause 3.2  Rent",
    "The Tenant shall pay rent of 2,500 per month",
    "in advance on the first day of each month",
    "Clause 5.1  Termination",
    "Either party may terminate with two months",
    "written notice served on the other party",
    "Clause 7.4  Deposit",
    "A deposit of 5,000 is held and protected",
    "SIGNED on behalf of both parties",
    "Case Reference: 2024-CV-00456",
]

DOCS = [(PRESCRIPTION, "prescription"), (BANK_STATEMENT, "banking"),
        (LEASE_AGREEMENT, "legal")]


def render_document(lines, width=900, seed=None, bg=248, ink=28, scale=1.0):
    """Render a clean document with jittered baseline/spacing (seeded)."""
    rng = random.Random(seed)
    fs = max(12, int(19 * scale))
    lh = int(30 * scale)
    height = int(90 + len(lines) * lh + 40)
    img = Image.new("RGB", (width, height), (bg, bg, bg))
    d = ImageDraw.Draw(img)
    head, body = _font(fs + 2, True), _font(fs)
    y = int(28 * scale)
    for i, line in enumerate(lines):
        f = head if i == 0 else body
        x = 48 + rng.randint(-3, 3)
        d.text((x, y), line, fill=(ink, ink, ink), font=f)
        y += lh
    return img


# ── Physical degradations ───────────────────────────────────────────────

def _kernel(img_filter, k):
    """
    Apply a convolution kernel via OpenCV.

    PIL's built-in ImageFilter.Kernel caps the kernel at 5x5 and silently
    raises "bad kernel size" for anything larger, which rules out the long
    motion-blur kernels that shaky-hand capture actually produces. OpenCV
    handles arbitrary kernel sizes and is an order of magnitude faster.
    """
    import cv2
    import numpy as np
    size = k.size[0]
    px = k.load()
    kern = np.asarray([[px[x, y] / 255.0 for x in range(size)]
                       for y in range(size)], dtype=np.float32)
    kern /= max(kern.sum(), 1e-6)
    a = np.asarray(img_filter.convert("RGB"))
    return Image.fromarray(cv2.filter2D(a, -1, kern, borderType=cv2.BORDER_REPLICATE))


def motion_blur(img, length=9, angle=25.0):
    """Directional blur from a shaky hand. Real motion blur is directional."""
    if length <= 1:
        return img
    rad = math.radians(angle)
    k = Image.new("L", (length, length), 0)
    d = ImageDraw.Draw(k)
    cx = cy = (length - 1) / 2
    d.line([(cx - math.cos(rad) * length / 2, cy - math.sin(rad) * length / 2),
            (cx + math.cos(rad) * length / 2, cy + math.sin(rad) * length / 2)],
           fill=255, width=1)
    k = k.filter(ImageFilter.GaussianBlur(0.6))
    return _kernel(img.convert("RGB"), k)


def defocus_blur(img, radius=2.2):
    """Circular bokeh disc — a real lens, not a Gaussian."""
    size = max(3, int(radius * 4) | 1)
    k = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(k)
    r = size / 2 - 1
    d.ellipse([size / 2 - r, size / 2 - r, size / 2 + r, size / 2 + r], fill=255)
    k = k.filter(ImageFilter.GaussianBlur(0.5))
    return _kernel(img, k)


def sensor_noise(img, iso=800, seed=None):
    """Poisson-Gaussian sensor noise. Scales with luminance and ISO."""
    rng = np.random.default_rng(seed)
    a = np.asarray(img).astype(np.float32) / 255.0
    shot = np.sqrt(np.clip(a, 0, 1) / max(iso, 1) * 0.02)
    read = 0.004 * math.log2(max(iso, 100) / 100)
    n = rng.normal(0, 1, a.shape).astype(np.float32) * (shot + read)
    # np.rint, not .astype(np.uint8): the cast TRUNCATES toward zero, which
    # would subtract a systematic 0.5 grey levels from every pixel and give
    # the restorer a brightness shift to learn instead of a noise inverse.
    return Image.fromarray(np.clip(np.rint((a + n) * 255), 0, 255).astype(np.uint8))


def hand_shadow(img, seed=None, strength=0.34):
    """Hard-edged shadow from the phone-holder's hand across the page."""
    rng = random.Random(seed)
    w, h = img.size
    ov = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(ov)
    cx = rng.randint(-int(w * 0.1), int(w * 0.4))
    cy = rng.randint(-int(h * 0.1), int(h * 0.45))
    rx, ry = int(w * rng.uniform(0.22, 0.40)), int(h * rng.uniform(0.16, 0.30))
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(int(255 * strength),) * 3)
    for f in range(4):
        a0 = rng.uniform(0, math.pi)
        d.line([(cx, cy), (cx + math.cos(a0) * rx * 2.2,
                            cy + math.sin(a0) * ry * 1.8)],
               fill=(int(255 * strength * 0.8),) * 3, width=int(ry * 0.35))
    ov = ov.filter(ImageFilter.GaussianBlur(radius=rng.uniform(3, 9)))
    return Image.blend(img, ov, rng.uniform(0.6, 0.95))


def perspective(img, strength=0.035, seed=None):
    """Off-axis camera pose → homography warp."""
    rng = random.Random(seed)
    w, h = img.size
    s = strength
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(rng.uniform(0, s) * w, rng.uniform(0, s) * h),
           (w - rng.uniform(0, s) * w, rng.uniform(0, s) * h),
           (w - rng.uniform(0, s) * w, h - rng.uniform(0, s) * h),
           (rng.uniform(0, s) * w, h - rng.uniform(0, s) * h)]
    coeffs = _perspective_coeffs(dst, src)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         Image.BICUBIC, fillcolor=(205, 205, 205))


def _perspective_coeffs(dst, src):
    import numpy as np
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        B += [u, v]
    res = np.linalg.solve(np.asarray(A, float), np.asarray(B, float))
    return res.tolist()


def lighting_gradient(img, strength=0.35, seed=None):
    """Room light falls off across the page — not flat."""
    rng = random.Random(seed)
    w, h = img.size
    a = np.asarray(img).astype(np.float32)
    ang = rng.uniform(0, 2 * math.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = (xx / w) * math.cos(ang) + (yy / h) * math.sin(ang)
    proj = (proj - proj.min()) / max(proj.max() - proj.min(), 1e-6)
    a *= (1.0 - strength * proj)[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def jpeg(img, quality=62):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def glare(img, strength=0.5, seed=None):
    """Specular hotspot from a flash or overhead light on glossy paper."""
    rng = random.Random(seed)
    w, h = img.size
    ov = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(ov)
    cx, cy = rng.randint(int(w * .2), int(w * .8)), rng.randint(int(h * .2), int(h * .7))
    r = int(min(w, h) * rng.uniform(0.12, 0.30))
    for i in range(6, 0, -1):
        rr = int(r * i / 6)
        v = int(255 * strength * (1 - i / 8))
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(v, v, v))
    ov = ov.filter(ImageFilter.GaussianBlur(radius=r * 0.35))
    return ImageChops_add(img, ov)


from PIL import ImageChops  # noqa: E402


def ImageChops_add(a, b):
    return ImageChops.add(a, b)


# ── Capture profiles (what a judge reproduces) ──────────────────────────

CAPTURE_PROFILES = {
    "studio_clean":    dict(),
    "handheld_light":  dict(defocus=1.6, motion=7, motion_angle=22, iso=400),
    "handheld_heavy":  dict(defocus=2.4, motion=11, motion_angle=35, iso=1200),
    "low_light":       dict(defocus=2.0, motion=9, iso=3200, light=0.55, noise_iso=6400),
    "hand_shadow":     dict(hand_shadow=True, light=0.30),
    "off_axis":        dict(perspective=0.045, rot=4.0),
    "glossy_glare":    dict(glare=True, light=0.25),
    "jpeg_social":     dict(jpeg=55, defocus=1.2),
    "paper_texture":   dict(noise_iso=1600, light=0.25, jpeg=70),
    "worst_case":      dict(defocus=3.0, motion=13, iso=3200, hand_shadow=True,
                            light=0.5, noise_iso=6400, perspective=0.05, rot=6.0,
                            glare=True, jpeg=50),
}


def apply_profile(img, seed=None, **p):
    """Apply a capture profile in the physical order a camera produces it."""
    out = img.convert("RGB")
    if p.get("hand_shadow"):
        out = hand_shadow(out, seed=seed)
    if p.get("perspective"):
        out = perspective(out, p["perspective"], seed=seed)
    if p.get("rot"):
        out = out.rotate(p["rot"], fillcolor=(205, 205, 205), expand=False,
                         resample=Image.BICUBIC)
    if p.get("light") is not None:
        out = lighting_gradient(out, p["light"], seed=seed)
    if p.get("defocus"):
        out = defocus_blur(out, p["defocus"])
    if p.get("motion"):
        out = motion_blur(out, p["motion"], p.get("motion_angle", 25.0))
    if p.get("glare"):
        out = glare(out, seed=seed)
    iso = p.get("noise_iso") or p.get("iso")
    if iso:
        out = sensor_noise(out, iso, seed=seed)
    if p.get("jpeg"):
        out = jpeg(out, p["jpeg"])
    return out


def make_capture_set(n=12, seed=123):
    """Yield (clean, degraded, profile, doc_type) tuples for the eval gate."""
    for name, params in CAPTURE_PROFILES.items():
        for i in range(n):
            s = seed * 7919 + seedutil.name_hash(name) % 100003 + i
            lines, dtype = DOCS[i % len(DOCS)]
            clean = render_document(lines, seed=s)
            deg = apply_profile(clean, seed=s, **params)
            yield {"profile": name, "doc_type": dtype, "clean": clean,
                   "degraded": deg, "seed": s}


def to_patch(img, x=48, y=100, w=256, h=64, y0=None):
    """Crop a 64x256 grayscale float32 patch in [0,1].

    `y0` is an alias for `y` so callers can read `to_patch(img, y0=...)` as
    "this text band", which is clearer than a bare pixel offset.
    """
    if y0 is not None:
        y = y0
    return np.asarray(img.convert("L").crop((x, y, x + w, y + h)),
                      dtype=np.float32) / 255.0


if __name__ == "__main__":
    import json
    print(json.dumps({"profiles": list(CAPTURE_PROFILES),
                      "docs": [d[1] for d in DOCS]}, indent=2))
    for d in make_capture_set(1):
        d["degraded"].save(f"/tmp/sample_{d['profile']}.png")
        print(f"  wrote /tmp/sample_{d['profile']}.png  {d['degraded'].size}")
