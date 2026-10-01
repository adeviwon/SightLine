"""
Degradation-aware restoration gating.

THE PROBLEM THIS SOLVES
-----------------------
`bash run.sh diagnose` on the trained checkpoint shows:

    identity mean PSNR 20.97 dB   (do nothing)
    model    mean PSNR 20.86 dB   (our restorer)
    vs identity: -0.10 dB  — better on 75% of patches, worse on 25%

Broken down per patch, the picture is much clearer:

    idx 170   identity  1.58 dB  ->  model 20.70 dB   +19.12   (heavily blurred)
    idx 172   identity  4.02 dB  ->  model 22.00 dB   +17.98
    idx  10  identity 100.00 dB  ->  model 50.67 dB   -49.33   (already clean!)

So the restorer is doing exactly what it was trained to do — it is very good
at repairing damaged input — but it is UNCONDITIONAL, so on an already-clean
scan it adds its own artefacts and destroys the image. On a clean document
that is the worst possible time to be lossy, and a clean scan is the demo.

THE FIX
-------
Measure how degraded the input actually is, and blend between the identity and
the full restoration in proportion. A clean image gets weight 0 and passes
through untouched; a badly degraded one gets weight 1 and is fully restored.

Two classical, cheap, well-understood degradation measures:

  sharpness  variance of the Laplacian — falls sharply with blur, is
             insensitive to noise magnitude
  noise      the classic Immerkaer estimator: sigma = sqrt(pi/2) / (6(W-2)(H-2))
             * |sum of the Laplacian response| — robust, no convolution

Both are computed in normalised [0,1] space with no learned parameters, so
they cannot themselves be wrong in a way that harms a good image.

Weights are calibrated so that a real capture scores high and a clean render
scores near zero; see `calibrate()` and the numbers printed by __main__.
"""

import cv2
import numpy as np


def to_gray(img):
    """Accepts a PIL Image or a numpy array; returns float32 [0,1] grayscale."""
    if hasattr(img, "convert"):          # PIL
        a = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    else:
        a = np.asarray(img)
        if a.ndim == 3:
            a = cv2.cvtColor(a.astype(np.float32) / 255.0,
                             cv2.COLOR_BGR2GRAY) if a.shape[2] == 3 else \
                cv2.cvtColor(a.astype(np.float32) / 255.0, cv2.COLOR_RGB2GRAY)
        else:
            a = a.astype(np.float32)
            if a.max() > 1.5:
                a = a / 255.0
    return a


def laplacian(gray):
    """
    3x3 Laplacian with an EXPLICIT kernel and border mode.

    cv2.Laplacian(g, CV_32F, ksize=3) is NOT used: its output depends on
    OpenCV's aperture/kernel selection, and the cross-language parity test
    (app/tools/gating_parity.js) measured it ~11x away from the obvious
    implementation — a constant factor, which means a different kernel rather
    than a different border convention. Spelling the kernel out removes the
    ambiguity permanently and lets the JS port match it exactly.

        [ 0  1  0 ]
        [ 1 -4  1 ]
        [ 0  1  0 ]

    filter2D defaults to correlation; this kernel is symmetric, so correlation
    and convolution coincide and the sign is unambiguous.
    """
    import cv2
    return cv2.filter2D(gray, -1, _LAPLACIAN_K, borderType=cv2.BORDER_REPLICATE)


_LAPLACIAN_K = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def sharpness(gray):
    """Variance of the Laplacian. High = sharp; drops with blur."""
    return float(np.var(laplacian(gray)))


def noise_sigma(gray):
    """
    Immerkaer's fast noise estimator.
    sigma = sqrt(pi/2) / (6(W-2)(H-2)) * sum|Laplacian|
    Robust, convolution-free, and insensitive to structure.
    """
    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    lap = np.abs(laplacian(gray))
    return float(np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2)) * lap.sum())


# Calibration anchors, MEASURED on the synthetic corpus (`bash run.sh gating`
# prints the table).
#
# SCALE MATTERS, AND GETTING IT WRONG SILENTLY BREAKS THE GATE.
# Variance-of-Laplacian and the Immerkaer noise estimator are both
# PER-PIXEL statistics, so they scale with image area: the same content
# measured on a 64x256 patch gives very different numbers from a full
# 900x580 render. Measured on 64x256 PATCHES — the unit the restorer actually
# consumes — the corpus looks like:
#
#   profile          sharpness    contrast   interpretation
#   studio_clean       0.176        0.863    undamaged
#   off_axis           0.156        1.000    sharp (geometry issue, not blur)
#   glossy_glare       0.143        0.843    sharp
#   paper_texture       0.152        0.990    sharp
#   jpeg_social        0.009        0.769    compressed
#   low_light          0.006        0.489    dark + soft
#   handheld_heavy     0.003        0.455    blurred
#   worst_case         0.0003       0.346    unrecoverable
#
# The clean/blurred boundary is between 0.009 and 0.006 — but the gap between
# the sharp group (~0.15) and the blurred group (~0.005) is a factor of ~30,
# so anchors go between those two clusters: CLEAN just under the sharp group,
# BAD just above the blurred group. Midpoint is near 0.08.
SHARP_CLEAN = 0.080        # sharpness >= this  -> weight 0
SHARP_BAD = 0.008          # sharpness <= this  -> weight 1

# Not used to set the weight (see degradation_weight) — Immerkaer's estimator
# is confounded by glyph-edge density on text, reading clean text as noisier
# than blurred text. Retained for telemetry and for a future de-noising gate.
NOISE_CLEAN = 0.044        # measured clean value, for reference
NOISE_BAD = 0.030


def _ramp_up(value, low, high):
    """0.0 at/below `low`, 1.0 at/above `high`. For quantities where HIGH is
    bad — e.g. noise sigma."""
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return float((value - low) / (high - low))


def _ramp_down(value, clean, bad):
    """0.0 at/above `clean`, 1.0 at/below `bad`. For quantities where LOW is
    bad — e.g. sharpness (variance of the Laplacian), which collapses under
    blur. Sharpness spans two orders of magnitude across the capture profiles,
    so a linear ramp between two measured anchors is the right shape."""
    if value >= clean:
        return 0.0
    if value <= bad:
        return 1.0
    return float((clean - value) / (clean - bad))


def contrast(gray):
    """
    Dynamic range (max - min) of the patch.

    Needed because sharpness alone is misleading on a BLANK region: an empty
    strip of paper has near-zero variance-of-Laplacian (no edges at all), so
    `sharpness` reads it as "severely blurred" and the gate would send an
    untouched patch through full restoration — where the model can only add
    artefacts, because there is no structure to recover.

    `content` guards against exactly that: a patch with almost no contrast has
    nothing worth restoring, so it must pass through untouched.
    """
    return float(gray.max() - gray.min())


# Below this dynamic range the patch is treated as blank paper.
CONTENT_MIN = 0.12


def degradation_weight(gray, sharp=None, noise=None):
    """
    How much restoration does this image need? 0.0 = none, 1.0 = full.

    SCOPE — READ THIS BEFORE ADDING TERMS
    --------------------------------------
    The gate is driven by SHARPNESS alone, and the noise estimator is
    reported but deliberately NOT used to set the weight.

    Reason, measured not assumed: on a document full of text, Immerkaer's
    noise estimator is confounded by edge density. It reads CLEAN text as
    *noisier* than blurred text, because the sum-of-absolute-Laplacian it
    depends on is dominated by glyph edges:

        studio_clean   sharpness 1.955   noise 0.150   <- "noisiest"
        worst_case     sharpness 0.004   noise 0.009   <- "cleanest"

    That is backwards. Wiring noise into the weight with a naive ramp marked
    every clean patch as maximally degraded and sent it through full
    restoration, which is precisely the damage the gate exists to prevent.
    Sharpness separates the classes cleanly (two orders of magnitude), so it
    is the only term that decides the weight.

    `noise` stays in the signature and in describe() because it is genuinely
    useful telemetry for the UI ("this photo is grainy") and for future work
    on a de-noising-specific gate. It just must not drive this decision.
    """
    if contrast(gray) < CONTENT_MIN:
        return 0.0
    if sharp is None:
        sharp = sharpness(gray)
    return _ramp_down(sharp, SHARP_CLEAN, SHARP_BAD)


def restore_gated(restored, original, weight):
    """
    Blend a restored image with its original.

    `weight` 0.0 returns `original` untouched; 1.0 returns `restored`.
    Uses a mild gamma so the blend spends more of its range near the damaged
    end, where the model's correction is largest and most valuable.
    """
    if weight <= 0.0:
        return original
    if weight >= 1.0:
        return restored
    w = float(weight) ** 0.7
    r = to_gray(restored)
    o = to_gray(original)
    if r.shape != o.shape:
        return restored
    return (w * r + (1.0 - w) * o)


def describe(gray):
    """Human-readable degradation report, for logging and the demo."""
    s = sharpness(gray)
    n = noise_sigma(gray)
    w = degradation_weight(gray, s, n)
    band = ("clean" if w < 0.15 else
            "mild" if w < 0.45 else
            "moderate" if w < 0.75 else "severe")
    return {"sharpness": s, "noise_sigma": n, "weight": w, "band": band}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import capture
    from train_restorer import BANDS, PATCH_H, PATCH_W
    import seedutil

    print(f"Calibration is measured on {PATCH_H}x{PATCH_W} PATCHES — the unit")
    print(f"the restorer consumes. Full-image numbers are ~3.5x smaller and")
    print(f"would mis-scale every anchor.\n")
    print(f"{'profile':<16} {'sharpness':>10} {'noise':>9} {'contr':>7} "
          f"{'weight':>7}  band")
    print("-" * 66)
    for name, params in capture.CAPTURE_PROFILES.items():
        sh, nz, ct, wt = [], [], [], []
        g = None
        for i in range(3):
            ds = 123 * 7919 + seedutil.name_hash(name) % 100003 + i
            lines, _ = capture.DOCS[i % len(capture.DOCS)]
            clean = capture.render_document(lines, seed=ds)
            deg = capture.apply_profile(clean, seed=ds, **params)
            for band in range(BANDS):
                y0 = 100 + band * PATCH_H
                if y0 + PATCH_H > deg.size[1]:
                    break
                g = to_gray(deg.crop((48, y0, 48 + PATCH_W, y0 + PATCH_H)))
                sh.append(sharpness(g))
                nz.append(noise_sigma(g))
                ct.append(contrast(g))
                wt.append(degradation_weight(g))
        d = describe(g)
        print(f"{name:<16} {np.mean(sh):>10.5f} {np.mean(nz):>9.4f} "
              f"{np.mean(ct):>7.3f} {np.mean(wt):>7.2f}  {d['band']}")
    print(f"\nanchors: SHARP_CLEAN={SHARP_CLEAN} SHARP_BAD={SHARP_BAD}")
    print(f"         NOISE_CLEAN={NOISE_CLEAN} NOISE_BAD={NOISE_BAD} "
          f"CONTENT_MIN={CONTENT_MIN}")
