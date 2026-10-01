"""
Render one sample per (document type x capture profile) for the browser E2E test.

capture.py's own __main__ only renders index 0 of make_capture_set(1), so every
file it writes is the same prescription. This walks the full DOCS x PROFILES grid
so the browser pipeline can be scored against a real mix.

Usage (from the project root):
    bash run.sh shell < app/tools/render_samples.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "ml/src")
import capture  # noqa: E402

OUT = Path("/tmp/sightline_samples")
OUT.mkdir(exist_ok=True)

# A representative spread, not all 10 profiles: clean, one realistic handheld
# case, and the two mild degradations. worst_case is excluded because
# evaluate.SUB_HUMAN marks it unreadable to a human, so a miss there is not a
# model failure and asserting on it would be dishonest.
PROFILES = ["studio_clean", "handheld_light", "off_axis", "jpeg_social"]

for dt_i, (lines, dtype) in enumerate(capture.DOCS):
    for p_i, pname in enumerate(PROFILES):
        seed = 123 * 7919 + p_i * 101 + dt_i
        clean = capture.render_document(lines, seed=seed)
        deg = capture.apply_profile(clean, seed=seed, **capture.CAPTURE_PROFILES[pname])
        out = OUT / f"{dtype}__{pname}.png"
        deg.save(out)
        print(f"  {out}  {deg.size}")

# A manifest the browser harness reads, so expectations cannot drift from the
# renderer: the ground truth comes from evaluate.FIELDS, not from this file.
import json  # noqa: E402
manifest = {
    "fields": {
        "prescription": ["500mg", "400mg"],
        "banking": ["40218877", "40-11-04"],
        "legal": ["2500", "2024-CV-00456"],
    },
    # pipeline category names, from corpus.CATEGORIES
    "category": {
        "prescription": "medical",
        "banking": "banking",
        "legal": "legal",
    },
    "samples": [
        {"file": f"/tmp/sightline_samples/{d}__{p}.png", "doc": d, "profile": p}
        for d in (x[1] for x in capture.DOCS) for p in PROFILES
    ],
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("\nwrote manifest: " + str(OUT / "manifest.json"))
