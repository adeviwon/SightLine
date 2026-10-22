"""
Render one sample per (document type x capture profile) for the browser E2E test,
plus the manifest the browser harness scores against.

capture.py's own __main__ only renders index 0 of make_capture_set(1), so every
file it writes is the same prescription. This walks the full DOCS x PROFILES grid
so the browser pipeline can be scored against a real mix, and it BUILDS THE
MANIFEST FROM evaluate.FIELDS / evaluate.DOC_CATEGORY rather than copying them:
a hand-copied ground truth is a typo away from silently scoring the wrong thing,
which is the failure mode test_evaluate.py exists to prevent. The text each
document was rendered from goes in too, so word accuracy and CER can be measured
against what was actually drawn, not against a retyped summary.

Usage (from the project root):
    bash run.sh py app/tools/render_samples.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "ml/src")
import capture   # noqa: E402
import evaluate  # noqa: E402

OUT = Path("/tmp/sightline_samples")
OUT.mkdir(exist_ok=True)

# A representative spread, not all 10 profiles: clean, one realistic handheld
# case, and the two mild degradations. worst_case is excluded because
# evaluate.SUB_HUMAN marks it unreadable to a human, so a miss there is not a
# model failure and asserting on it would be dishonest.
PROFILES = ["studio_clean", "handheld_light", "off_axis", "jpeg_social"]

text_by_doc = {}
for dt_i, (lines, dtype) in enumerate(capture.DOCS):
    text_by_doc[dtype] = " ".join(lines)
    # The same integrity rule the eval harness enforces: a ground-truth field that
    # is not in the rendered text makes every field metric measure a typo.
    body = evaluate.norm(text_by_doc[dtype])
    missing = [f for f in evaluate.FIELDS[dtype] if evaluate.norm(f) not in body]
    if missing:
        raise SystemExit(f"{dtype}: ground-truth fields absent from the rendered "
                         f"text: {missing}")
    for p_i, pname in enumerate(PROFILES):
        seed = 123 * 7919 + p_i * 101 + dt_i
        clean = capture.render_document(lines, seed=seed)
        deg = capture.apply_profile(clean, seed=seed, **capture.CAPTURE_PROFILES[pname])
        out = OUT / f"{dtype}__{pname}.png"
        deg.save(out)
        print(f"  {out}  {deg.size}")

manifest = {
    # WHICH PROFILE PARAMETERS these images were rendered with. Without this the
    # fixtures silently go stale when the profiles are re-tuned, and every benchmark
    # run afterwards measures a distribution the model never trained on.
    "profiles_fingerprint": capture.profiles_fingerprint(),
    "fields": {d: evaluate.FIELDS[d] for d in text_by_doc},
    "category": {d: evaluate.DOC_CATEGORY[d] for d in text_by_doc},
    "text": text_by_doc,
    "profiles": PROFILES,
    "samples": [
        {"file": f"/tmp/sightline_samples/{d}__{p}.png", "doc": d, "profile": p}
        for d in text_by_doc for p in PROFILES
    ],
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("\nwrote manifest: " + str(OUT / "manifest.json"))
print("  from evaluate.FIELDS / evaluate.DOC_CATEGORY, with the rendered text")
