"""
Measure the JS layout detector's recall on real hand-photographed pages.

Dump each page as raw grayscale plus its transcript line count, then let
app/tools/layout_recall.js run the actual browser code. The point is to test
the detector that will ship, not a Python reimplementation of it -- a Python
mirror could drift from the JS and quietly measure the wrong thing.

Usage:  bash run.sh py ml/src/dump_pages.py <outdir> <n_pages>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realdata as R  # noqa: E402


def main(outdir, n_pages=20):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pages = R.handheld_pages()[:int(n_pages)]
    manifest = []
    for i, (name, jp, lines) in enumerate(pages):
        g = np.asarray(Image.open(jp).convert("L"), np.uint8)
        h, w = g.shape
        raw = out / f"{i:03d}.raw"
        raw.write_bytes(g.tobytes())
        # Transcript lines of >= 3 chars are what the evaluator scores.
        want = [l.strip() for l in lines if len(l.strip()) >= 3]
        manifest.append({
            "i": i, "name": name, "w": w, "h": h,
            "raw": str(raw),
            "want": len(want),
        })
    mf = out / "manifest.json"
    mf.write_text(json.dumps(manifest, indent=1))
    print(f"dumped {len(manifest)} pages to {out}")
    print(f"  total transcript lines: {sum(m['want'] for m in manifest)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pages",
         sys.argv[2] if len(sys.argv) > 2 else 20)