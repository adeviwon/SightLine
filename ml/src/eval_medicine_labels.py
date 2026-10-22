"""Evaluate the SHIPPED recogniser on REAL MEDICINE LABELS -- the product's actual domain.

WHY THIS EXISTS. Every number this repo publishes was measured on synthetic renders, restaurant
receipts, or bank statements. The product is a medicine-label reader, and it had never been run on
a medicine label. This measures it on real ones and reports whatever comes out.

SOURCE. `twinkle-ai/tw-drug-labels-vision` -- 72,326 real medicine-label images from Taiwan's FDA
leaflet archive, CC-BY-4.0, each carrying the drug's English and Chinese name. Real photographs and
real page renders of real labels, not our simulator.

WHAT IS MEASURED, and the two honest limits of the measurement itself:

  1. latin_any        did the recogniser produce ANY Latin word (>=3 letters) from the page?
                      If not, there was no English on the label to read -- which is a property of
                      the SOURCE, not of the model: these are Taiwanese labels and much of the
                      printed text is Traditional Chinese, which our 37-symbol alphabet cannot
                      represent at all.
  2. name_recoverable did the drug's English name, as printed in the dataset metadata, survive into
                      our read (substring, or >=70% of its tokens)? This is the step the Cantonese
                      output depends on: no name in the read, no Cantonese name to speak.
  3. lexicon_hit      would the shipped zh/en lexicon have found a translatable name in our read?

THE LIMIT OF THE LIMIT: dataset metadata names the drug, but does not guarantee the ENGLISH name is
printed on the page. A miss can therefore mean "we misread it" OR "it was never printed in Latin
script". These are reported separately and NOT conflated -- a miss rate is not an error rate unless
the text was there to read, and only a human looking at the images can settle that.

    env -u PYTHONPATH -u VIRTUAL_ENV ./.venv/bin/python ml/src/eval_medicine_labels.py [N]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

import tesseract_compare as TC          # noqa: E402  (CRNN, detect_lines, deskew, crop_for_recognizer)
import realdata as R                    # noqa: E402  (CHARSET)

OUT = ROOT / "artifacts" / "medicine_labels_eval.json"
SHARD = ("https://huggingface.co/api/datasets/twinkle-ai/tw-drug-labels-vision"
         "/parquet/default/train/%d.parquet")
LEX = ROOT / "app" / "data" / "drug_lexicon.json"

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", str(s).upper())).strip()
tokens = lambda s: [t for t in norm(s).split(" ") if len(t) >= 3]


def name_recovered(gt_name, read_text):
    """Did the drug's English name survive into the read? Conservative on purpose: substring, or
    >=70% of the name's tokens present. Reported per image so a human can look at the misses."""
    gt, rd = norm(gt_name), norm(read_text)
    if not gt or not rd:
        return False
    if gt in rd:
        return True
    gt_t = tokens(gt)
    if not gt_t:
        return False
    got = [t for t in gt_t if t in rd.split(" ")]
    return len(got) / len(gt_t) >= 0.70


# dosage forms / excipients are keyed in the lexicon (real label data) but are not drug names;
# same stop-list the shipped JS matcher uses, kept in sync with say_cantonese.js
NOT_A_NAME = set(("TABLET TABLETS CAPSULE CAPSULES SOLUTION SOLUTIONS SUSPENSION POWDER CREAM "
                  "OINTMENT GEL LIQUID INJECTION DROPS SYRUP GRANULES PATCH SPRAY INHALER WATER SALT").split())


def lexicon_hit(read_text, lex):
    """Would the shipped lexicon find a translatable name in what we read? Longest wins, and a
    dosage form is never accepted -- the bug the first JS matcher had."""
    rd, best = norm(read_text), None
    for k in lex:
        if len(k) < 5 or k in NOT_A_NAME or k.split(" ") and all(w in NOT_A_NAME for w in k.split(" ")):
            continue
        if k in rd and (best is None or len(k) > len(best)):
            best = k
    return best


def main(n=150):
    import fsspec
    import pyarrow.parquet as pq
    from PIL import Image

    lex = json.loads(LEX.read_text(encoding="utf-8"))["pairs"]
    crnn = TC.CRNN()
    print("shipped graph: %s (%.2f MB, dynamic width=%s)"
          % (TC.ONNX.name, crnn.bytes / 1048576, crnn.dynamic))
    print("lexicon: %d pairs | sampling %d real medicine labels\n" % (len(lex), n))

    rows, seen = [], 0
    fh = fsspec.open(SHARD % 0, "rb").open()
    try:
        pf = pq.ParquetFile(fh)
        for batch in pf.iter_batches(batch_size=32,
                                    columns=["images", "drug_name_en", "strength", "source_type"]):
            for r in batch.to_pylist():
                if seen >= n:
                    break
                seen += 1
                # the column is `images`: a LIST of blobs (WEBP), usually one page per record
                imgs = r.get("images") or []
                first = imgs[0] if isinstance(imgs, list) and imgs else imgs
                raw = first.get("bytes") if isinstance(first, dict) else first
                if not raw:
                    continue
                try:
                    im = Image.open(__import__("io").BytesIO(raw)).convert("L")
                except Exception:
                    continue
                # cap the long side: a flat A4 render at full resolution is slow and adds nothing
                w, h = im.size
                if max(w, h) > 2000:
                    sc = 2000.0 / max(w, h)
                    im = im.resize((int(w * sc), int(h * sc)))
                gray = np.asarray(im)
                if im.size[0] > im.size[1]:          # landscape scan -> portrait page
                    gray = np.asarray(im.rotate(90, expand=True))
                try:
                    page, angle = TC.deskew(gray)   # returns (page, estimated angle)
                    boxes = TC.detect_lines(page)
                    crops = [TC.crop_for_recognizer(page, b) for b in boxes]
                    lines = [crnn.read(c) for c in crops]
                except Exception as e:
                    rows.append({"gt_name": r.get("drug_name_en"), "source_type": r.get("source_type"),
                                 "error": repr(e)[:80]})
                    continue
                read = " ".join(lines)
                gt = r.get("drug_name_en") or ""
                latin = len(tokens(read))
                rows.append({
                    "gt_name": gt,
                    "source_type": r.get("source_type"),
                    "strength": r.get("strength"),
                    "bands": len(boxes), "skew": round(float(angle), 1),
                    "latin_words": latin,
                    "latin_any": latin > 0,
                    "name_recoverable": name_recovered(gt, read),
                    "lexicon_hit": lexicon_hit(read, lex),
                    "read_head": read[:150],
                })
            if seen >= n:
                break
    finally:
        fh.close()

    ok = [r for r in rows if "error" not in r]
    latin = [r for r in ok if r["latin_any"]]
    named = [r for r in latin if r["name_recoverable"]]
    lexh = [r for r in latin if r["lexicon_hit"]]
    total = len(ok)
    print("images evaluated:            %d%s" % (total, "" if not (len(rows) - total) else "  (%d unreadable)" % (len(rows) - total)))
    print("produced ANY Latin text:     %d  (%.1f%%)   <- if low, the labels are Chinese, not our error"
          % (len(latin), 100.0 * len(latin) / max(total, 1)))
    print("  of those, drug name recovered: %d  (%.1f%%)" % (len(named), 100.0 * len(named) / max(len(latin), 1)))
    print("  of those, lexicon found a name: %d  (%.1f%%)" % (len(lexh), 100.0 * len(lexh) / max(len(latin), 1)))
    from collections import Counter
    types = Counter(r.get("source_type") or "?" for r in ok)
    print("  by source_type:            " + ", ".join("%s=%d" % kv for kv in types.most_common()))
    for t in types:
        sub = [r for r in latin if (r.get("source_type") or "?") == t]
        tot_t = [r for r in ok if (r.get("source_type") or "?") == t]
        print("    %-10s %3d/%3d produced Latin text" % (t, len(sub), len(tot_t)))
    print("\n  a sample of what the shipped model actually read off real labels:")
    for r in latin[:6]:
        print("    GT %-28s -> %s" % (str(r["gt_name"])[:28], r["read_head"][:70]))

    OUT.write_text(json.dumps({
        "source": "twinkle-ai/tw-drug-labels-vision (CC-BY-4.0, Taiwan FDA leaflet archive)",
        "graph": TC.ONNX.name, "n": total,
        "latin_any": len(latin), "name_recovered": len(named), "lexicon_hit": len(lexh),
        "scope_note": ("Dataset metadata names the drug but does not guarantee the ENGLISH name is "
                       "printed on the page, so a miss may be a misread OR absent Latin text. "
                       "Reported separately; not conflated."),
        "rows": rows,
    }, ensure_ascii=False, indent=1))
    print("\nwrote %s" % OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 150))
