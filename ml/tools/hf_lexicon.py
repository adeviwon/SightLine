"""Harvest a shippable zh/en medicine-name lexicon.

Source: `twinkle-ai/tw-drug-labels-vision` -- 72,326 real medicine-label images from Taiwan's
FDA leaflet archive, released **CC-BY-4.0**, pairing `drug_name_en` with `drug_name_zh`.

WHY: the refined idea is "photograph the label, hear it in Cantonese". Our recogniser reads the
ENGLISH on the label (the `0-9 A-Z` alphabet), so the missing piece on the output side is a
bilingual name — and this is a real, licensed, culturally-correct one rather than a guess.

Reads ONLY the text columns, so the images are never downloaded.
    env -u PYTHONPATH -u VIRTUAL_ENV ./.venv/bin/python hf_lexicon.py
"""
import json
import re
import sys

import fsspec
import pyarrow.parquet as pq

BASE = "https://huggingface.co/api/datasets/twinkle-ai/tw-drug-labels-vision/parquet/default/train/%d.parquet"
OUT = "/home/ubuntu/Projects/sightline/app/data/drug_lexicon.json"
COLS = ["drug_name_en", "drug_name_zh", "strength"]

norm = lambda s: re.sub(r"\s+", " ", str(s or "")).strip()
# a shippable key: what our recogniser can actually emit (upper-case alnum + space)
key = lambda s: re.sub(r"[^A-Z0-9 ]", " ", norm(s).upper()).strip()


def main(max_shards=40):
    pairs, seen = {}, 0
    for shard in range(max_shards):
        url = BASE % shard
        try:
            fh = fsspec.open(url, "rb").open()
        except Exception as e:
            print("shard %d unavailable (%r) -- stopping" % (shard, str(e)[:40]))
            break
        try:
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches(batch_size=2048, columns=COLS):
                for r in batch.to_pylist():
                    seen += 1
                    en, zh = norm(r.get("drug_name_en")), norm(r.get("drug_name_zh"))
                    k = key(en)
                    if not k or not zh or len(k) < 3 or len(k) > 48:
                        continue
                    if k not in pairs:
                        pairs[k] = {"zh": zh, "strength": norm(r.get("strength")) or None}
        except Exception as e:
            print("shard %d read error: %r" % (shard, str(e)[:60]))
        finally:
            fh.close()
        if shard % 5 == 0:
            print("  shard %-3d rows=%-7d unique keys=%d" % (shard, seen, len(pairs)), flush=True)
        if seen and shard >= max_shards - 1:
            break
    print("\nrows scanned: %d   unique English keys: %d" % (seen, len(pairs)))
    if not pairs:
        print("NOTHING HARVESTED -- do not ship an empty lexicon")
        return 1
    import pathlib
    p = pathlib.Path(OUT)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "_licence": "CC-BY-4.0 -- twinkle-ai/tw-drug-labels-vision (Taiwan FDA leaflet archive). "
                    "Attribution required; see docs/26_IDEA_REFINED.md.",
        "_note": "English medicine name (as our recogniser would emit it, upper-case) -> Cantonese "
                 "name. Built by ml/tools/hf_lexicon.py; covers names as published, NOT every brand.",
        "pairs": pairs,
    }, ensure_ascii=False, indent=0))
    kb = p.stat().st_size / 1024
    print("wrote %s  (%d pairs, %.0f KB)" % (OUT, len(pairs), kb))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40))
