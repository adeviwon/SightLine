"""Paired end-to-end comparison of two pipeline_raw records, using the PRODUCT's
own metric definitions from bench_hackathon (never a re-implementation)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))
import bench_hackathon as BH  # noqa: E402

files = sys.argv[1:] or ["artifacts/pipeline_raw_pre_wordlm.json",
                         "artifacts/pipeline_raw_wordlm.json"]
for f in files:
    recs = json.loads((ROOT / f).read_text())
    m = BH.aggregate_pipeline(recs)
    print("%-42s fields %-7s word %-7s CER %-7s class %s"
          % (Path(f).name,
             "%.1f%%" % (100 * m["field_recovery"]),
             "%.1f%%" % (100 * m["word_accuracy"]),
             "%.3f" % m["cer"],
             "%.1f%%" % (100 * m["classification"])))

# per-row, on the rows both runs scored (paired by doc+profile)
A = {r["doc"] + "/" + r["profile"]: r for r in json.loads((ROOT / files[0]).read_text())}
B = {r["doc"] + "/" + r["profile"]: r for r in json.loads((ROOT / files[1]).read_text())}
print("\n%-28s %8s %8s %8s" % ("doc/profile", "wa old", "wa new", "delta"))
better = worse = 0
for k in A:
    ia, ib = BH.score_pair(A[k]["expected_text"], A[k]["ocr_text"]), \
             BH.score_pair(B[k]["expected_text"], B[k]["ocr_text"])
    d = ib["word_accuracy"] - ia["word_accuracy"]
    better += d > 0.001
    worse += d < -0.001
    print("%-28s %7.0f%% %7.0f%% %+7.0f%%" % (k[:28], 100 * ia["word_accuracy"],
                                              100 * ib["word_accuracy"], 100 * d))
print("\npages better: %d   worse: %d   unchanged: %d" % (better, worse, len(A) - better - worse))
