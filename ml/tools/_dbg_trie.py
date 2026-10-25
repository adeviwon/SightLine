import json
import sys

sys.path.insert(0, "ml/src")
import word_lm as W  # noqa: E402

lm = W.WordLM(W.load_lexicon())
print("CAY in trie:", lm._node("CAY") is not None)
print("CA children:", sorted(lm._node("CA")[2].keys())[:20])
W.export_trie(lm, "/tmp/t.json")
raw = json.load(open("/tmp/t.json"))["root"]
print("root children:", sorted(raw["c"].keys())[:10], "C in root:", "C" in raw["c"])
ca = raw["c"]["C"]["c"].get("A")
print("C>A node:", None if ca is None else sorted(ca["c"].keys())[:20])
tj = W.load_trie_json("/tmp/t.json")
print("tj._node('CA'):", tj._node("CA"))
print("tj logp CA,Y:", tj.logp("CA", "Y"))
