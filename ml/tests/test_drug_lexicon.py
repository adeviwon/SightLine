"""The zh/en medicine-name lexicon is a shipped asset with a licence and a hard invariant.

WHY A TEST AND NOT A CONVENTION. `app/data/drug_lexicon.json` is 1.7 MB of third-party data whose
use is only lawful with attribution, and whose every key must be something the recogniser can
actually emit -- a key outside the product's `0-9 A-Z space` alphabet is dead weight that can never
match, and a missing attribution field is a licence problem, not a bug. Both are silent failures,
so both are asserted.
"""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEX = ROOT / "app" / "data" / "drug_lexicon.json"
SW = ROOT / "app" / "sw.js"
APP = ROOT / "app" / "js" / "app.js"
SAY = ROOT / "app" / "js" / "say_cantonese.js"


class TestDrugLexicon(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(LEX.read_text(encoding="utf-8"))
        cls.pairs = cls.doc["pairs"]

    def test_licence_and_attribution_are_recorded(self):
        """CC-BY-4.0 requires attribution; the file must carry it, not just the docs."""
        lic = self.doc.get("_licence", "")
        self.assertIn("CC-BY-4.0", lic, "the licence field must name CC-BY-4.0")
        self.assertIn("tw-drug-labels-vision", lic, "attribution must name the source dataset")

    def test_the_lexicon_is_not_empty_or_vestigial(self):
        self.assertGreater(len(self.pairs), 10000, "a lexicon this small means the harvest failed")

    def test_every_key_is_inside_the_recognisers_alphabet(self):
        """The model emits 0-9 A-Z and space. A key with punctuation or lower case can never be
        matched, so it would be silent dead weight in the payload."""
        bad = [k for k in self.pairs if re.search(r"[^A-Z0-9 ]", k)]
        self.assertEqual(bad, [], "keys outside the alphabet cannot ever match: %r" % bad[:5])

    def test_every_key_is_a_usable_length(self):
        short = [k for k in self.pairs if len(k) < 3]
        long_ = [k for k in self.pairs if len(k) > 30]
        self.assertEqual(short, [], "too short to be a name: %r" % short[:5])
        self.assertEqual(long_, [], "past the 30-char cap: %r" % long_[:5])

    def test_every_entry_has_a_cantonese_name(self):
        empty = [k for k, v in self.pairs.items() if not (v.get("zh") or "").strip()]
        self.assertEqual(empty, [], "entries with no zh value: %r" % empty[:5])

    def test_the_asset_is_not_shipped_orphaned(self):
        """Precached by the service worker AND fetched by the app, or it ships unused."""
        self.assertIn("drug_lexicon.json", SW.read_text(encoding="utf-8"),
                      "the lexicon must be in the service worker precache")
        self.assertIn("drug_lexicon.json", APP.read_text(encoding="utf-8"),
                      "the app must fetch the lexicon")

    def test_the_refusal_property_is_intact(self):
        """The safety case: the module must not name a medicine it cannot translate."""
        src = SAY.read_text(encoding="utf-8")
        self.assertIn("NOT_A_NAME", src, "the dosage-form stop-list must be present")
        self.assertIn("ok: false", src, "an unreadable label must return ok:false, not a guess")


if __name__ == "__main__":
    unittest.main()
