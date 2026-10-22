"""A rendered artifact must carry a stamp matching the CURRENT profile parameters.

THE INCIDENT THIS EXISTS FOR. The browser bench fixtures were rendered at 07:37 and
`CAPTURE_PROFILES` was re-tuned at 23:54 the same day. From then on the bench scored every
model against images made with the OLD profiles -- measured 17-67x harsher than real
photographs -- while the trainer learned the new ones. A retrained model that was hugely
better on held-out crops (+40.4 points exact on the render corpus) measured as a 2.7-point
REGRESSION on the bench, one profile collapsing from 46.5% to 19.4%: the bench was a stale
picture of a world that no longer existed, and the number looked exactly like a fact.

Nothing catches that by reading the code. The profile parameters are correct; the images
were correct when they were made. The only thing that catches it is a stamp recorded at
render time and compared at use time -- which is this file. The same trap bit the browser
width fixture and the legacy ONNX graph in the same session, so the rule is now general:
an artifact derived from a moving parameter carries a stamp, and a gate compares it.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))
import capture as C  # noqa: E402

FIXTURES = Path("/tmp/sightline_samples/manifest.json")
CORPUS_INDEX = ROOT / "data" / "render_corpus" / "index.json"


def _check(path, rerender, what):
    if not path.exists():
        pytest.skip("%s not present on this machine (%s)" % (what, path))
    doc = json.loads(path.read_text())
    stamp = doc.get("profiles_fingerprint")
    current = C.profiles_fingerprint()
    assert stamp, (
        "%s carries NO profiles_fingerprint, so it cannot be shown to match the current\n"
        "capture profiles. Render it again: %s" % (what, rerender))
    assert stamp == current, (
        "STALE %s: it was rendered with profile parameters %s, but CAPTURE_PROFILES is\n"
        "now %s. Benchmarking against it measures a distribution the model never trained\n"
        "on. Render it again: %s" % (what.upper(), stamp, current, rerender))


def test_bench_fixtures_match_the_current_profiles():
    """The 12 documents the WebKit bench scores on."""
    _check(FIXTURES, "bash run.sh py app/tools/render_samples.py", "bench fixture manifest")


def test_render_corpus_matches_the_current_profiles():
    """The training corpus. A rebuild after a re-tune is a different distribution, so a
    before/after across it measures the corpus change and not the model."""
    _check(CORPUS_INDEX, "bash run.sh render-corpus", "render corpus index")
