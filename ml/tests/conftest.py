"""
Shared fixtures for the SightLine test suite.

The suite is deliberately FAST: it exercises the physics and the invariants,
not the volume of the training corpus. Anything that would need a trained
checkpoint, a network download, or a full capture sweep is either skipped
or restricted to 2-3 documents per profile.
"""

import sys
from pathlib import Path

import pytest

# Project modules live in ml/src and are plain modules (not an installed
# package), so they must be put on sys.path explicitly. Each test module
# also does this itself, so a test file can be run in isolation.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import capture  # noqa: E402
import corpus  # noqa: E402,F401


@pytest.fixture(scope="session")
def clean_docs():
    """The three real document renders, built once for the whole session."""
    return [(lines, dtype) for lines, dtype in capture.DOCS]


@pytest.fixture(scope="session")
def clean_img():
    """One rendered prescription page, reused by the degradation tests."""
    return capture.render_document(capture.PRESCRIPTION, seed=7)


@pytest.fixture(scope="session")
def flat_doc():
    """A synthetic high-detail image with no text, for measuring noise."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (120, 320), dtype=np.uint8)
    return Image.fromarray(np.stack([a] * 3, -1), "RGB")
