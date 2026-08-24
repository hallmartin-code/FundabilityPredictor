"""Shared fixtures.

The sample decks are generated on demand so a fresh clone can run the suite
without any binary fixtures checked in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_DIR = ROOT / "sample_data"

from analysis.pipeline import analyse, extract_profile  # noqa: E402
from config.settings import load_config  # noqa: E402


def _ensure_samples() -> None:
    expected = [
        SAMPLE_DIR / "northwind_saas_series_a.pptx",
        SAMPLE_DIR / "helix_bio_preseed.pdf",
        SAMPLE_DIR / "sparse_deck.pdf",
        SAMPLE_DIR / "unsupported.txt",
    ]
    if all(path.exists() for path in expected):
        return
    sys.path.insert(0, str(SAMPLE_DIR))
    from generate_samples import main as generate  # noqa: PLC0415

    generate()


@pytest.fixture(scope="session", autouse=True)
def samples() -> None:
    _ensure_samples()


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def saas_deck() -> Path:
    return SAMPLE_DIR / "northwind_saas_series_a.pptx"


@pytest.fixture(scope="session")
def biotech_deck() -> Path:
    return SAMPLE_DIR / "helix_bio_preseed.pdf"


@pytest.fixture(scope="session")
def sparse_deck() -> Path:
    return SAMPLE_DIR / "sparse_deck.pdf"


@pytest.fixture(scope="session")
def unsupported_file() -> Path:
    return SAMPLE_DIR / "unsupported.txt"


@pytest.fixture(scope="session")
def saas_profile(saas_deck):
    return extract_profile(saas_deck)


@pytest.fixture(scope="session")
def saas_result(saas_deck):
    return analyse(saas_deck)


@pytest.fixture(scope="session")
def biotech_result(biotech_deck):
    return analyse(biotech_deck)


@pytest.fixture(scope="session")
def sparse_result(sparse_deck):
    return analyse(sparse_deck)
