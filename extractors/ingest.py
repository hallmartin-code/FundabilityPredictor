"""Format dispatch: pick the right extractor for a file."""

from __future__ import annotations

import logging
from pathlib import Path

from extractors.base import (
    DocumentExtractor,
    ExtractedDocument,
    ExtractionError,
    UnsupportedFormatError,
)
from extractors.pdf_extractor import PdfExtractor
from extractors.ppt_extractor import LegacyPptExtractor, PptxExtractor

LOGGER = logging.getLogger(__name__)

#: Register new formats here; the CLI help and error messages read from this.
EXTRACTORS: tuple[DocumentExtractor, ...] = (
    PdfExtractor(),
    PptxExtractor(),
    LegacyPptExtractor(),
)


def supported_extensions() -> tuple[str, ...]:
    seen: list[str] = []
    for extractor in EXTRACTORS:
        for ext in extractor.extensions:
            if ext not in seen:
                seen.append(ext)
    return tuple(seen)


def extractor_for(path: Path) -> DocumentExtractor:
    suffix = path.suffix.lower()
    for extractor in EXTRACTORS:
        if suffix in extractor.extensions:
            return extractor
    raise UnsupportedFormatError(
        f"Unsupported file type {suffix or '(none)'!r} for {path.name}. "
        f"Supported formats: {', '.join(supported_extensions())}."
    )


def ingest(path: str | Path) -> ExtractedDocument:
    """Read a deck into an :class:`ExtractedDocument`."""
    p = Path(path)
    if not p.exists():
        raise ExtractionError(f"File not found: {p}")
    if not p.is_file():
        raise ExtractionError(f"Not a file: {p}")
    if p.stat().st_size == 0:
        raise ExtractionError(f"{p.name} is empty.")

    extractor = extractor_for(p)
    LOGGER.info("Ingesting %s with %s", p.name, type(extractor).__name__)
    document = extractor.extract(p)
    if not document.pages:
        raise ExtractionError(f"{p.name} produced no readable pages.")
    LOGGER.info(
        "Extracted %d pages (%d characters) from %s",
        len(document.pages),
        sum(pg.char_count for pg in document.pages),
        p.name,
    )
    return document
