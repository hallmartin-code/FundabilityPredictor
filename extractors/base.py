"""Shared types and the extractor registry.

Adding a new input format means writing a subclass of
:class:`DocumentExtractor` and registering its extensions - nothing downstream
needs to change.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from models.evidence import ExtractionQuality
from models.profile import SourceDocument

LOGGER = logging.getLogger(__name__)

#: Below this many machine-readable characters a page is treated as image-based.
IMAGE_PAGE_CHAR_THRESHOLD = 40


class ExtractionError(RuntimeError):
    """Raised when a document cannot be read at all."""


class UnsupportedFormatError(ExtractionError):
    """Raised for a file extension no registered extractor handles."""


@dataclass
class Page:
    """One page (PDF) or slide (PPT/PPTX) of the source document."""

    number: int
    text: str = ""
    tables: list[list[list[str]]] = field(default_factory=list)
    notes: str = ""
    title: str = ""

    @property
    def char_count(self) -> int:
        return len(self.text.strip())

    @property
    def is_image_based(self) -> bool:
        """True when the page carries almost no machine-readable text."""
        return self.char_count < IMAGE_PAGE_CHAR_THRESHOLD

    def table_text(self) -> str:
        """Tables flattened to text so field patterns can search them too."""
        lines: list[str] = []
        for table in self.tables:
            for row in table:
                cells = [str(c).strip() for c in row if c and str(c).strip()]
                if cells:
                    lines.append(" | ".join(cells))
        return "\n".join(lines)

    def searchable_text(self) -> str:
        """Everything on the page that a pattern should be able to match."""
        parts = [self.text, self.table_text(), self.notes]
        return "\n".join(p for p in parts if p and p.strip())


@dataclass
class ExtractedDocument:
    """The full parsed document, before any field interpretation."""

    source: SourceDocument
    pages: list[Page] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(p.searchable_text() for p in self.pages)

    def quality(self) -> ExtractionQuality:
        return ExtractionQuality(
            pages_total=len(self.pages),
            pages_with_text=sum(1 for p in self.pages if not p.is_image_based),
            image_based_pages=[p.number for p in self.pages if p.is_image_based],
            total_characters=sum(p.char_count for p in self.pages),
            tables_found=sum(len(p.tables) for p in self.pages),
            notes_found=sum(1 for p in self.pages if p.notes.strip()),
        )

    def text_by_page(self) -> dict[int, str]:
        return {p.number: p.searchable_text() for p in self.pages}


class DocumentExtractor(ABC):
    """Base class for format-specific extractors."""

    #: Lower-case extensions, with the leading dot.
    extensions: tuple[str, ...] = ()
    file_type: str = "unknown"

    @abstractmethod
    def parse(self, path: Path) -> list[Page]:
        """Return the document's pages. Raise ExtractionError on failure."""

    def extract(self, path: Path) -> ExtractedDocument:
        pages = self.parse(path)
        source = SourceDocument(
            path=str(path),
            filename=path.name,
            file_type=self.file_type,
            page_count=len(pages),
            sha256=_sha256(path),
        )
        doc = ExtractedDocument(source=source, pages=pages)
        quality = doc.quality()
        if quality.image_based_pages:
            LOGGER.warning(
                "%s: %d of %d pages are image-based; extraction confidence reduced.",
                path.name,
                len(quality.image_based_pages),
                quality.pages_total,
            )
        return doc


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:  # pragma: no cover - unreadable file already raised earlier
        return None
