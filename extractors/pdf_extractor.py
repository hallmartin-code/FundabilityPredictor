"""PDF ingestion via pdfplumber, with a pypdf fallback."""

from __future__ import annotations

import logging
from pathlib import Path

from extractors.base import DocumentExtractor, ExtractionError, Page

LOGGER = logging.getLogger(__name__)


class PdfExtractor(DocumentExtractor):
    extensions = (".pdf",)
    file_type = "pdf"

    def parse(self, path: Path) -> list[Page]:
        pages = self._parse_with_pdfplumber(path)
        if pages is not None:
            return pages
        LOGGER.warning("pdfplumber failed on %s; falling back to pypdf.", path.name)
        return self._parse_with_pypdf(path)

    # ------------------------------------------------------------------ impl
    def _parse_with_pdfplumber(self, path: Path) -> list[Page] | None:
        try:
            import pdfplumber
        except ImportError:  # pragma: no cover - dependency is declared
            LOGGER.warning("pdfplumber is not installed.")
            return None

        pages: list[Page] = []
        try:
            with pdfplumber.open(str(path)) as pdf:
                for index, page in enumerate(pdf.pages, start=1):
                    text = ""
                    tables: list[list[list[str]]] = []
                    try:
                        text = page.extract_text() or ""
                    except Exception as exc:  # noqa: BLE001 - per-page resilience
                        LOGGER.debug("Text extraction failed on page %d: %s", index, exc)
                    try:
                        for table in page.extract_tables() or []:
                            cleaned = [
                                [(cell or "").strip() for cell in row]
                                for row in table
                                if any(cell for cell in row)
                            ]
                            if cleaned:
                                tables.append(cleaned)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.debug("Table extraction failed on page %d: %s", index, exc)
                    pages.append(
                        Page(
                            number=index,
                            text=text,
                            tables=tables,
                            title=_first_line(text),
                        )
                    )
        except Exception as exc:  # noqa: BLE001 - fall through to pypdf
            LOGGER.debug("pdfplumber could not open %s: %s", path.name, exc)
            return None
        return pages

    def _parse_with_pypdf(self, path: Path) -> list[Page]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ExtractionError(
                "No PDF backend available. Install pdfplumber or pypdf."
            ) from exc
        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(f"Could not read PDF {path.name}: {exc}") from exc

        pages: list[Page] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("pypdf text extraction failed on page %d: %s", index, exc)
                text = ""
            pages.append(Page(number=index, text=text, title=_first_line(text)))
        if not pages:
            raise ExtractionError(f"{path.name} contains no readable pages.")
        return pages


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return ""
