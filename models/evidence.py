"""Evidence-carrying values.

Every extracted datum in this system is wrapped in a :class:`FieldValue` so
that the report can always answer "where did this number come from, and is it
something the deck said, something we computed, or something we guessed?".
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ValueStatus(str, Enum):
    """Provenance of a value.

    EXPLICIT   - stated in the source document
    CALCULATED - derived arithmetically from explicit values
    ESTIMATED  - inferred by the model from weaker evidence
    MISSING    - not present in the source and not derivable
    """

    EXPLICIT = "EXPLICIT"
    CALCULATED = "CALCULATED"
    ESTIMATED = "ESTIMATED"
    MISSING = "MISSING"


#: Statuses that represent an actual known value.
KNOWN_STATUSES = (ValueStatus.EXPLICIT, ValueStatus.CALCULATED, ValueStatus.ESTIMATED)

#: Relative trust in each provenance class, used by the confidence model.
EVIDENCE_WEIGHT: dict[ValueStatus, float] = {
    ValueStatus.EXPLICIT: 1.0,
    ValueStatus.CALCULATED: 0.85,
    ValueStatus.ESTIMATED: 0.45,
    ValueStatus.MISSING: 0.0,
}


class FieldValue(BaseModel):
    """A single extracted field together with its provenance."""

    field: str
    value: Any | None = None
    status: ValueStatus = ValueStatus.MISSING
    confidence: float = 0.0
    source_page: int | None = None
    source_text: str | None = None
    unit: str | None = None
    note: str | None = None

    #: True when a human corrected the value after extraction. The
    #: automatically extracted value is preserved in ``extracted_value`` so the
    #: two are never conflated in the audit trail.
    overridden: bool = False
    extracted_value: Any | None = None
    extracted_status: ValueStatus | None = None

    @property
    def is_known(self) -> bool:
        return self.status in KNOWN_STATUSES and self.value is not None

    @property
    def is_missing(self) -> bool:
        return not self.is_known

    def display(self, missing: str = "Not Provided") -> str:
        """Human-readable rendering, never fabricating a value."""
        if not self.is_known:
            return missing
        return str(self.value)

    @classmethod
    def missing_field(cls, name: str, note: str | None = None) -> FieldValue:
        return cls(field=name, value=None, status=ValueStatus.MISSING, confidence=0.0, note=note)

    def with_override(self, new_value: Any, note: str | None = None) -> FieldValue:
        """Return a copy carrying a human override, preserving the original."""
        return self.model_copy(
            update={
                "value": new_value,
                "status": ValueStatus.EXPLICIT,
                "confidence": 1.0,
                "overridden": True,
                "extracted_value": self.value if not self.overridden else self.extracted_value,
                "extracted_status": self.status if not self.overridden else self.extracted_status,
                "note": note or "Value supplied by user override.",
                "source_text": None if not self.overridden else self.source_text,
            }
        )


class Inconsistency(BaseModel):
    """A contradiction found in the extracted data.

    Inconsistencies are surfaced for human review. The pipeline never silently
    "fixes" a contradiction in the source material.
    """

    code: str
    message: str
    fields: list[str] = Field(default_factory=list)
    severity: str = "REVIEW"


class ExtractionQuality(BaseModel):
    """Document-level signal about how well the source could be read."""

    pages_total: int = 0
    pages_with_text: int = 0
    image_based_pages: list[int] = Field(default_factory=list)
    total_characters: int = 0
    tables_found: int = 0
    notes_found: int = 0

    @property
    def image_based_ratio(self) -> float:
        if self.pages_total == 0:
            return 0.0
        return len(self.image_based_pages) / self.pages_total

    @property
    def is_image_heavy(self) -> bool:
        return self.image_based_ratio >= 0.4

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.image_based_pages:
            pages = ", ".join(str(p) for p in self.image_based_pages[:12])
            more = "..." if len(self.image_based_pages) > 12 else ""
            out.append(
                f"IMAGE-BASED CONTENT - EXTRACTION CONFIDENCE REDUCED (pages {pages}{more})"
            )
        if self.pages_total and self.total_characters / max(self.pages_total, 1) < 120:
            out.append("Low machine-readable text density across the document.")
        return out
