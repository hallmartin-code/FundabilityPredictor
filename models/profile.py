"""The extracted deck profile: a provenance-carrying store of every field."""

from __future__ import annotations

from typing import Any, Iterator

from pydantic import BaseModel, Field

from models.evidence import ExtractionQuality, FieldValue, Inconsistency, ValueStatus
from models.fields import FIELD_SPECS, FieldCategory, spec_for


class SourceDocument(BaseModel):
    """Metadata about the ingested file."""

    path: str
    filename: str
    file_type: str
    page_count: int = 0
    sha256: str | None = None


class DeckProfile(BaseModel):
    """All extracted fields plus the evidence needed to defend them.

    Field access is deliberately funnelled through :meth:`get` / :meth:`value`
    so that a missing field can never masquerade as a real one: an unknown
    field name raises, and an unpopulated field returns an explicit MISSING
    ``FieldValue`` rather than ``None``.
    """

    source: SourceDocument
    values: dict[str, FieldValue] = Field(default_factory=dict)
    extraction_quality: ExtractionQuality = Field(default_factory=ExtractionQuality)
    inconsistencies: list[Inconsistency] = Field(default_factory=list)
    raw_text_by_page: dict[int, str] = Field(default_factory=dict, exclude=True)
    #: Summary of the Claude extraction pass, when one ran. ``None`` means the
    #: analysis was purely deterministic.
    llm_run: dict[str, Any] | None = None

    # ------------------------------------------------------------------ access
    def get(self, name: str) -> FieldValue:
        spec_for(name)  # validates the name against the registry
        return self.values.get(name) or FieldValue.missing_field(name)

    def value(self, name: str, default: Any = None) -> Any:
        fv = self.get(name)
        return fv.value if fv.is_known else default

    def has(self, name: str) -> bool:
        return self.get(name).is_known

    def number(self, name: str, default: float | None = None) -> float | None:
        """Numeric access that refuses to coerce non-numeric junk."""
        v = self.value(name)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return default
        return float(v)

    def text(self, name: str, default: str = "") -> str:
        v = self.value(name)
        return v if isinstance(v, str) and v.strip() else default

    def set(self, fv: FieldValue) -> None:
        spec_for(fv.field)
        self.values[fv.field] = fv

    def put(
        self,
        name: str,
        value: Any,
        status: ValueStatus,
        confidence: float,
        source_page: int | None = None,
        source_text: str | None = None,
        note: str | None = None,
    ) -> None:
        self.set(
            FieldValue(
                field=name,
                value=value,
                status=status,
                confidence=confidence,
                source_page=source_page,
                source_text=source_text,
                note=note,
            )
        )

    # ------------------------------------------------------------- inspection
    def known_fields(self) -> list[str]:
        return [n for n in self.values if self.get(n).is_known]

    def missing_fields(self) -> list[str]:
        known = set(self.known_fields())
        return [s.name for s in FIELD_SPECS if s.name not in known]

    def estimated_fields(self) -> list[str]:
        return [n for n, fv in self.values.items() if fv.status is ValueStatus.ESTIMATED]

    def overridden_fields(self) -> list[str]:
        return sorted(n for n, fv in self.values.items() if fv.overridden)

    def by_category(self, category: FieldCategory) -> Iterator[FieldValue]:
        for spec in FIELD_SPECS:
            if spec.category is category:
                yield self.get(spec.name)

    def completeness(self) -> float:
        """Share of all registry fields that carry a known value (0-1)."""
        return len(self.known_fields()) / len(FIELD_SPECS)

    def apply_override(self, name: str, value: Any, note: str | None = None) -> FieldValue:
        """Record a human correction, preserving the extracted original."""
        updated = self.get(name).with_override(value, note=note)
        self.set(updated)
        return updated

    def add_inconsistency(self, inc: Inconsistency) -> None:
        if not any(existing.code == inc.code for existing in self.inconsistencies):
            self.inconsistencies.append(inc)
