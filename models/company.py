"""Company-level views over a :class:`DeckProfile`: stage and sector."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Stage(str, Enum):
    PRE_SEED = "Pre-Seed"
    SEED = "Seed"
    SERIES_A = "Series A"
    SERIES_B = "Series B"
    SERIES_C_PLUS = "Series C+"
    GROWTH = "Growth"
    UNKNOWN = "Unknown"

    @property
    def is_known(self) -> bool:
        return self is not Stage.UNKNOWN


#: Ordering used for "is this later than that" comparisons. UNKNOWN sorts last
#: on purpose: it must never satisfy a ">= Series A" style test.
STAGE_ORDER: dict[Stage, int] = {
    Stage.PRE_SEED: 0,
    Stage.SEED: 1,
    Stage.SERIES_A: 2,
    Stage.SERIES_B: 3,
    Stage.SERIES_C_PLUS: 4,
    Stage.GROWTH: 5,
    Stage.UNKNOWN: -1,
}


def stage_from_text(text: str | None) -> Stage:
    """Map free text such as "seed round" or "Series-A" onto a Stage."""
    if not text:
        return Stage.UNKNOWN
    t = text.strip().lower().replace("-", " ").replace("_", " ")
    t = " ".join(t.split())
    if "pre seed" in t or "preseed" in t or t == "ps":
        return Stage.PRE_SEED
    if "series c" in t or "series d" in t or "series e" in t or "series f" in t:
        return Stage.SERIES_C_PLUS
    if "series b" in t:
        return Stage.SERIES_B
    if "series a" in t:
        return Stage.SERIES_A
    if "growth" in t or "late stage" in t or "pre ipo" in t:
        return Stage.GROWTH
    if "seed" in t:
        return Stage.SEED
    if "bridge" in t or "friends and family" in t or "angel" in t:
        return Stage.PRE_SEED
    return Stage.UNKNOWN


def at_least(stage: Stage, floor: Stage) -> bool:
    """True when ``stage`` is known and at or beyond ``floor``."""
    if stage is Stage.UNKNOWN:
        return False
    return STAGE_ORDER[stage] >= STAGE_ORDER[floor]


class StageAssessment(BaseModel):
    """Declared vs. model-inferred stage, with the reasoning kept."""

    declared: Stage = Stage.UNKNOWN
    inferred: Stage = Stage.UNKNOWN
    confidence: str = "LOW"  # HIGH / MEDIUM / LOW
    rationale: list[str] = []
    signals_used: int = 0

    @property
    def effective(self) -> Stage:
        """The stage the model should reason with."""
        if self.inferred.is_known:
            return self.inferred
        return self.declared


class SectorProfile(BaseModel):
    """Resolved sector rules for the company."""

    key: str = "unknown"
    label: str = "Not Provided"
    traction_mode: str = "hybrid"  # revenue | milestone | hybrid
    capital_intensity: str = "medium"  # low | medium | high
    regulated: bool = False
    revenue_expected_from: str = "Series A"
    probability_modifier: dict[str, float] = {"p3": 1.0, "p6": 1.0, "p12": 1.0}
    note: str = ""
    matched_keywords: list[str] = []
    confidence: str = "LOW"

    @property
    def is_capital_intensive(self) -> bool:
        return self.capital_intensity == "high"

    @property
    def is_milestone_driven(self) -> bool:
        return self.traction_mode == "milestone"
