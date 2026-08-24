"""Analysis output models: scores, probabilities, blockers, actions, result."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from models.company import SectorProfile, StageAssessment
from models.evidence import Inconsistency
from models.profile import DeckProfile

HORIZONS: tuple[str, str, str] = ("p3", "p6", "p12")
HORIZON_LABELS: dict[str, str] = {"p3": "3 Months", "p6": "6 Months", "p12": "12 Months"}


class SignalScore(BaseModel):
    """One evaluated signal inside a readiness category."""

    key: str
    label: str
    weight: float
    awarded: float
    evidenced: bool
    detail: str = ""


class CategoryScore(BaseModel):
    key: str
    label: str
    weight: float
    points: float
    signals: list[SignalScore] = Field(default_factory=list)

    @property
    def pct(self) -> float:
        return (self.points / self.weight * 100.0) if self.weight else 0.0

    @property
    def evidenced_signals(self) -> int:
        return sum(1 for s in self.signals if s.evidenced)


class ReadinessScore(BaseModel):
    total: float
    label: str
    categories: list[CategoryScore] = Field(default_factory=list)

    def category(self, key: str) -> CategoryScore | None:
        return next((c for c in self.categories if c.key == key), None)


class BridgeStep(BaseModel):
    """One line of the probability bridge, e.g. "+ Lead investor: +8%"."""

    label: str
    kind: str  # baseline | signal | sector | market | clamp | monotonicity
    delta: dict[str, float] = Field(default_factory=dict)
    running: dict[str, float] = Field(default_factory=dict)

    def render(self, horizon: str = "p6") -> str:
        if self.kind == "baseline":
            return f"Baseline probability: {self.running.get(horizon, 0):.0f}%"
        d = self.delta.get(horizon, 0.0)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.0f}% {self.label}"


class Probability(BaseModel):
    p3: float
    p6: float
    p12: float

    def get(self, horizon: str) -> float:
        return {"p3": self.p3, "p6": self.p6, "p12": self.p12}[horizon]

    def as_dict(self) -> dict[str, float]:
        return {"p3": self.p3, "p6": self.p6, "p12": self.p12}


class MarketCondition(BaseModel):
    key: str = "neutral"
    label: str = "Neutral"
    multiplier: dict[str, float] = Field(default_factory=lambda: {"p3": 1.0, "p6": 1.0, "p12": 1.0})
    #: True only when the environment came from verified live data. The
    #: default model has no live feed, so this stays False and the report
    #: states the value as an assumption.
    is_verified: bool = False

    @property
    def disclosure(self) -> str:
        if self.is_verified:
            return f"MARKET CONDITION: {self.label} (verified)"
        return f"MARKET CONDITION ASSUMPTION: {self.label}"


class PredictionResult(BaseModel):
    baseline: Probability
    final: Probability
    bridge: list[BridgeStep] = Field(default_factory=list)
    positive_signals: list[str] = Field(default_factory=list)
    negative_signals: list[str] = Field(default_factory=list)
    market_condition: MarketCondition = Field(default_factory=MarketCondition)
    sector_modifier: dict[str, float] = Field(default_factory=dict)

    def bridge_text(self, horizon: str = "p6") -> str:
        lines = [step.render(horizon) for step in self.bridge]
        lines.append(
            f"\nFinal {HORIZON_LABELS[horizon].lower()} probability: "
            f"{self.final.get(horizon):.0f}%"
        )
        return "\n".join(lines)


class ConfidenceResult(BaseModel):
    score: float
    level: str
    components: dict[str, float] = Field(default_factory=dict)
    missing_critical_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Blocker(BaseModel):
    key: str
    severity: str  # CRITICAL | HIGH | MODERATE
    title: str
    detail: str

    @property
    def rank(self) -> int:
        return {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2}.get(self.severity, 3)


class Action(BaseModel):
    action: str
    why: str
    impact: str  # HIGH | MEDIUM | LOW
    timeframe: str  # Immediate | 30 Days | 60 Days | 90 Days

    @property
    def rank(self) -> int:
        impact = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(self.impact, 3)
        time = {"Immediate": 0, "30 Days": 1, "60 Days": 2, "90 Days": 3}.get(self.timeframe, 4)
        return impact * 10 + time


class ComparableProfile(BaseModel):
    key: str
    label: str
    rationale: str


class TenCapitalFlag(BaseModel):
    key: str
    label: str
    value: str  # YES | NO | UNKNOWN | REVIEW


class TenCapitalAssessment(BaseModel):
    fit_score: float
    flags: list[TenCapitalFlag] = Field(default_factory=list)
    recommended_next_step: str
    explanation: str

    def flag(self, key: str) -> str:
        return next((f.value for f in self.flags if f.key == key), "UNKNOWN")


class AnalysisResult(BaseModel):
    """Everything the report writers need, and nothing they have to re-derive."""

    profile: DeckProfile
    stage: StageAssessment
    sector: SectorProfile
    readiness: ReadinessScore
    prediction: PredictionResult
    confidence: ConfidenceResult
    blockers: list[Blocker] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    comparable: ComparableProfile
    ten_capital: TenCapitalAssessment
    inconsistencies: list[Inconsistency] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def company_name(self) -> str:
        return self.profile.text("company_name", "Not Provided")
