"""Shared predicates over a :class:`DeckProfile`.

Both the readiness scorer and the probability adjusters ask the same
questions ("is there a lead investor?", "is the round oversized for the
stage?"), so the answers live here once. Every helper returns ``None`` when
the deck is silent, which callers must treat as "unknown", never as "no".
"""

from __future__ import annotations

import re

from config.settings import AppConfig
from extractors.normalization import months_until
from models.company import SectorProfile, Stage
from models.financing import (
    commercial_engagement_count,
    effective_annual_revenue,
    round_snapshot,
    total_customers,
)
from models.profile import DeckProfile

# --------------------------------------------------------------------------
# Document-level text probes
# --------------------------------------------------------------------------
_FINANCIAL_MODEL_RE = re.compile(
    r"\bfinancial(?:s|\s+model|\s+projections?)\b|\bprojections?\b|\bforecast\b|"
    r"\bpro\s*forma\b|\bP&L\b|\bincome\s+statement\b|\bcash\s+flow\b|\brevenue\s+model\b",
    re.IGNORECASE,
)
_MILESTONE_RE = re.compile(
    r"\bmilestones?\b|\broadmap\b|\btimeline\b|\bnext\s+12\s+months\b|\bQ[1-4]\s*20\d\d\b",
    re.IGNORECASE,
)
_NON_DILUTIVE_RE = re.compile(
    r"\bnon-?dilutive\b|\bgrant(?:s|\s+funding|\s+award)?\b|\bSBIR\b|\bSTTR\b|"
    r"\bNIH\b|\bNSF\b|\baward(?:ed)?\b|\btax\s+credit\b",
    re.IGNORECASE,
)
_STRATEGIC_INVESTOR_RE = re.compile(
    r"\bstrategic\s+(?:investor|partner|investment)\b|\bcorporate\s+(?:vc|venture|investor)\b|"
    r"\bCVC\b",
    re.IGNORECASE,
)
_REPEAT_FOUNDER_RE = re.compile(
    r"\bserial\s+(?:entrepreneur|founder)\b|\brepeat\s+founder\b|\bsecond[\s-]time\s+founder\b|"
    r"\bpreviously\s+founded\b|\bfounded\s+(?:two|three|\d+)\b",
    re.IGNORECASE,
)
_PRIOR_EXIT_RE = re.compile(
    r"\b(?:successful\s+)?exit(?:ed|s)?\b|\bacquired\s+by\b|\bIPO\b|\bsold\s+to\b|\bM&A\s+exit\b",
    re.IGNORECASE,
)
_ADVISOR_CONTEXT_RE = re.compile(r"\badvisor(?:y)?\b|\badviser\b|\bboard\s+member\b|\bconsultant\b", re.IGNORECASE)
_FOUNDER_CONTEXT_RE = re.compile(
    r"\bco-?founder\b|\bfounder\b|\bfounding\b|\bCEO\b|\bCTO\b|\bCOO\b|\bchief\s+executive\b",
    re.IGNORECASE,
)
_REGULATORY_ACHIEVED_RE = re.compile(
    r"\bfda\s+(?:cleared|approved)\b|\bce\s+marked\b|\b510\(?k\)?\s+cleared\b|"
    r"\bbreakthrough\s+device\s+designation\s+(?:granted|received)\b|\borphan\s+drug\s+designation\b|"
    r"\bIND\s+cleared\b|\breceived\s+(?:clearance|approval|designation)\b",
    re.IGNORECASE,
)
_LONG_REGULATORY_RE = re.compile(
    r"\bPMA\b|\bphase\s+(?:ii|iii|2|3)\b|\bpivotal\s+trial\b|\bclinical\s+trial\b|"
    r"\bIDE\b|\bNDA\s+submission\b|\bBLA\b|\bpre-?market\s+approval\b",
    re.IGNORECASE,
)
_LEGAL_DEPENDENCY_RE = re.compile(
    r"\blitigation\b|\blawsuit\b|\bdispute\b|\blicen[cs]e\s+from\b|\bexclusive\s+licen[cs]e\s+with\b|"
    r"\bpending\s+approval\s+from\b|\bregulatory\s+approval\s+required\b",
    re.IGNORECASE,
)
_DIFFERENTIATION_RE = re.compile(
    r"\bdifferentiat|\bunique\b|\bproprietary\b|\bmoat\b|\bbarrier\s+to\s+entry\b|"
    r"\bcompetitive\s+advantage\b|\bonly\s+(?:solution|company|platform)\b|\bfirst\s+to\b",
    re.IGNORECASE,
)
_PARTNERSHIP_RE = re.compile(
    r"\b(?:signed|executed|entered)\s+(?:a\s+)?(?:partnership|agreement|MOU|LOI|contract)\b|"
    r"\bdistribution\s+agreement\b|\bjoint\s+development\b|\bstrategic\s+partnership\b|"
    r"\bpartnership\s+with\b|\bpartnered\s+with\b",
    re.IGNORECASE,
)


#: Words that turn a claim into a plan. "Once fully developed and FDA
#: approved" is an aspiration, not an approval, and a corpus-wide regex will
#: happily read it as one unless the surrounding lines are inspected.
_CONDITIONAL_RE = re.compile(
    r"\bonce\b|\bwhen\b|\bupon\b|\bafter\b|\bwill\b|\bwould\b|\bcould\b|\bplans?\b|"
    r"\bintends?\b|\bexpects?\b|\binterested\b|\btargets?\b|\bgoal\b|\bmilestone\b|"
    r"\bpath(?:way)?\s+to\b|\bto\s+reach\b|\bseeking\b|\bpending\b|\bupcoming\b|"
    r"\bplanned\b|\bprojected\b|\bfuture\b|\bif\b",
    re.IGNORECASE,
)


def _corpus(profile: DeckProfile) -> str:
    return " ".join(profile.raw_text_by_page.values())


def _lines(profile: DeckProfile) -> list[str]:
    out: list[str] = []
    for text in profile.raw_text_by_page.values():
        out.extend(ln.strip() for ln in text.splitlines() if ln.strip())
    return out


def find_in_context(
    profile: DeckProfile,
    pattern: re.Pattern[str],
    forbid: re.Pattern[str] | None = None,
    require: re.Pattern[str] | None = None,
    window: int = 1,
) -> bool:
    """Match ``pattern`` on a line whose neighbourhood passes the guards.

    Deck slides put a claim on one line and its qualifier on another, so a
    corpus-wide search cannot tell "FDA approved" from "once ... FDA
    approved". Looking at a window of surrounding lines can.
    """
    lines = _lines(profile)
    for index, line in enumerate(lines):
        if not pattern.search(line):
            continue
        context = " ".join(lines[max(0, index - window): index + window + 1])
        if forbid is not None and forbid.search(context):
            continue
        if require is not None and not require.search(context):
            continue
        return True
    return False


def has_financial_model(profile: DeckProfile) -> bool:
    return bool(_FINANCIAL_MODEL_RE.search(_corpus(profile)))


def has_milestones(profile: DeckProfile) -> bool:
    return bool(_MILESTONE_RE.search(_corpus(profile)))


def has_non_dilutive_funding(profile: DeckProfile) -> bool:
    return bool(_NON_DILUTIVE_RE.search(_corpus(profile)))


def has_strategic_investor(profile: DeckProfile) -> bool:
    haystack = " ".join(
        filter(None, [profile.text("existing_investors"), profile.text("lead_investor"), _corpus(profile)])
    )
    return bool(_STRATEGIC_INVESTOR_RE.search(haystack))


def is_repeat_founder(profile: DeckProfile) -> bool:
    return bool(_REPEAT_FOUNDER_RE.search(_corpus(profile)))


def has_founder_prior_exit(profile: DeckProfile) -> bool:
    """A prior exit only counts when it belongs to the founding team.

    Team slides list founders and advisors together, and an advisor's exit
    says nothing about the people who will execute this round. The advisor
    label often sits a line or two away from the claim, so the check needs a
    window rather than a single line.
    """
    return find_in_context(
        profile,
        _PRIOR_EXIT_RE,
        forbid=_ADVISOR_CONTEXT_RE,
        require=_FOUNDER_CONTEXT_RE,
        window=2,
    )


def has_regulatory_milestone(profile: DeckProfile) -> bool:
    """True only for approvals already held, never for approvals sought."""
    if profile.has("regulatory_approvals"):
        return True
    return find_in_context(profile, _REGULATORY_ACHIEVED_RE, forbid=_CONDITIONAL_RE, window=1)


def has_long_regulatory_path(profile: DeckProfile, sector: SectorProfile) -> bool:
    if not sector.regulated:
        return False
    if has_regulatory_milestone(profile):
        return False
    return bool(_LONG_REGULATORY_RE.search(_corpus(profile))) or sector.key in {
        "biotech", "medtech", "life_sciences"
    }


def has_legal_dependency(profile: DeckProfile) -> bool:
    return bool(_LEGAL_DEPENDENCY_RE.search(_corpus(profile)))


def has_differentiation(profile: DeckProfile) -> bool:
    if profile.has("competitive_environment") or profile.has("intellectual_property"):
        return True
    return bool(_DIFFERENTIATION_RE.search(_corpus(profile)))


def has_partnership(profile: DeckProfile) -> bool:
    if profile.has("partnerships"):
        source = profile.get("partnerships").source_text or ""
        return bool(_PARTNERSHIP_RE.search(source)) or bool(_PARTNERSHIP_RE.search(_corpus(profile)))
    return bool(_PARTNERSHIP_RE.search(_corpus(profile)))


def has_ip(profile: DeckProfile) -> bool:
    return profile.has("patents") or profile.has("intellectual_property")


# --------------------------------------------------------------------------
# Numeric probes
# --------------------------------------------------------------------------
def committed_pct(profile: DeckProfile) -> float | None:
    return round_snapshot(profile).committed_ratio


def has_lead_investor(profile: DeckProfile) -> bool | None:
    return round_snapshot(profile).has_lead_investor


def revenue(profile: DeckProfile) -> float | None:
    return effective_annual_revenue(profile)


def customers(profile: DeckProfile) -> float | None:
    return total_customers(profile)


def commercial_engagements(profile: DeckProfile) -> float | None:
    return commercial_engagement_count(profile)


def runway_months(profile: DeckProfile) -> float | None:
    return profile.number("runway_months")


def founder_is_full_time(profile: DeckProfile) -> bool | None:
    value = profile.value("founder_full_time")
    if isinstance(value, bool):
        return value
    text = profile.text("founder_full_time")
    if not text:
        return None
    low = text.lower()
    if "full" in low and "part" not in low:
        return True
    if "part-time" in low or "part time" in low or "fractional" in low:
        return False
    return None


def months_to_maturity(profile: DeckProfile) -> float | None:
    value = profile.value("maturity_date")
    return months_until(str(value)) if value else None


# --------------------------------------------------------------------------
# Benchmark comparisons
# --------------------------------------------------------------------------
def benchmark_range(
    cfg: AppConfig, stage: Stage, key: str, sector: SectorProfile
) -> tuple[float, float] | None:
    """The stage's typical range for ``key``, scaled for capital intensity."""
    if not stage.is_known:
        return None
    bench = cfg.stage_benchmarks.get(stage.value)
    if not bench or key not in bench:
        return None
    low, high = bench[key]
    multiplier = cfg.capital_intensity_multiplier.get(sector.capital_intensity, 1.0)
    if key in ("raise_typical", "pre_money_typical"):
        return low * multiplier, high * multiplier
    return float(low), float(high)


def benchmark_multiple(
    value: float | None, cfg: AppConfig, stage: Stage, key: str, sector: SectorProfile
) -> float | None:
    """How far above the top of the typical range a value sits (1.0 = at the top)."""
    if value is None:
        return None
    rng = benchmark_range(cfg, stage, key, sector)
    if not rng:
        return None
    _, high = rng
    return value / high if high else None


def within_benchmark(
    value: float | None, cfg: AppConfig, stage: Stage, key: str, sector: SectorProfile
) -> bool | None:
    if value is None:
        return None
    rng = benchmark_range(cfg, stage, key, sector)
    if not rng:
        return None
    low, high = rng
    return low <= value <= high


# --------------------------------------------------------------------------
# Scoring maths
# --------------------------------------------------------------------------
def piecewise(value: float, points: list[tuple[float, float]]) -> float:
    """Linear interpolation across ``(threshold, score)`` pairs.

    Scores are clamped to the first and last points, so the curve is defined
    for every input.
    """
    if not points:
        return 0.0
    if value <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if value <= x1:
            if x1 == x0:
                return y1
            ratio = (value - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return points[-1][1]


def join_readable(items: list[str]) -> str:
    """Readable list join: "a", "a and b", "a, b and c"."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"
