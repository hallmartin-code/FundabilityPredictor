"""TEN Capital lead qualification: readiness flags, fit score, next step.

Flags are tri-state on purpose. "UNKNOWN" is a real answer and must never
collapse into "NO": a deck that is silent about round commitments has not
told us there are none, and treating silence as a negative would quietly
disqualify companies for a documentation gap.
"""

from __future__ import annotations

import logging

from analysis import signals as sig
from models.company import SectorProfile, Stage
from models.prediction import ConfidenceResult, TenCapitalAssessment, TenCapitalFlag
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)

YES, NO, UNKNOWN, REVIEW = "YES", "NO", "UNKNOWN", "REVIEW"

#: Qualification thresholds.
MIN_RAISE = 2_000_000.0
MIN_REVENUE = 700_000.0
MIN_RUNWAY_MONTHS = 9.0
COMMITTED_LOW = 25.0
COMMITTED_HIGH = 50.0

#: Contribution of each flag to the fit score (points out of 100).
FLAG_POINTS: dict[str, float] = {
    "raise_over_2m": 18.0,
    "twenty_five_percent_committed": 12.0,
    "fifty_percent_committed": 10.0,
    "revenue_over_700k": 14.0,
    "revenue_generating": 8.0,
    "institutional_round": 10.0,
    "lead_investor": 10.0,
    "runway_over_9_months": 8.0,
    "materials_complete": 10.0,
}

#: Credit awarded when a flag is UNKNOWN. A gap should cost a company
#: something, but far less than a confirmed negative.
UNKNOWN_CREDIT = 0.35


def _tri(value: bool | None) -> str:
    if value is None:
        return UNKNOWN
    return YES if value else NO


def assess_ten_capital_fit(
    profile: DeckProfile,
    sector: SectorProfile,
    stage: Stage,
    readiness_total: float,
    confidence: ConfidenceResult,
) -> TenCapitalAssessment:
    """Produce the flags, fit score and recommended engagement action."""
    flags = _build_flags(profile, stage)
    fit_score = _fit_score(flags)
    step, explanation = _next_step(
        flags, fit_score, readiness_total, confidence, profile, sector, stage
    )
    LOGGER.info("TEN Capital fit score %.1f -> %s", fit_score, step)
    return TenCapitalAssessment(
        fit_score=fit_score,
        flags=flags,
        recommended_next_step=step,
        explanation=explanation,
    )


def _build_flags(profile: DeckProfile, stage: Stage) -> list[TenCapitalFlag]:
    target = profile.number("target_raise")
    pct = sig.committed_pct(profile)
    revenue = sig.revenue(profile)
    runway = sig.runway_months(profile)
    lead = sig.has_lead_investor(profile)

    raise_flag = _tri(target >= MIN_RAISE if target is not None else None)
    committed_25 = _tri(pct >= COMMITTED_LOW if pct is not None else None)
    committed_50 = _tri(pct >= COMMITTED_HIGH if pct is not None else None)

    # Revenue is the one place where absence is genuinely informative: a
    # company with revenue puts it in the deck, so silence reads as
    # pre-revenue. Both revenue flags are derived from the same evidence so
    # they can never contradict one another.
    revenue_generating = _tri(revenue is not None and revenue > 0)
    revenue_over = _tri(revenue is not None and revenue >= MIN_REVENUE)

    institutional = _institutional_flag(profile, stage)
    materials = _materials_flag(profile)

    flags = [
        TenCapitalFlag(key="raise_over_2m", label="Raise >= $2M", value=raise_flag),
        TenCapitalFlag(key="twenty_five_percent_committed", label=">= 25% committed", value=committed_25),
        TenCapitalFlag(key="fifty_percent_committed", label=">= 50% committed", value=committed_50),
        TenCapitalFlag(key="revenue_over_700k", label="Revenue >= $700K", value=revenue_over),
        TenCapitalFlag(key="revenue_generating", label="Revenue generating", value=revenue_generating),
        TenCapitalFlag(key="institutional_round", label="Institutional round", value=institutional),
        TenCapitalFlag(key="lead_investor", label="Lead investor identified", value=_tri(lead)),
        TenCapitalFlag(
            key="runway_over_9_months",
            label="Runway >= 9 months",
            value=_tri(runway >= MIN_RUNWAY_MONTHS if runway is not None else None),
        ),
        TenCapitalFlag(key="materials_complete", label="Investor materials complete", value=materials),
    ]
    flags.append(
        TenCapitalFlag(
            key="likely_fundraising_candidate",
            label="Likely fundraising candidate",
            value=_candidate_flag(flags, profile),
        )
    )
    return flags


def _institutional_flag(profile: DeckProfile, stage: Stage) -> str:
    """Institutional if the instrument, investors or stage imply a priced round."""
    if sig.has_strategic_investor(profile) or profile.has("existing_investors"):
        return YES
    instrument = profile.text("security_type").lower()
    if instrument:
        if any(word in instrument for word in ("preferred", "priced round", "series")):
            return YES
        if any(word in instrument for word in ("safe", "convertible", "note")):
            return NO
    if stage in (Stage.SERIES_A, Stage.SERIES_B, Stage.SERIES_C_PLUS, Stage.GROWTH):
        return YES
    return UNKNOWN


def _materials_flag(profile: DeckProfile) -> str:
    """Are the investor materials complete enough to run a process on?"""
    quality = profile.extraction_quality
    if not quality.pages_total:
        return UNKNOWN
    essentials = ["target_raise", "use_of_funds", "founders", "tam", "product_stage"]
    present = sum(1 for field in essentials if profile.has(field))
    if quality.is_image_heavy:
        return NO
    if present >= 4 and sig.has_financial_model(profile):
        return YES
    if present <= 2:
        return NO
    return REVIEW


def _candidate_flag(flags: list[TenCapitalFlag], profile: DeckProfile) -> str:
    """Overall YES / NO / REVIEW verdict on fundraising candidacy."""
    lookup = {flag.key: flag.value for flag in flags}
    positives = sum(1 for key in ("raise_over_2m", "revenue_generating", "institutional_round")
                    if lookup.get(key) == YES)
    hard_negatives = sum(1 for key in ("raise_over_2m",) if lookup.get(key) == NO)

    if hard_negatives:
        return NO
    if positives >= 2:
        return YES
    if lookup.get("raise_over_2m") == YES:
        return REVIEW
    return REVIEW


def _fit_score(flags: list[TenCapitalFlag]) -> float:
    """Weighted flag score, with partial credit for unknowns."""
    total = sum(FLAG_POINTS.values())
    earned = 0.0
    for flag in flags:
        points = FLAG_POINTS.get(flag.key)
        if points is None:
            continue
        if flag.value == YES:
            earned += points
        elif flag.value == REVIEW:
            earned += points * 0.5
        elif flag.value == UNKNOWN:
            earned += points * UNKNOWN_CREDIT
    return round(earned / total * 100.0, 1)


def _next_step(
    flags: list[TenCapitalFlag],
    fit_score: float,
    readiness: float,
    confidence: ConfidenceResult,
    profile: DeckProfile,
    sector: SectorProfile,
    stage: Stage,
) -> tuple[str, str]:
    """Choose the engagement action and say why in one sentence."""
    lookup = {flag.key: flag.value for flag in flags}
    raise_ok = lookup.get("raise_over_2m")
    committed_25 = lookup.get("twenty_five_percent_committed") == YES
    has_lead = lookup.get("lead_investor") == YES

    if raise_ok == NO:
        target = profile.number("target_raise")
        return (
            "NOT CURRENTLY QUALIFIED",
            f"The {target:,.0f} raise sits below the $2M threshold TEN Capital engages at."
            if target
            else "The stated raise sits below the $2M threshold TEN Capital engages at.",
        )

    if fit_score >= 70 and readiness >= 70 and (committed_25 or has_lead):
        return (
            "HIGH PRIORITY - Immediate fundraising discussion",
            f"Strong fit at {fit_score:.0f}/100 with a {readiness:.0f}/100 readiness score and "
            "visible round momentum, so the round is live now.",
        )

    if fit_score >= 55 and readiness >= 55:
        return (
            "QUALIFIED - Schedule fundability review",
            f"Fit of {fit_score:.0f}/100 and readiness of {readiness:.0f}/100 support engagement "
            "once the outstanding gaps are reviewed.",
        )

    if fit_score >= 40 or readiness >= 45:
        gap = _primary_gap(lookup, profile, sector, stage)
        return (
            "NURTURE - Improve milestones before active outreach",
            f"The profile is credible but {gap} should be resolved before an active process.",
        )

    if confidence.level == "LOW" and confidence.missing_critical_fields:
        return (
            "EARLY - Reassess after additional traction",
            f"Too much of the financing picture is undisclosed ({len(confidence.missing_critical_fields)} "
            "critical fields missing) to qualify the company on this deck alone.",
        )

    return (
        "EARLY - Reassess after additional traction",
        f"Readiness of {readiness:.0f}/100 is below the level at which an active financing "
        "process is likely to succeed.",
    )


def _primary_gap(
    lookup: dict[str, str], profile: DeckProfile, sector: SectorProfile, stage: Stage
) -> str:
    if lookup.get("lead_investor") != YES:
        return "the absence of a committed lead investor"
    if lookup.get("twenty_five_percent_committed") != YES:
        return "round commitments below 25%"
    if lookup.get("revenue_generating") == NO:
        return "the absence of revenue or documented paid commitments"
    if lookup.get("materials_complete") != YES:
        return "incomplete investor materials"
    if lookup.get("runway_over_9_months") == NO:
        return "runway below nine months"
    return "the remaining readiness gaps"
