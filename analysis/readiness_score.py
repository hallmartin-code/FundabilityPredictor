"""The Financing Readiness Score (0-100).

Nine categories, each weighted by ``config/scoring_weights.yaml``. Inside a
category, individual signals carry relative weights and are scored 0.0-1.0.

Two rules keep the score honest:

* A signal with no data is not scored zero. Zero would mean "we know this is
  bad"; instead it receives ``missing_data_credit``, which costs the company
  points against a fully evidenced deck without pretending to a finding.
* Traction is judged by the norms of the company's sector. A pre-revenue
  biotech is not penalised for lacking SaaS-style revenue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from analysis import signals as sig
from config.settings import AppConfig, load_config
from models.company import STAGE_ORDER, SectorProfile, Stage, at_least, stage_from_text
from models.prediction import CategoryScore, ReadinessScore, SignalScore
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)


@dataclass
class Evaluated:
    """One scored signal. ``score=None`` means the deck gave us nothing."""

    key: str
    label: str
    weight: float
    score: float | None
    detail: str = ""


#: Signature of a category scorer.
Scorer = Callable[[DeckProfile, SectorProfile, Stage, AppConfig], list[Evaluated]]


def score_readiness(
    profile: DeckProfile,
    sector: SectorProfile,
    stage: Stage,
    config: AppConfig | None = None,
) -> ReadinessScore:
    """Compute the readiness score and its full breakdown."""
    cfg = config or load_config()
    categories: list[CategoryScore] = []

    for key, scorer in SCORERS.items():
        meta = cfg.categories[key]
        weight = float(meta["weight"])
        evaluated = scorer(profile, sector, stage, cfg)
        categories.append(_assemble(key, str(meta["label"]), weight, evaluated, cfg))

    total = round(sum(c.points for c in categories), 1)
    label = cfg.readiness_label(total)
    LOGGER.info("Financing readiness score: %.1f/100 (%s)", total, label)
    return ReadinessScore(total=total, label=label, categories=categories)


def _assemble(
    key: str, label: str, weight: float, evaluated: list[Evaluated], cfg: AppConfig
) -> CategoryScore:
    credit = cfg.missing_data_credit
    total_weight = sum(e.weight for e in evaluated) or 1.0
    signal_scores: list[SignalScore] = []
    accumulated = 0.0

    for item in evaluated:
        fraction = credit if item.score is None else max(0.0, min(1.0, item.score))
        share = item.weight / total_weight
        accumulated += share * fraction
        signal_scores.append(
            SignalScore(
                key=item.key,
                label=item.label,
                weight=round(weight * share, 2),
                awarded=round(weight * share * fraction, 2),
                evidenced=item.score is not None,
                detail=item.detail or ("Not stated in the deck." if item.score is None else ""),
            )
        )

    return CategoryScore(
        key=key,
        label=label,
        weight=weight,
        points=round(weight * accumulated, 2),
        signals=signal_scores,
    )


# ---------------------------------------------------------------------------
# A. Traction & Revenue - sector aware
# ---------------------------------------------------------------------------
def _score_traction(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Evaluated]:
    if sector.traction_mode == "milestone":
        return _traction_milestone(profile, sector, stage, cfg)
    if sector.traction_mode == "hybrid":
        return _traction_hybrid(profile, sector, stage, cfg)
    return _traction_revenue(profile, sector, stage, cfg)


def _revenue_signal(profile: DeckProfile, sector: SectorProfile, stage: Stage, weight: float) -> Evaluated:
    value = sig.revenue(profile)
    if value is None:
        return Evaluated("revenue_scale", "Revenue scale", weight, None)
    # Revenue is judged against what the stage would normally show.
    expectation = {
        Stage.PRE_SEED: 100_000.0,
        Stage.SEED: 750_000.0,
        Stage.SERIES_A: 3_000_000.0,
        Stage.SERIES_B: 12_000_000.0,
        Stage.SERIES_C_PLUS: 40_000_000.0,
        Stage.GROWTH: 80_000_000.0,
    }.get(stage, 1_000_000.0)
    ratio = value / expectation if expectation else 0.0
    score = sig.piecewise(ratio, [(0.0, 0.0), (0.25, 0.35), (0.5, 0.55), (1.0, 0.8), (2.0, 1.0)])
    return Evaluated(
        "revenue_scale", "Revenue scale", weight, score,
        f"${value:,.0f} annual revenue against a ~${expectation:,.0f} {stage.value} benchmark.",
    )


def _growth_signal(profile: DeckProfile, weight: float) -> Evaluated:
    growth = profile.number("revenue_growth")
    if growth is None:
        return Evaluated("revenue_growth", "Revenue growth", weight, None)
    score = sig.piecewise(growth, [(0, 0.1), (30, 0.4), (60, 0.6), (100, 0.8), (200, 1.0)])
    return Evaluated("revenue_growth", "Revenue growth", weight, score, f"{growth:.0f}% growth reported.")


def _customer_signal(profile: DeckProfile, weight: float) -> Evaluated:
    count = sig.customers(profile)
    if count is None:
        return Evaluated("customers", "Customer adoption", weight, None)
    score = sig.piecewise(count, [(0, 0.0), (1, 0.3), (5, 0.5), (20, 0.7), (100, 0.9), (500, 1.0)])
    return Evaluated("customers", "Customer adoption", weight, score, f"{count:.0f} customers reported.")


def _retention_signal(profile: DeckProfile, weight: float) -> Evaluated:
    renewal = profile.number("renewal_rate")
    churn = profile.number("churn")
    if renewal is not None:
        score = sig.piecewise(renewal, [(50, 0.1), (70, 0.4), (85, 0.7), (95, 0.9), (110, 1.0)])
        return Evaluated("retention", "Customer retention", weight, score, f"{renewal:.0f}% retention.")
    if churn is not None:
        score = sig.piecewise(churn, [(0, 1.0), (5, 0.8), (15, 0.5), (30, 0.2), (60, 0.0)])
        return Evaluated("retention", "Customer retention", weight, score, f"{churn:.0f}% churn.")
    return Evaluated("retention", "Customer retention", weight, None)


def _pipeline_signal(profile: DeckProfile, weight: float) -> Evaluated:
    pipeline = profile.number("pipeline") or profile.number("backlog")
    contracts = profile.number("contracts") or profile.number("lois")
    if pipeline is None and contracts is None:
        return Evaluated("pipeline", "Pipeline & contracts", weight, None)
    score = 0.0
    detail = []
    if pipeline is not None:
        score = max(score, sig.piecewise(pipeline, [(0, 0.0), (250_000, 0.4), (1e6, 0.7), (5e6, 1.0)]))
        detail.append(f"${pipeline:,.0f} pipeline")
    if contracts is not None:
        score = max(score, sig.piecewise(contracts, [(0, 0.0), (1, 0.4), (5, 0.7), (20, 1.0)]))
        detail.append(f"{contracts:.0f} contracts/LOIs")
    return Evaluated("pipeline", "Pipeline & contracts", weight, score, "; ".join(detail) + ".")


def _validation_signal(profile: DeckProfile, weight: float) -> Evaluated:
    has = profile.has("customer_validation")
    if not has:
        return Evaluated("validation", "Market validation", weight, None)
    return Evaluated(
        "validation", "Market validation", weight, 0.8,
        "Customer validation or endorsement evidence is present.",
    )


def _traction_revenue(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Evaluated]:
    return [
        _revenue_signal(profile, sector, stage, 3.0),
        _growth_signal(profile, 2.0),
        _customer_signal(profile, 2.0),
        _retention_signal(profile, 1.0),
        _pipeline_signal(profile, 1.0),
        _validation_signal(profile, 1.0),
    ]


def _traction_milestone(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Evaluated]:
    """Milestone sectors prove themselves with science, not revenue."""
    clinical = profile.text("clinical_stage") or profile.text("regulatory_stage")
    clinical_eval = (
        Evaluated(
            "clinical_progress", "Clinical / regulatory progress", 3.0,
            0.9 if sig.has_regulatory_milestone(profile) else 0.6,
            clinical[:110] if clinical else "Regulatory milestone evidenced.",
        )
        if clinical or sig.has_regulatory_milestone(profile)
        else Evaluated("clinical_progress", "Clinical / regulatory progress", 3.0, None)
    )
    return [
        clinical_eval,
        Evaluated(
            "technical_validation", "Technical / preclinical validation", 2.0,
            0.8 if profile.has("technology_readiness") or profile.has("customer_validation") else None,
            "Validation data referenced in the deck.",
        ),
        Evaluated(
            "partnerships", "Partnerships & collaborations", 2.0,
            0.85 if sig.has_partnership(profile) else None,
            "Partnership or collaboration evidenced.",
        ),
        Evaluated(
            "ip_position", "IP position", 1.0,
            0.85 if sig.has_ip(profile) else None,
            "Patent or IP position stated.",
        ),
        _engagement_signal(profile, 1.0),
        _validation_signal(profile, 1.0),
    ]


def _engagement_signal(profile: DeckProfile, weight: float) -> Evaluated:
    """Pre-revenue commercial evidence: pilots, LOIs, contracts, customers."""
    count = sig.commercial_engagements(profile)
    if count is None:
        return Evaluated("engagement", "Commercial engagement", weight, None)
    score = sig.piecewise(count, [(0, 0.0), (1, 0.35), (3, 0.55), (10, 0.8), (30, 1.0)])
    return Evaluated(
        "engagement", "Commercial engagement", weight, score,
        f"{count:.0f} pilots, LOIs, contracts or customers.",
    )


def _traction_hybrid(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Evaluated]:
    regulatory = (
        Evaluated(
            "regulatory_progress", "Regulatory / technical progress", 2.0,
            0.9 if sig.has_regulatory_milestone(profile) else (0.55 if profile.has("regulatory_stage") else None),
            "Regulatory milestone achieved." if sig.has_regulatory_milestone(profile)
            else "Regulatory pathway identified.",
        )
    )
    return [
        _revenue_signal(profile, sector, stage, 2.0),
        _engagement_signal(profile, 2.0),
        regulatory,
        Evaluated(
            "partnerships", "Partnerships", 1.0,
            0.85 if sig.has_partnership(profile) else None,
            "Partnership or distribution agreement evidenced.",
        ),
        _validation_signal(profile, 1.0),
        _growth_signal(profile, 1.0),
    ]


# ---------------------------------------------------------------------------
# B. Round Progress
# ---------------------------------------------------------------------------
def _score_round_progress(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Evaluated]:
    pct = sig.committed_pct(profile)
    committed = (
        Evaluated("percent_committed", "Round committed", 5.0, None)
        if pct is None
        else Evaluated(
            "percent_committed", "Round committed", 5.0,
            sig.piecewise(pct, [(0, 0.05), (10, 0.25), (25, 0.5), (50, 0.75), (75, 0.92), (100, 1.0)]),
            f"{pct:.0f}% of the round is committed.",
        )
    )

    lead = sig.has_lead_investor(profile)
    lead_eval = (
        Evaluated("lead_investor", "Lead investor", 4.0, None)
        if lead is None
        else Evaluated(
            "lead_investor", "Lead investor", 4.0, 1.0 if lead else 0.1,
            "Lead investor named." if lead else "The deck states no lead is committed.",
        )
    )

    return [
        committed,
        lead_eval,
        Evaluated(
            "existing_investors", "Existing investor participation", 2.0,
            0.85 if profile.has("existing_investors") else None,
            "Existing investors named.",
        ),
        Evaluated(
            "prior_funding", "Prior capital raised", 2.0,
            sig.piecewise(profile.number("prior_funding") or 0.0,
                          [(0, 0.2), (250_000, 0.5), (1e6, 0.75), (5e6, 1.0)])
            if profile.has("prior_funding") else None,
            f"${profile.number('prior_funding'):,.0f} raised to date."
            if profile.has("prior_funding") else "",
        ),
        Evaluated(
            "instrument_defined", "Instrument defined", 1.0,
            0.9 if profile.has("security_type") else None,
            f"Instrument: {profile.text('security_type')[:60]}." if profile.has("security_type") else "",
        ),
        Evaluated(
            "non_dilutive", "Non-dilutive funding", 1.0,
            0.85 if sig.has_non_dilutive_funding(profile) else None,
            "Grants or other non-dilutive funding referenced.",
        ),
    ]


# ---------------------------------------------------------------------------
# C. Market Opportunity
# ---------------------------------------------------------------------------
def _score_market(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Evaluated]:
    tam = profile.number("tam")
    tam_eval = (
        Evaluated("tam", "Market size (TAM)", 3.0, None)
        if tam is None
        else Evaluated(
            "tam", "Market size (TAM)", 3.0,
            # Both an implausibly small and an implausibly vast TAM read badly.
            sig.piecewise(tam, [(0, 0.1), (1e8, 0.5), (1e9, 0.85), (1e10, 1.0), (1e11, 0.7), (1e12, 0.5)]),
            f"TAM stated at ${tam:,.0f}.",
        )
    )
    bottom_up = profile.number("sam") or profile.number("som")
    return [
        tam_eval,
        Evaluated(
            "bottom_up", "Serviceable market logic", 2.0,
            0.9 if bottom_up is not None else None,
            "SAM or SOM given, indicating bottom-up sizing." if bottom_up is not None else "",
        ),
        Evaluated(
            "market_growth", "Market growth", 2.0,
            sig.piecewise(profile.number("market_growth_rate") or 0.0,
                          [(0, 0.2), (5, 0.45), (10, 0.65), (20, 0.85), (40, 1.0)])
            if profile.has("market_growth_rate") else None,
            f"Market growth of {profile.number('market_growth_rate'):.0f}%."
            if profile.has("market_growth_rate") else "",
        ),
        Evaluated(
            "sector_attractiveness", "Sector investor appetite", 2.0,
            _sector_appetite(sector),
            f"{sector.label}: {sector.note}" if sector.note else sector.label,
        ),
        Evaluated(
            "timing", "Market timing", 1.0,
            0.8 if profile.has("market_timing") else None,
            "A 'why now' argument is present.",
        ),
    ]


def _sector_appetite(sector: SectorProfile) -> float | None:
    """Investor appetite proxied by the sector's own probability modifier."""
    if sector.key == "unknown":
        return None
    modifier = sector.probability_modifier.get("p6", 1.0)
    return max(0.0, min(1.0, sig.piecewise(modifier, [(0.85, 0.35), (1.0, 0.7), (1.06, 1.0)])))


# ---------------------------------------------------------------------------
# D. Team
# ---------------------------------------------------------------------------
def _score_team(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Evaluated]:
    founders = profile.value("founders")
    founder_count = len(founders) if isinstance(founders, list) else None
    full_time = sig.founder_is_full_time(profile)
    headcount = profile.number("employees")
    headcount_score = None
    headcount_detail = ""
    if headcount is not None:
        rng = sig.benchmark_range(cfg, stage, "headcount_typical", sector)
        if rng:
            low, high = rng
            headcount_score = sig.piecewise(headcount, [(0, 0.1), (low, 0.6), (high, 1.0)])
            headcount_detail = f"{headcount:.0f} employees against a {stage.value} norm of {low:.0f}-{high:.0f}."
        else:
            headcount_score = sig.piecewise(headcount, [(0, 0.1), (3, 0.5), (15, 0.8), (50, 1.0)])
            headcount_detail = f"{headcount:.0f} employees."

    return [
        Evaluated(
            "founders", "Founding team identified", 2.0,
            sig.piecewise(float(founder_count), [(0, 0.0), (1, 0.6), (2, 0.9), (3, 1.0)])
            if founder_count else None,
            f"{founder_count} founders named." if founder_count else "",
        ),
        Evaluated(
            "domain_expertise", "Domain expertise", 2.0,
            0.85 if profile.has("domain_expertise") or profile.has("founder_experience") else None,
            "Relevant domain experience described.",
        ),
        Evaluated(
            "prior_exits", "Prior exits / repeat founder", 2.0,
            1.0 if sig.has_founder_prior_exit(profile)
            else (0.8 if sig.is_repeat_founder(profile) else None),
            "Founder prior exit evidenced." if sig.has_founder_prior_exit(profile)
            else ("Repeat founder evidenced." if sig.is_repeat_founder(profile) else ""),
        ),
        Evaluated(
            "commercial_capability", "Commercial leadership", 1.5,
            0.85 if profile.has("commercial_expertise") else None,
            "Commercial leadership identified.",
        ),
        Evaluated(
            "full_time", "Founder commitment", 1.5,
            (1.0 if full_time else 0.15) if full_time is not None else None,
            "Founders are full-time." if full_time else
            ("Founders are not full-time." if full_time is False else ""),
        ),
        Evaluated("headcount", "Team scale for stage", 1.0, headcount_score, headcount_detail),
    ]


# ---------------------------------------------------------------------------
# E. Financial Quality
# ---------------------------------------------------------------------------
def _score_financial_quality(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Evaluated]:
    margin = profile.number("gross_margin")
    runway = sig.runway_months(profile)
    burn = profile.number("burn_rate")
    revenue_value = sig.revenue(profile)
    prior = profile.number("prior_funding")

    efficiency_score = None
    efficiency_detail = ""
    if revenue_value is not None and prior:
        ratio = revenue_value / prior
        efficiency_score = sig.piecewise(ratio, [(0, 0.1), (0.2, 0.4), (0.5, 0.7), (1.0, 0.9), (2.0, 1.0)])
        efficiency_detail = f"${revenue_value:,.0f} revenue on ${prior:,.0f} raised."

    return [
        Evaluated(
            "gross_margin", "Gross margin", 2.5,
            sig.piecewise(margin, [(0, 0.05), (20, 0.3), (40, 0.55), (60, 0.75), (80, 1.0)])
            if margin is not None else None,
            f"{margin:.0f}% gross margin." if margin is not None else "",
        ),
        Evaluated(
            "runway", "Runway", 2.5,
            sig.piecewise(runway, [(0, 0.0), (3, 0.2), (6, 0.45), (9, 0.65), (12, 0.85), (18, 1.0)])
            if runway is not None else None,
            f"{runway:.0f} months of runway." if runway is not None else "",
        ),
        Evaluated(
            "burn_clarity", "Burn disclosed", 1.5,
            0.9 if burn is not None else None,
            f"${burn:,.0f} monthly burn disclosed." if burn is not None else "",
        ),
        Evaluated("capital_efficiency", "Capital efficiency", 1.5, efficiency_score, efficiency_detail),
        Evaluated(
            "revenue_quality", "Revenue quality", 1.0,
            0.9 if profile.has("arr") or profile.has("mrr") else (0.5 if revenue_value else None),
            "Recurring revenue reported." if profile.has("arr") or profile.has("mrr") else "",
        ),
        Evaluated(
            "financial_clarity", "Financial model", 1.0,
            0.85 if sig.has_financial_model(profile) else None,
            "A financial model or projections are included.",
        ),
    ]


# ---------------------------------------------------------------------------
# F. Financing Terms
# ---------------------------------------------------------------------------
def _score_terms(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Evaluated]:
    valuation = (
        profile.number("pre_money_valuation")
        or profile.number("note_cap")
        or profile.number("safe_cap")
    )
    raise_amount = profile.number("target_raise")

    valuation_score, valuation_detail = _benchmark_score(
        valuation, cfg, stage, "pre_money_typical", sector, "Valuation"
    )
    raise_score, raise_detail = _benchmark_score(
        raise_amount, cfg, stage, "raise_typical", sector, "Round size"
    )

    disclosed = sum(
        1 for f in ("discount", "interest_rate", "maturity_date", "minimum_investment", "note_cap", "safe_cap")
        if profile.has(f)
    )
    return [
        Evaluated("valuation_fit", "Valuation vs. stage benchmark", 3.0, valuation_score, valuation_detail),
        Evaluated("round_size_fit", "Round size vs. stage benchmark", 3.0, raise_score, raise_detail),
        Evaluated(
            "instrument", "Instrument suitability", 2.0,
            0.9 if profile.has("security_type") else None,
            f"Instrument stated: {profile.text('security_type')[:60]}."
            if profile.has("security_type") else "",
        ),
        Evaluated(
            "terms_disclosure", "Terms disclosure", 2.0,
            sig.piecewise(float(disclosed), [(0, 0.0), (1, 0.4), (2, 0.6), (3, 0.85), (5, 1.0)])
            if disclosed else None,
            f"{disclosed} key deal terms disclosed." if disclosed else "",
        ),
    ]


def _benchmark_score(
    value: float | None,
    cfg: AppConfig,
    stage: Stage,
    key: str,
    sector: SectorProfile,
    label: str,
) -> tuple[float | None, str]:
    """Score how well a value sits inside its stage benchmark range.

    Both ends are penalised: an oversized ask is a financing risk, and an
    unusually small one usually means the round will not fund the milestones.
    """
    if value is None:
        return None, ""
    rng = sig.benchmark_range(cfg, stage, key, sector)
    if not rng:
        return None, ""
    low, high = rng
    midpoint = (low + high) / 2
    if value < low:
        score = sig.piecewise(value / low, [(0.0, 0.3), (0.5, 0.6), (1.0, 0.9)])
        note = "below"
    elif value <= high:
        score = 1.0
        note = "within"
    else:
        score = sig.piecewise(value / high, [(1.0, 0.9), (1.5, 0.6), (2.0, 0.35), (3.0, 0.15), (5.0, 0.05)])
        note = "above"
    detail = (
        f"{label} of ${value:,.0f} is {note} the ${low:,.0f}-${high:,.0f} "
        f"{stage.value} range (midpoint ${midpoint:,.0f})."
    )
    return score, detail


# ---------------------------------------------------------------------------
# G. Product & Defensibility
# ---------------------------------------------------------------------------
def _score_defensibility(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Evaluated]:
    return [
        Evaluated(
            "ip", "Intellectual property", 3.0,
            0.9 if profile.has("patents") else (0.7 if profile.has("intellectual_property") else None),
            f"IP: {(profile.text('patents') or profile.text('intellectual_property'))[:80]}."
            if sig.has_ip(profile) else "",
        ),
        Evaluated(
            "differentiation", "Competitive differentiation", 2.0,
            0.8 if sig.has_differentiation(profile) else None,
            "Differentiation or competitive positioning is articulated.",
        ),
        Evaluated(
            "technology_readiness", "Technology readiness", 2.0,
            0.85 if profile.has("technology_readiness") or profile.has("product_stage") else None,
            f"Product stage: {profile.text('product_stage')[:70]}."
            if profile.has("product_stage") else "",
        ),
        Evaluated(
            "regulatory_barrier", "Regulatory barrier to entry", 1.5,
            (0.9 if sig.has_regulatory_milestone(profile) else 0.6) if sector.regulated else None,
            "Regulatory requirements raise the barrier to entry." if sector.regulated else "",
        ),
        Evaluated(
            "switching_costs", "Switching costs / lock-in", 1.5,
            0.75 if profile.has("renewal_rate") or profile.has("average_contract_value") else None,
            "Contract or retention structure implies switching costs.",
        ),
    ]


# ---------------------------------------------------------------------------
# H. Investor Fit
# ---------------------------------------------------------------------------
def _score_investor_fit(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Evaluated]:
    raise_amount = profile.number("target_raise")
    within_raise = sig.within_benchmark(raise_amount, cfg, stage, "raise_typical", sector)
    revenue_value = sig.revenue(profile)

    maturity_score = None
    maturity_detail = ""
    if stage.is_known:
        expected_from = stage_from_text(sector.revenue_expected_from)
        if revenue_value is not None and revenue_value > 0:
            maturity_score = 1.0
            maturity_detail = "Revenue is present, matching investor expectations for the stage."
        elif at_least(stage, expected_from):
            maturity_score = 0.2
            maturity_detail = (
                f"No revenue at {stage.value}, whereas {sector.label} investors typically "
                f"expect revenue from {expected_from.value}."
            )
        else:
            maturity_score = 0.75
            maturity_detail = (
                f"Pre-revenue is normal for {sector.label} at {stage.value}."
            )

    return [
        Evaluated(
            "stage_size_match", "Ask matches stage norms", 3.0,
            None if within_raise is None else (1.0 if within_raise else 0.3),
            "Round size sits inside the stage norm." if within_raise
            else ("Round size sits outside the stage norm." if within_raise is False else ""),
        ),
        Evaluated(
            "sector_investability", "Sector has an active investor pool", 2.0,
            _sector_appetite(sector),
            sector.note,
        ),
        Evaluated("maturity_match", "Company maturity matches the ask", 2.0, maturity_score, maturity_detail),
        Evaluated(
            "geography", "Geography identified", 1.5,
            0.8 if profile.has("headquarters") or profile.has("primary_geography")
            or profile.has("geographic_market") else None,
            "Geography stated.",
        ),
        Evaluated(
            "check_size", "Check size defined", 1.5,
            0.8 if profile.has("minimum_investment") else None,
            "Minimum investment stated.",
        ),
    ]


# ---------------------------------------------------------------------------
# I. Fundraising Readiness
# ---------------------------------------------------------------------------
def _score_fundraising_readiness(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Evaluated]:
    quality = profile.extraction_quality
    materials_score = None
    if quality.pages_total:
        # A readable deck of reasonable length is itself evidence that the
        # investor materials exist and are usable.
        length_score = sig.piecewise(float(quality.pages_total), [(3, 0.3), (10, 0.8), (15, 1.0), (40, 0.9), (80, 0.6)])
        readable = 1.0 - quality.image_based_ratio
        materials_score = max(0.0, min(1.0, length_score * (0.5 + 0.5 * readable)))

    return [
        Evaluated(
            "use_of_funds", "Use of proceeds", 1.5,
            0.9 if profile.has("use_of_funds") else None,
            f"Use of funds: {profile.text('use_of_funds')[:70]}." if profile.has("use_of_funds") else "",
        ),
        Evaluated(
            "milestones", "Milestones defined", 1.0,
            0.85 if sig.has_milestones(profile) else None,
            "Milestones or a roadmap are presented.",
        ),
        Evaluated(
            "forecast", "Financial forecast", 1.0,
            0.85 if sig.has_financial_model(profile) else None,
            "Financial projections are included.",
        ),
        Evaluated(
            "materials", "Investor materials", 1.0, materials_score,
            f"{quality.pages_total} pages, {len(quality.image_based_pages)} image-based."
            if quality.pages_total else "",
        ),
        Evaluated(
            "strategy", "Financing strategy", 0.5,
            0.8 if profile.has("security_type") and profile.has("target_raise") else None,
            "Instrument and target raise are both defined.",
        ),
    ]


SCORERS: dict[str, Scorer] = {
    "traction_revenue": _score_traction,
    "round_progress": _score_round_progress,
    "market_opportunity": _score_market,
    "team": _score_team,
    "financial_quality": _score_financial_quality,
    "financing_terms": _score_terms,
    "product_defensibility": _score_defensibility,
    "investor_fit": _score_investor_fit,
    "fundraising_readiness": _score_fundraising_readiness,
}
