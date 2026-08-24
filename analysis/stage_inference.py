"""Model-inferred company stage.

A company's own label is only one input. Prior capital, revenue, product
maturity, traction, team size, round size, valuation and institutional
participation each vote for a stage; the winner is the modal vote, weighted by
how diagnostic each signal is.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from config.settings import AppConfig, load_config
from models.company import Stage, StageAssessment, stage_from_text
from models.financing import effective_annual_revenue, total_customers
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)

#: How diagnostic each signal is for stage. Prior capital and revenue are the
#: most reliable; a round size can be distorted by capital intensity.
SIGNAL_WEIGHTS: dict[str, float] = {
    "prior_funding": 2.0,
    "revenue": 2.0,
    "round_size": 1.25,
    "valuation": 1.25,
    "headcount": 1.0,
    "product_maturity": 1.0,
    "traction": 1.0,
    "institutional": 0.75,
}

_INSTITUTIONAL_MARKERS = re.compile(
    r"\b(?:ventures?|capital|partners|vc\b|fund\b|sequoia|andreessen|accel|"
    r"khosla|lightspeed|bessemer|institutional\s+investor|series\s+[a-c]\s+lead)\b",
    re.IGNORECASE,
)
_LATE_PRODUCT_MARKERS = ("commercially available", "generally available", "in market", "shipping", "scaling")
_MID_PRODUCT_MARKERS = ("beta", "pilot", "first revenue", "launched", "commercial launch")
_EARLY_PRODUCT_MARKERS = ("prototype", "mvp", "proof of concept", "poc", "preclinical", "concept", "design")


def infer_stage(profile: DeckProfile, config: AppConfig | None = None) -> StageAssessment:
    """Reconcile the declared stage with what the numbers actually imply."""
    cfg = config or load_config()
    declared = stage_from_text(profile.text("stage_declared"))

    votes: dict[Stage, float] = defaultdict(float)
    rationale: list[str] = []
    signals_used = 0

    for signal, stage, detail in _collect_signals(profile, cfg):
        weight = SIGNAL_WEIGHTS.get(signal, 1.0)
        votes[stage] += weight
        rationale.append(detail)
        signals_used += 1

    if not votes:
        LOGGER.debug("No stage signals available; falling back to the declared stage.")
        return StageAssessment(
            declared=declared,
            inferred=declared,
            confidence="LOW",
            rationale=["No quantitative stage signals were available in the deck."],
            signals_used=0,
        )

    # The declared stage gets a vote of its own, but never a decisive one.
    if declared.is_known:
        votes[declared] += 1.0
        rationale.insert(0, f"Deck describes the round as {declared.value}.")

    # Stage is ordinal, so the weighted mean position is more faithful than a
    # modal vote: signals pointing at Pre-Seed and Series A should resolve to
    # Seed rather than to whichever of the two happened to win a tie.
    total_weight = sum(votes.values())
    mean_index = sum(_stage_index(cfg, s) * w for s, w in votes.items()) / total_weight
    best_index = max(0, min(round(mean_index), len(cfg.stage_order) - 1))
    best_stage = Stage(cfg.stage_order[best_index])
    dominance = votes[best_stage] / total_weight if total_weight else 0.0

    confidence = _confidence_for(signals_used, dominance, declared, best_stage)
    return StageAssessment(
        declared=declared,
        inferred=best_stage,
        confidence=confidence,
        rationale=rationale[:8],
        signals_used=signals_used,
    )


def _collect_signals(profile: DeckProfile, cfg: AppConfig) -> list[tuple[str, Stage, str]]:
    signals: list[tuple[str, Stage, str]] = []
    benchmarks = cfg.stage_benchmarks

    prior = profile.number("prior_funding") or profile.number("total_funding_to_date")
    if prior is not None:
        stage = _stage_from_prior_funding(prior, benchmarks, cfg.stage_order)
        signals.append(("prior_funding", stage, f"Prior funding of ${prior:,.0f} is typical of {stage.value}."))

    revenue = effective_annual_revenue(profile)
    if revenue is not None:
        stage = _stage_from_range(revenue, benchmarks, cfg.stage_order, "revenue_typical")
        signals.append(("revenue", stage, f"Revenue of ${revenue:,.0f} is typical of {stage.value}."))

    raise_amount = profile.number("target_raise")
    if raise_amount is not None:
        stage = _stage_from_range(raise_amount, benchmarks, cfg.stage_order, "raise_typical")
        signals.append(("round_size", stage, f"A ${raise_amount:,.0f} round is typical of {stage.value}."))

    valuation = profile.number("pre_money_valuation") or profile.number("note_cap") or profile.number("safe_cap")
    if valuation is not None:
        stage = _stage_from_range(valuation, benchmarks, cfg.stage_order, "pre_money_typical")
        signals.append(("valuation", stage, f"A ${valuation:,.0f} valuation is typical of {stage.value}."))

    headcount = profile.number("employees")
    if headcount is not None:
        stage = _stage_from_range(headcount, benchmarks, cfg.stage_order, "headcount_typical")
        signals.append(("headcount", stage, f"A team of {headcount:.0f} is typical of {stage.value}."))

    product_stage = _stage_from_product(profile)
    if product_stage is not None:
        stage, detail = product_stage
        signals.append(("product_maturity", stage, detail))

    customers = total_customers(profile)
    if customers is not None:
        if customers >= 100:
            stage = Stage.SERIES_B
        elif customers >= 20:
            stage = Stage.SERIES_A
        elif customers >= 3:
            stage = Stage.SEED
        else:
            stage = Stage.PRE_SEED
        signals.append(("traction", stage, f"{customers:.0f} customers is typical of {stage.value}."))

    investors = profile.text("existing_investors") or profile.text("lead_investor")
    if investors and _INSTITUTIONAL_MARKERS.search(investors):
        signals.append(
            ("institutional", Stage.SERIES_A, "Institutional investors are named in the deck.")
        )

    return signals


def _stage_index(cfg: AppConfig, stage: Stage) -> int:
    try:
        return cfg.stage_order.index(stage.value)
    except ValueError:
        return len(cfg.stage_order)


def _stage_from_range(
    value: float, benchmarks: dict, order: list[str], key: str
) -> Stage:
    """The latest stage whose typical range this value reaches."""
    best = Stage.PRE_SEED
    for name in order:
        low, high = benchmarks[name][key]
        if value >= low:
            best = Stage(name)
        if low <= value <= high:
            return Stage(name)
    return best


def _stage_from_prior_funding(value: float, benchmarks: dict, order: list[str]) -> Stage:
    """Prior capital places a company at the stage *after* what it raised."""
    for index, name in enumerate(order):
        if value <= benchmarks[name]["prior_funding_max"]:
            # Having raised up to a stage's ceiling implies being at that stage.
            return Stage(order[min(index, len(order) - 1)])
    return Stage(order[-1])


def _stage_from_product(profile: DeckProfile) -> tuple[Stage, str] | None:
    text = " ".join(
        filter(
            None,
            [
                profile.text("product_stage"),
                profile.text("commercial_product_status"),
                profile.text("mvp_status"),
                profile.text("technology_readiness"),
            ],
        )
    ).lower()
    if not text:
        return None
    for markers, stage in (
        (_LATE_PRODUCT_MARKERS, Stage.SERIES_A),
        (_MID_PRODUCT_MARKERS, Stage.SEED),
        (_EARLY_PRODUCT_MARKERS, Stage.PRE_SEED),
    ):
        for marker in markers:
            if marker in text:
                return stage, f"Product described as '{marker}', typical of {stage.value}."
    return None


def _confidence_for(
    signals_used: int, dominance: float, declared: Stage, inferred: Stage
) -> str:
    if signals_used >= 4 and dominance >= 0.5:
        base = "HIGH"
    elif signals_used >= 3 and dominance >= 0.35:
        base = "MEDIUM"
    elif signals_used >= 2:
        base = "MEDIUM"
    else:
        base = "LOW"
    # Disagreement between what the company says and what the numbers say is
    # itself a reason to be less certain.
    if declared.is_known and declared is not inferred and base == "HIGH":
        return "MEDIUM"
    return base
