"""Contradiction checks.

The rule throughout is to *flag*, never to silently correct. If a deck says
two incompatible things, that is information about the deck, and quietly
overwriting one of the values would hide it from the reader.
"""

from __future__ import annotations

import logging

from analysis.stage_inference import infer_stage
from extractors.normalization import format_money, months_until
from models.company import Stage, at_least
from models.evidence import Inconsistency
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)

#: Tolerance when comparing a stated percentage against a computed one.
PERCENT_TOLERANCE = 5.0
#: Tolerance when comparing stated runway against cash / burn.
RUNWAY_TOLERANCE_MONTHS = 2.0

_PRE_REVENUE_MARKERS = ("pre-revenue", "pre revenue", "prerevenue", "no revenue yet")


def check_consistency(profile: DeckProfile) -> list[Inconsistency]:
    """Return every contradiction found, and record it on the profile."""
    found: list[Inconsistency] = []
    for check in (
        _check_pre_revenue_contradiction,
        _check_committed_exceeds_target,
        _check_percent_committed_mismatch,
        _check_valuation_ordering,
        _check_runway_vs_cash_burn,
        _check_stage_without_prior_funding,
        _check_negative_values,
        _check_note_maturity_in_past,
        _check_market_sizing_order,
    ):
        result = check(profile)
        if result:
            found.append(result)

    for inc in found:
        profile.add_inconsistency(inc)
        LOGGER.info("Consistency flag [%s]: %s", inc.code, inc.message)
    return found


def _check_pre_revenue_contradiction(profile: DeckProfile) -> Inconsistency | None:
    revenue = profile.number("annual_revenue") or profile.number("arr")
    if not revenue or revenue <= 0:
        return None
    haystack = " ".join(profile.raw_text_by_page.values()).lower()
    if not any(marker in haystack for marker in _PRE_REVENUE_MARKERS):
        return None
    return Inconsistency(
        code="pre_revenue_with_revenue",
        message=(
            f"The deck describes the company as pre-revenue but also reports "
            f"{format_money(revenue)} of revenue."
        ),
        fields=["annual_revenue", "arr"],
    )


def _check_committed_exceeds_target(profile: DeckProfile) -> Inconsistency | None:
    committed = profile.number("amount_committed")
    target = profile.number("target_raise")
    if committed is None or target is None or committed <= target:
        return None
    return Inconsistency(
        code="committed_exceeds_target",
        message=(
            f"Amount committed ({format_money(committed)}) exceeds the target raise "
            f"({format_money(target)})."
        ),
        fields=["amount_committed", "target_raise"],
    )


def _check_percent_committed_mismatch(profile: DeckProfile) -> Inconsistency | None:
    stated = profile.get("percent_committed")
    committed = profile.number("amount_committed")
    target = profile.number("target_raise")
    # Only meaningful when the percentage was stated independently, not derived.
    if not stated.is_known or stated.status.value == "CALCULATED":
        return None
    if committed is None or not target:
        return None
    computed = committed / target * 100.0
    if abs(computed - float(stated.value)) <= PERCENT_TOLERANCE:
        return None
    return Inconsistency(
        code="percent_committed_mismatch",
        message=(
            f"Stated percentage committed ({stated.value:.0f}%) does not match the "
            f"amounts given ({format_money(committed)} of {format_money(target)} = "
            f"{computed:.0f}%)."
        ),
        fields=["percent_committed", "amount_committed", "target_raise"],
    )


def _check_valuation_ordering(profile: DeckProfile) -> Inconsistency | None:
    pre = profile.number("pre_money_valuation")
    post = profile.number("post_money_valuation")
    if pre is None or post is None or post >= pre:
        return None
    return Inconsistency(
        code="post_below_pre_money",
        message=(
            f"Post-money valuation ({format_money(post)}) is below the pre-money "
            f"valuation ({format_money(pre)})."
        ),
        fields=["pre_money_valuation", "post_money_valuation"],
    )


def _check_runway_vs_cash_burn(profile: DeckProfile) -> Inconsistency | None:
    stated = profile.get("runway_months")
    if not stated.is_known or stated.status.value == "CALCULATED":
        return None
    cash = profile.number("cash_balance")
    burn = profile.number("burn_rate")
    if cash is None or not burn or burn <= 0:
        return None
    computed = cash / burn
    if abs(computed - float(stated.value)) <= RUNWAY_TOLERANCE_MONTHS:
        return None
    return Inconsistency(
        code="runway_mismatch",
        message=(
            f"Stated runway ({stated.value:.0f} months) is inconsistent with cash "
            f"({format_money(cash)}) divided by burn ({format_money(burn)}/mo = "
            f"{computed:.0f} months)."
        ),
        fields=["runway_months", "cash_balance", "burn_rate"],
    )


def _check_stage_without_prior_funding(profile: DeckProfile) -> Inconsistency | None:
    assessment = infer_stage(profile)
    stage = assessment.declared if assessment.declared.is_known else assessment.inferred
    if not at_least(stage, Stage.SERIES_A):
        return None
    if profile.has("prior_funding") or profile.has("total_funding_to_date"):
        return None
    return Inconsistency(
        code="late_stage_no_prior_funding",
        message=(
            f"The round is presented as {stage.value} but no prior financing is "
            "stated anywhere in the deck."
        ),
        fields=["stage_declared", "prior_funding"],
    )


def _check_negative_values(profile: DeckProfile) -> Inconsistency | None:
    negatives = [
        name
        for name in ("target_raise", "amount_committed", "annual_revenue", "arr", "customers")
        if (value := profile.number(name)) is not None and value < 0
    ]
    if not negatives:
        return None
    return Inconsistency(
        code="negative_values",
        message=f"Negative values were extracted for: {', '.join(negatives)}.",
        fields=negatives,
    )


def _check_note_maturity_in_past(profile: DeckProfile) -> Inconsistency | None:
    maturity = profile.value("maturity_date")
    if not maturity:
        return None
    months = months_until(str(maturity))
    if months is None or months >= 0:
        return None
    return Inconsistency(
        code="maturity_passed",
        message=f"The stated note maturity date ({maturity}) has already passed.",
        fields=["maturity_date"],
        severity="HIGH",
    )


def _check_market_sizing_order(profile: DeckProfile) -> Inconsistency | None:
    tam, sam, som = (profile.number(n) for n in ("tam", "sam", "som"))
    problems: list[str] = []
    if tam is not None and sam is not None and sam > tam:
        problems.append("SAM exceeds TAM")
    if sam is not None and som is not None and som > sam:
        problems.append("SOM exceeds SAM")
    if not problems:
        return None
    return Inconsistency(
        code="market_sizing_order",
        message=f"Market sizing is internally inconsistent: {'; '.join(problems)}.",
        fields=["tam", "sam", "som"],
    )
