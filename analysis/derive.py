"""Fill in values that follow arithmetically from what the deck stated.

Every value produced here is recorded as CALCULATED, never EXPLICIT, so the
evidence trail always distinguishes "the deck said this" from "we worked this
out". Nothing is derived unless all of its inputs are known.
"""

from __future__ import annotations

import logging

from models.evidence import ValueStatus
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)

#: Guard rails: a derived value outside these bounds means the inputs were
#: mis-extracted, so the derivation is abandoned rather than published.
MAX_RUNWAY_MONTHS = 120.0
MAX_PERCENT = 1000.0


def derive_fields(profile: DeckProfile) -> None:
    """Populate calculable fields in place."""
    _derive_percent_committed(profile)
    _derive_amount_committed(profile)
    _derive_valuations(profile)
    _derive_runway(profile)
    _derive_recurring_revenue(profile)
    _derive_total_funding(profile)


def _put(
    profile: DeckProfile,
    field: str,
    value: float,
    confidence: float,
    note: str,
    sources: tuple[str, ...],
) -> None:
    pages = [profile.get(s).source_page for s in sources if profile.get(s).source_page]
    profile.put(
        field,
        value,
        ValueStatus.CALCULATED,
        confidence,
        source_page=pages[0] if pages else None,
        source_text=" + ".join(f"{s}={profile.value(s)}" for s in sources),
        note=note,
    )
    LOGGER.debug("Derived %s = %s (%s)", field, value, note)


def _confidence_of(profile: DeckProfile, *fields: str) -> float:
    """A derived value is only as trustworthy as its weakest input."""
    values = [profile.get(f).confidence for f in fields if profile.has(f)]
    return round(min(values) * 0.95, 3) if values else 0.5


def _derive_percent_committed(profile: DeckProfile) -> None:
    if profile.has("percent_committed"):
        return
    committed = profile.number("amount_committed")
    target = profile.number("target_raise")
    if committed is None or not target:
        return
    pct = committed / target * 100.0
    if 0.0 <= pct <= MAX_PERCENT:
        _put(
            profile, "percent_committed", round(pct, 1),
            _confidence_of(profile, "amount_committed", "target_raise"),
            "Committed amount divided by target raise.",
            ("amount_committed", "target_raise"),
        )


def _derive_amount_committed(profile: DeckProfile) -> None:
    if profile.has("amount_committed"):
        return
    pct = profile.number("percent_committed")
    target = profile.number("target_raise")
    if pct is None or not target:
        return
    _put(
        profile, "amount_committed", round(target * pct / 100.0, 2),
        _confidence_of(profile, "percent_committed", "target_raise"),
        "Target raise multiplied by the stated percentage committed.",
        ("percent_committed", "target_raise"),
    )


def _derive_valuations(profile: DeckProfile) -> None:
    pre = profile.number("pre_money_valuation")
    post = profile.number("post_money_valuation")
    raise_amount = profile.number("target_raise")
    if raise_amount is None:
        return
    if pre is not None and post is None:
        _put(
            profile, "post_money_valuation", pre + raise_amount,
            _confidence_of(profile, "pre_money_valuation", "target_raise"),
            "Pre-money valuation plus the target raise.",
            ("pre_money_valuation", "target_raise"),
        )
    elif post is not None and pre is None and post > raise_amount:
        _put(
            profile, "pre_money_valuation", post - raise_amount,
            _confidence_of(profile, "post_money_valuation", "target_raise"),
            "Post-money valuation less the target raise.",
            ("post_money_valuation", "target_raise"),
        )


def _derive_runway(profile: DeckProfile) -> None:
    if profile.has("runway_months"):
        return
    cash = profile.number("cash_balance")
    burn = profile.number("burn_rate")
    if cash is None or not burn or burn <= 0:
        return
    months = cash / burn
    if 0 < months <= MAX_RUNWAY_MONTHS:
        _put(
            profile, "runway_months", round(months, 1),
            _confidence_of(profile, "cash_balance", "burn_rate"),
            "Cash balance divided by monthly burn rate.",
            ("cash_balance", "burn_rate"),
        )


def _derive_recurring_revenue(profile: DeckProfile) -> None:
    mrr = profile.number("mrr")
    arr = profile.number("arr")
    if arr is None and mrr is not None:
        _put(
            profile, "arr", round(mrr * 12.0, 2),
            _confidence_of(profile, "mrr"),
            "Monthly recurring revenue annualised.",
            ("mrr",),
        )
        arr = mrr * 12.0
    if not profile.has("annual_revenue") and arr is not None:
        _put(
            profile, "annual_revenue", round(arr, 2),
            _confidence_of(profile, "arr") if profile.has("arr") else _confidence_of(profile, "mrr"),
            "Annual revenue taken from recurring revenue.",
            ("arr",) if profile.has("arr") else ("mrr",),
        )


def _derive_total_funding(profile: DeckProfile) -> None:
    if profile.has("total_funding_to_date"):
        return
    prior = profile.number("prior_funding")
    if prior is None:
        return
    _put(
        profile, "total_funding_to_date", prior,
        _confidence_of(profile, "prior_funding"),
        "Total funding to date taken from prior funding raised.",
        ("prior_funding",),
    )
