"""Financing-shaped views and helpers over a :class:`DeckProfile`."""

from __future__ import annotations

from pydantic import BaseModel

from models.profile import DeckProfile


class RoundSnapshot(BaseModel):
    """The numbers that drive the round-progress side of the model.

    Every field is optional; ``None`` means "the deck did not say", which the
    model treats very differently from zero.
    """

    target_raise: float | None = None
    amount_committed: float | None = None
    percent_committed: float | None = None
    pre_money_valuation: float | None = None
    post_money_valuation: float | None = None
    instrument: str | None = None
    lead_investor: str | None = None
    has_lead_investor: bool | None = None
    prior_funding: float | None = None
    runway_months: float | None = None
    use_of_funds: str | None = None

    @property
    def committed_ratio(self) -> float | None:
        """Round progress as a percentage, from either the stated percentage
        or the committed/target pair."""
        if self.percent_committed is not None:
            return self.percent_committed
        if self.amount_committed is not None and self.target_raise:
            return self.amount_committed / self.target_raise * 100.0
        return None


def round_snapshot(profile: DeckProfile) -> RoundSnapshot:
    lead = profile.text("lead_investor") or None
    has_lead: bool | None
    if lead is None:
        has_lead = None
    else:
        has_lead = _is_affirmative_lead(lead)
    return RoundSnapshot(
        target_raise=profile.number("target_raise"),
        amount_committed=profile.number("amount_committed"),
        percent_committed=profile.number("percent_committed"),
        pre_money_valuation=profile.number("pre_money_valuation"),
        post_money_valuation=profile.number("post_money_valuation"),
        instrument=profile.text("security_type") or None,
        lead_investor=lead,
        has_lead_investor=has_lead,
        prior_funding=profile.number("prior_funding"),
        runway_months=profile.number("runway_months"),
        use_of_funds=profile.text("use_of_funds") or None,
    )


#: Phrases that appear where a lead investor would be named but which mean the
#: opposite. Without this, "seeking a lead investor" would score as a
#: committed lead.
_NEGATIVE_LEAD_PHRASES = (
    "seeking",
    "looking for",
    "no lead",
    "none",
    "tbd",
    "to be determined",
    "in discussion",
    "in discussions",
    "identifying",
    "open",
    "n/a",
    "not yet",
    "pending",
)


def _is_affirmative_lead(text: str) -> bool:
    low = text.strip().lower()
    if not low:
        return False
    return not any(phrase in low for phrase in _NEGATIVE_LEAD_PHRASES)


def effective_annual_revenue(profile: DeckProfile) -> float | None:
    """Best available annual revenue figure: stated revenue, else ARR, else
    MRR annualised. Returns ``None`` when the deck says nothing at all."""
    for name in ("annual_revenue", "arr"):
        v = profile.number(name)
        if v is not None:
            return v
    mrr = profile.number("mrr")
    if mrr is not None:
        return mrr * 12.0
    return None


def total_customers(profile: DeckProfile) -> float | None:
    """Widest credible customer count across the customer-ish fields."""
    candidates = [
        profile.number(n)
        for n in ("customers", "paying_customers", "enterprise_customers")
    ]
    known = [c for c in candidates if c is not None]
    return max(known) if known else None


def commercial_engagement_count(profile: DeckProfile) -> float | None:
    """Customers plus pre-revenue commercial evidence (pilots, LOIs,
    contracts). Used for sectors where paid revenue is not yet the norm."""
    parts = [
        total_customers(profile),
        profile.number("pilots"),
        profile.number("lois"),
        profile.number("contracts"),
    ]
    known = [p for p in parts if p is not None]
    return sum(known) if known else None
