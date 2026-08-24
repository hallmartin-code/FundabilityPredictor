"""Critical financing blockers.

Each rule returns a :class:`Blocker` or ``None``. Rules describe what would
actually stop this round from closing, ranked CRITICAL / HIGH / MODERATE, and
are phrased with the company's own numbers so a founder can act on them.
"""

from __future__ import annotations

import logging
from typing import Callable

from analysis import signals as sig
from config.settings import AppConfig
from extractors.normalization import format_money
from models.company import SectorProfile, Stage, at_least, stage_from_text
from models.prediction import Blocker
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)

#: How many blockers the report shows.
MAX_BLOCKERS = 5

BlockerRule = Callable[[DeckProfile, SectorProfile, Stage, AppConfig], "Blocker | None"]


def _valuation_of(profile: DeckProfile) -> float | None:
    return (
        profile.number("pre_money_valuation")
        or profile.number("note_cap")
        or profile.number("safe_cap")
    )


def _valuation_too_high(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Blocker | None:
    valuation = _valuation_of(p)
    multiple = sig.benchmark_multiple(valuation, cfg, st, "pre_money_typical", s)
    if multiple is None or multiple < 1.5:
        return None
    rng = sig.benchmark_range(cfg, st, "pre_money_typical", s)
    severity = "CRITICAL" if multiple >= 2.5 else "HIGH"
    return Blocker(
        key="valuation_above_tolerance",
        severity=severity,
        title="Valuation exceeds likely investor tolerance",
        detail=(
            f"The {format_money(valuation)} valuation sits {multiple:.1f}x above the top of the "
            f"{format_money(rng[0])}-{format_money(rng[1])} range typical for {st.value} "
            f"{s.label} companies."
        ),
    )


def _round_oversized(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Blocker | None:
    target = p.number("target_raise")
    multiple = sig.benchmark_multiple(target, cfg, st, "raise_typical", s)
    if multiple is None or multiple < 1.5:
        return None
    rng = sig.benchmark_range(cfg, st, "raise_typical", s)
    return Blocker(
        key="round_too_large",
        severity="CRITICAL" if multiple >= 2.5 else "HIGH",
        title="Round size is large for the stage",
        detail=(
            f"The {format_money(target)} ask is {multiple:.1f}x the top of the "
            f"{format_money(rng[0])}-{format_money(rng[1])} range typical for {st.value} "
            f"{s.label} rounds, which narrows the pool of investors able to lead."
        ),
    )


def _no_lead(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Blocker | None:
    if sig.has_lead_investor(p) is True:
        return None
    pct = sig.committed_pct(p)
    stated = "no lead investor is named" if not p.has("lead_investor") else "no lead is committed"
    progress = f" with {pct:.0f}% of the round committed" if pct is not None else ""
    return Blocker(
        key="no_lead_investor",
        severity="CRITICAL" if (pct is None or pct < 25) else "HIGH",
        title="No financing lead",
        detail=(
            f"The deck shows {stated}{progress}. Rounds without a lead typically take "
            "materially longer to close because no investor is setting terms."
        ),
    )


def _insufficient_runway(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Blocker | None:
    runway = sig.runway_months(p)
    if runway is None or runway >= 9:
        return None
    return Blocker(
        key="inadequate_runway",
        severity="CRITICAL" if runway < 6 else "HIGH",
        title="Inadequate runway",
        detail=(
            f"At {runway:.0f} months of runway the company will be negotiating from a "
            "weakening position, and investors will price that in."
        ),
    )


def _no_commercial_proof(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Blocker | None:
    revenue = sig.revenue(p)
    engagements = sig.commercial_engagements(p)
    if (revenue and revenue > 0) or (engagements and engagements > 0):
        return None
    expected_from = stage_from_text(s.revenue_expected_from)
    if not at_least(st, Stage.SEED):
        return None
    severity = "CRITICAL" if at_least(st, expected_from) else "HIGH"
    return Blocker(
        key="no_commercial_proof",
        severity=severity,
        title="Insufficient proof of commercial demand",
        detail=(
            f"No revenue, customers, pilots, letters of intent or signed contracts were found "
            f"in the deck. At {st.value}, {s.label} investors expect at least documented "
            "customer commitments."
        ),
    )


def _regulatory_gap(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Blocker | None:
    if not sig.has_long_regulatory_path(p, s):
        return None
    runway = sig.runway_months(p)
    tail = (
        f" Stated runway of {runway:.0f} months does not cover that path."
        if runway is not None and runway < 18
        else ""
    )
    return Blocker(
        key="regulatory_progress",
        severity="HIGH",
        title="Insufficient regulatory progress",
        detail=(
            f"No regulatory clearance or approval has been achieved, and {s.label} financing is "
            f"typically gated on that milestone.{tail}"
        ),
    )


def _weak_financial_visibility(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Blocker | None:
    missing = [
        label
        for field, label in (
            ("burn_rate", "burn rate"),
            ("runway_months", "runway"),
            ("gross_margin", "gross margin"),
            ("cash_balance", "cash balance"),
        )
        if not p.has(field)
    ]
    if len(missing) < 3:
        return None
    return Blocker(
        key="weak_financial_visibility",
        severity="HIGH" if len(missing) == 4 else "MODERATE",
        title="Weak financial visibility",
        detail=(
            f"The deck does not disclose {sig.join_readable(missing)}. Investors cannot size the ask "
            "or assess urgency without those figures, which extends diligence."
        ),
    )


def _customer_concentration(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Blocker | None:
    concentration = p.number("customer_concentration")
    if concentration is None or concentration < 40:
        return None
    return Blocker(
        key="customer_concentration",
        severity="HIGH" if concentration >= 60 else "MODERATE",
        title="Customer concentration",
        detail=(
            f"{concentration:.0f}% of revenue depends on a single customer, which investors "
            "treat as a material revenue risk."
        ),
    )


def _missing_management(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Blocker | None:
    full_time = sig.founder_is_full_time(p)
    gaps: list[str] = []
    if full_time is False:
        gaps.append("the founding team is not full-time")
    if not p.has("commercial_expertise") and at_least(st, Stage.SEED):
        gaps.append("no commercial or go-to-market leader is identified")
    if p.has("missing_roles"):
        gaps.append("the deck itself flags open leadership roles")
    if not gaps:
        return None
    return Blocker(
        key="management_gaps",
        severity="HIGH" if full_time is False else "MODERATE",
        title="Missing management capabilities",
        detail=f"Team gaps that investors will probe: {'; '.join(gaps)}.",
    )


def _materials_quality(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Blocker | None:
    quality = p.extraction_quality
    problems: list[str] = []
    if quality.is_image_heavy:
        problems.append(
            f"{len(quality.image_based_pages)} of {quality.pages_total} pages carry no "
            "machine-readable text"
        )
    if not p.has("use_of_funds"):
        problems.append("no use of proceeds is given")
    if not sig.has_financial_model(p):
        problems.append("no financial model or projections are included")
    if len(problems) < 2:
        return None
    return Blocker(
        key="materials_incomplete",
        severity="MODERATE",
        title="Lack of institutional-quality materials",
        detail=f"Diligence will stall because {sig.join_readable(problems)}.",
    )



def _note_maturity(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Blocker | None:
    months = sig.months_to_maturity(p)
    if months is None or months >= 6:
        return None
    if months < 0:
        detail = (
            f"The convertible instrument matured {abs(months):.0f} months ago, which must be "
            "resolved with existing holders before new capital can be placed."
        )
        severity = "CRITICAL"
    else:
        detail = (
            f"The convertible instrument matures in {months:.0f} months, ahead of a realistic "
            "close, creating pressure to renegotiate with existing holders."
        )
        severity = "HIGH"
    return Blocker(key="note_maturity", severity=severity, title="Instrument maturity pressure", detail=detail)


RULES: tuple[BlockerRule, ...] = (
    _valuation_too_high,
    _round_oversized,
    _no_lead,
    _insufficient_runway,
    _no_commercial_proof,
    _note_maturity,
    _regulatory_gap,
    _customer_concentration,
    _missing_management,
    _weak_financial_visibility,
    _materials_quality,
)


def identify_blockers(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Blocker]:
    """Return the most severe blockers, worst first."""
    found: list[Blocker] = []
    for rule in RULES:
        try:
            blocker = rule(profile, sector, stage, cfg)
        except Exception:  # noqa: BLE001 - one rule must not sink the analysis
            LOGGER.exception("Blocker rule %s failed.", getattr(rule, "__name__", rule))
            continue
        if blocker:
            found.append(blocker)

    found.sort(key=lambda b: b.rank)
    LOGGER.info("Identified %d blockers (showing up to %d).", len(found), MAX_BLOCKERS)
    return found[:MAX_BLOCKERS]
