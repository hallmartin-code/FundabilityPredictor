"""Exactly five prioritised actions to raise the probability of closing.

Every action is generated from something the model actually found in this
deck and is phrased with the company's own numbers. Generic advice
("improve traction") is worthless to a founder, so the generators either
produce a concrete instruction or produce nothing.
"""

from __future__ import annotations

import logging
from typing import Callable

from analysis import signals as sig
from config.settings import AppConfig
from extractors.normalization import format_money
from models.company import SectorProfile, Stage, at_least
from models.prediction import Action
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)

REQUIRED_ACTIONS = 5

Generator = Callable[[DeckProfile, SectorProfile, Stage, AppConfig], "Action | None"]


def _raise_label(profile: DeckProfile) -> str:
    """The raise amount, or a neutral stand-in when it was not stated."""
    target = profile.number("target_raise")
    return format_money(target) if target else "the current round"


def _round_phrase(profile: DeckProfile) -> str:
    """A complete noun phrase, so templates never produce "the the ... round"."""
    target = profile.number("target_raise")
    return f"the {format_money(target)} round" if target else "the current round"


def _cohort(sector: SectorProfile, stage: Stage) -> str:
    """How to describe this company's peer group to an investor.

    Degrades cleanly: "Seed MedTech", then "MedTech", then "Seed", and finally
    a neutral phrase rather than printing "Unknown Not Provided".
    """
    parts = []
    if stage.is_known:
        parts.append(stage.value)
    if sector.key != "unknown":
        parts.append(sector.label)
    return " ".join(parts) if parts else "comparable early-stage"


def _sector_phrase(sector: SectorProfile, fallback: str = "target-market") -> str:
    return sector.label if sector.key != "unknown" else fallback


def _secure_lead(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    if sig.has_lead_investor(p) is True:
        return None
    return Action(
        action=(
            f"Name and close a lead investor for {_round_phrase(p)}, targeting funds that "
            f"already invest in {_cohort(s, st)} companies, and publish the lead's cheque size "
            "and terms in the deck."
        ),
        why=(
            "A committed lead sets terms and gives followers a reason to move; rounds without one "
            "stall in diligence regardless of company quality."
        ),
        impact="HIGH",
        timeframe="Immediate",
    )


def _build_commitments(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    pct = sig.committed_pct(p)
    target = p.number("target_raise")
    if pct is not None and pct >= 50:
        return None
    if target is None:
        return None
    if pct is None:
        detail = (
            f"State how much of the {format_money(target)} round is already committed, in dollars "
            "and as a percentage, and name the investors behind it."
        )
        why = (
            "Round progress is the strongest near-term closing signal, and its absence is read as "
            "an absence of commitments."
        )
    else:
        gap = target * (0.5 - pct / 100.0)
        detail = (
            f"Convert soft interest into {format_money(gap)} of additional signed commitments to "
            f"move the round from {pct:.0f}% to 50% subscribed."
        )
        why = (
            "Crossing 50% subscribed changes the conversation from raising a round to filling one, "
            "and materially compresses time to close."
        )
    return Action(action=detail, why=why, impact="HIGH", timeframe="30 Days")


def _commercial_proof(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    revenue = sig.revenue(p)
    if revenue and revenue > 0:
        return None
    pilots = p.number("pilots")
    lois = p.number("lois")
    customers = sig.customers(p)

    if pilots:
        action = (
            f"Convert {min(int(pilots), 3)} of the {pilots:.0f} current pilots into paid contracts "
            "and put the resulting contract value and start dates in the financing materials."
        )
    elif lois:
        action = (
            f"Convert the {lois:.0f} letters of intent into signed purchase orders with stated "
            "volumes and pricing, and show the revenue they represent."
        )
    elif customers:
        action = (
            f"Move the {customers:.0f} named customers onto paid agreements and publish the "
            "resulting annual contract value."
        )
    else:
        action = (
            f"Sign three named {_sector_phrase(s)} design partners to paid evaluation agreements and "
            "disclose the contract value, so demand is evidenced rather than asserted."
        )
    return Action(
        action=action,
        why=(
            "Documented paid commitments are the evidence investors use to underwrite demand; "
            "without them the round is priced on belief alone."
        ),
        impact="HIGH",
        timeframe="60 Days",
    )


def _valuation_reset(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    valuation = (
        p.number("pre_money_valuation") or p.number("note_cap") or p.number("safe_cap")
    )
    multiple = sig.benchmark_multiple(valuation, cfg, st, "pre_money_typical", s)
    if multiple is None or multiple < 1.5:
        return None
    low, high = sig.benchmark_range(cfg, st, "pre_money_typical", s)
    return Action(
        action=(
            f"Reset the {format_money(valuation)} valuation into the "
            f"{format_money(low)}-{format_money(high)} range typical for {st.value} {s.label} "
            "rounds, or publish the traction that justifies the premium."
        ),
        why=(
            "A valuation above the range investors underwrite at this stage removes the company "
            "from consideration before diligence begins."
        ),
        impact="HIGH",
        timeframe="Immediate",
    )


def _resize_round(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    target = p.number("target_raise")
    multiple = sig.benchmark_multiple(target, cfg, st, "raise_typical", s)
    if multiple is None or multiple < 1.5:
        return None
    low, high = sig.benchmark_range(cfg, st, "raise_typical", s)
    return Action(
        action=(
            f"Restructure the {format_money(target)} ask into a first tranche of "
            f"{format_money(high)} tied to the next milestone, with the balance raised on the "
            "resulting proof points."
        ),
        why=(
            f"An ask {multiple:.1f}x the {st.value} norm limits the round to a small number of "
            "funds; a right-sized first tranche widens the pool of investors who can lead."
        ),
        impact="HIGH",
        timeframe="30 Days",
    )


def _extend_runway(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    runway = sig.runway_months(p)
    if runway is None or runway >= 9:
        return None
    burn = p.number("burn_rate")
    lever = (
        f"reducing the {format_money(burn)} monthly burn or"
        if burn
        else "reducing monthly burn or"
    )
    return Action(
        action=(
            f"Extend runway beyond 9 months from the current {runway:.0f} by {lever} closing a "
            "bridge from existing holders, before opening the priced round."
        ),
        why=(
            "Investors price urgency. Negotiating with under nine months of cash transfers "
            "leverage to the buyer and depresses terms."
        ),
        impact="HIGH",
        timeframe="Immediate",
    )


def _use_of_funds(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    if p.has("use_of_funds"):
        return None
    return Action(
        action=(
            f"Add a line-item use of proceeds for {_round_phrase(p)}, allocating each dollar to "
            "a named milestone with its completion date and the runway it buys."
        ),
        why=(
            "Investors underwrite the milestone the money buys, not the amount requested; an "
            "unallocated ask reads as an unplanned one."
        ),
        impact="MEDIUM",
        timeframe="Immediate",
    )


def _financial_model(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    if sig.has_financial_model(p):
        return None
    return Action(
        action=(
            "Publish a three-year operating model showing monthly burn, headcount, revenue build "
            "and the cash position at each milestone."
        ),
        why=(
            "Without a model investors cannot test whether the raise funds the plan, which pushes "
            "the decision past the point most rounds close."
        ),
        impact="MEDIUM",
        timeframe="30 Days",
    )


def _financial_disclosure(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    missing = [
        label
        for field, label in (
            ("burn_rate", "monthly burn"),
            ("runway_months", "runway"),
            ("cash_balance", "cash on hand"),
            ("gross_margin", "gross margin"),
        )
        if not p.has(field)
    ]
    if len(missing) < 2:
        return None
    return Action(
        action=(
            f"Add a financial summary slide disclosing {sig.join_readable(missing)}, so the ask can be "
            "sized against the company's actual position."
        ),
        why=(
            "These four figures are the first questions in every diligence call; supplying them "
            "up front removes a round trip from the process."
        ),
        impact="MEDIUM",
        timeframe="Immediate",
    )


def _regulatory_milestone(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    if not sig.has_long_regulatory_path(p, s):
        return None
    return Action(
        action=(
            f"Publish the regulatory plan to the next {s.label} gate - submission type, filing "
            "date, cost and the evidence already assembled - and name the regulatory consultant "
            "or CRO engaged to deliver it."
        ),
        why=(
            f"{s.label} financing is gated on regulatory progress; a dated, costed pathway is what "
            "converts a technical story into an investable milestone."
        ),
        impact="HIGH",
        timeframe="60 Days",
    )


def _unit_economics(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    if p.has("gross_margin"):
        return None
    if not (p.has("business_model") or p.number("average_contract_value")):
        return None
    return Action(
        action=(
            "Publish unit economics at volume: price, cost of goods, gross margin per unit and how "
            "margin improves with scale."
        ),
        why=(
            "Margin structure determines how much capital the business needs after this round, "
            "which is what a lead is really underwriting."
        ),
        impact="MEDIUM",
        timeframe="30 Days",
    )


def _commercial_hire(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    if p.has("commercial_expertise") or not at_least(st, Stage.SEED):
        return None
    return Action(
        action=(
            "Appoint a full-time commercial lead with sector distribution experience and name them "
            "in the deck with their prior results."
        ),
        why=(
            "A technical founding team without a commercial owner reads as execution risk on the "
            "revenue plan the round is meant to fund."
        ),
        impact="MEDIUM",
        timeframe="90 Days",
    )


def _machine_readable_deck(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    quality = p.extraction_quality
    if not quality.is_image_heavy:
        return None
    return Action(
        action=(
            f"Rebuild the {len(quality.image_based_pages)} image-only slides with selectable text, "
            "and provide the key financial and traction figures in a data room summary."
        ),
        why=(
            "Image-only slides are invisible to the screening and diligence tools most funds run "
            "first, so the company's own numbers never reach the analyst."
        ),
        impact="MEDIUM",
        timeframe="Immediate",
    )


def _ip_position(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    if sig.has_ip(p):
        return None
    return Action(
        action=(
            "File provisional patent applications on the core mechanism and summarise the claims, "
            "jurisdictions and expiry dates in the deck."
        ),
        why=(
            "Without a documented IP position the defensibility question stays open, and it is the "
            "question that most often ends technical diligence."
        ),
        impact="MEDIUM",
        timeframe="90 Days",
    )


def _investor_targeting(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    """Always available: a concrete outreach plan built from stage and sector."""
    return Action(
        action=(
            f"Build a named target list of 25 funds that led {_cohort(s, st)} rounds in the "
            "last 18 months, and run a time-boxed process with all first meetings inside four weeks."
        ),
        why=(
            "A concentrated, time-boxed process creates the competitive tension that produces a "
            "lead; sequential outreach lets the round drift."
        ),
        impact="MEDIUM",
        timeframe="30 Days",
    )


def _milestone_narrative(p: DeckProfile, s: SectorProfile, st: Stage, cfg: AppConfig) -> Action | None:
    if sig.has_milestones(p):
        return None
    return Action(
        action=(
            f"Define the three milestones {_round_phrase(p)} must deliver, each with a date, an "
            "owner and the metric that proves completion."
        ),
        why=(
            "The next round is priced on these milestones, so investors need to see that this "
            "round reaches them before they commit."
        ),
        impact="MEDIUM",
        timeframe="Immediate",
    )


#: Order matters only as a tie-break; ranking is by impact then timeframe.
GENERATORS: tuple[Generator, ...] = (
    _secure_lead,
    _valuation_reset,
    _extend_runway,
    _resize_round,
    _build_commitments,
    _commercial_proof,
    _regulatory_milestone,
    _use_of_funds,
    _financial_disclosure,
    _financial_model,
    _unit_economics,
    _machine_readable_deck,
    _milestone_narrative,
    _commercial_hire,
    _ip_position,
    _investor_targeting,
)


def recommend_actions(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[Action]:
    """Return exactly five actions, highest impact first."""
    candidates: list[Action] = []
    for generator in GENERATORS:
        try:
            action = generator(profile, sector, stage, cfg)
        except Exception:  # noqa: BLE001 - one generator must not sink the analysis
            LOGGER.exception("Recommendation generator %s failed.", getattr(generator, "__name__", generator))
            continue
        if action:
            candidates.append(action)

    # Stable sort, so generator order breaks ties at equal impact/timeframe.
    ordered = sorted(candidates, key=lambda a: a.rank)

    if len(ordered) < REQUIRED_ACTIONS:
        LOGGER.info(
            "%d situational actions found; topping up to %d from the strengthening set.",
            len(ordered),
            REQUIRED_ACTIONS,
        )
        existing = {a.action for a in ordered}
        for fallback in _fallback_actions(profile, sector, stage):
            if len(ordered) >= REQUIRED_ACTIONS:
                break
            if fallback.action not in existing:
                ordered.append(fallback)
                existing.add(fallback.action)

    return ordered[:REQUIRED_ACTIONS]


def _fallback_actions(profile: DeckProfile, sector: SectorProfile, stage: Stage) -> list[Action]:
    """Used only to reach five when a deck has few outright gaps.

    These still reference the company's stage, sector and ask - they are
    strengthening moves for an already-solid raise, not filler.
    """
    round_phrase = _round_phrase(profile)
    return [
        Action(
            action=(
                f"Publish a diligence pack alongside {round_phrase} - cap table, financial "
                "model, customer references and IP schedule - so a lead can complete confirmatory "
                "diligence without a document request cycle."
            ),
            why=(
                "Diligence latency, not investor interest, is what usually separates a six-week "
                "close from a six-month one."
            ),
            impact="MEDIUM",
            timeframe="30 Days",
        ),
        Action(
            action=(
                f"Set a dated first close for {round_phrase} and offer participating "
                "investors defined terms for committing before it."
            ),
            why=(
                "A dated close converts open-ended interest into decisions and gives followers a "
                "reason to move now rather than wait."
            ),
            impact="MEDIUM",
            timeframe="30 Days",
        ),
        Action(
            action=(
                f"Agree the syndicate structure for {round_phrase} - allocation per "
                "investor, minimum cheque and the order in which commitments convert - before "
                "opening the remaining allocation."
            ),
            why=(
                "An undefined allocation invites investors to wait and see how the round fills, "
                "which is what turns a filling round into a stalled one."
            ),
            impact="MEDIUM",
            timeframe="60 Days",
        ),
        Action(
            action=(
                f"Secure two written references from {_sector_phrase(sector)} customers or partners that a "
                "prospective lead can contact directly."
            ),
            why=(
                "Third-party confirmation of demand shortens the reference-checking stage, which "
                "is where late-stage diligence most often slows."
            ),
            impact="LOW",
            timeframe="60 Days",
        ),
        Action(
            action=(
                f"Prepare the governance package a {_cohort(sector, stage)} lead will require: cap table, "
                "board composition, information rights and the reporting cadence after close."
            ),
            why=(
                "Governance terms are negotiated last and delay signing most often; settling them "
                "in advance removes the final source of slippage."
            ),
            impact="LOW",
            timeframe="90 Days",
        ),
    ]
