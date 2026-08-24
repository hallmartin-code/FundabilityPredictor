"""Financing probability: readiness score -> calibrated closing probabilities.

The pipeline is deliberately explicit so that every percentage point can be
traced:

    baseline band -> signal adjustments -> sector modifier
                  -> market environment -> clamp -> monotonicity

Detection lives in code (``SIGNAL_DETECTORS``) and magnitude lives in
``config/probability_rules.yaml``. A config edit can retune or disable a rule
but can never invent a signal the data does not support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from analysis import signals as sig
from config.settings import AppConfig, load_config
from models.company import SectorProfile, Stage, at_least, stage_from_text
from models.prediction import HORIZONS, BridgeStep, MarketCondition, PredictionResult, Probability
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalContext:
    """Everything a detector may consult."""

    profile: DeckProfile
    sector: SectorProfile
    stage: Stage
    cfg: AppConfig
    rule: dict


Detector = Callable[[SignalContext], bool]

#: The largest share of the remaining headroom a single adjustment may take.
MAX_ROOM_CONSUMED = 0.9


# ---------------------------------------------------------------------------
# Positive detectors
# ---------------------------------------------------------------------------
def _lead_investor_committed(ctx: SignalContext) -> bool:
    return sig.has_lead_investor(ctx.profile) is True


def _committed_at_least(ctx: SignalContext) -> bool:
    pct = sig.committed_pct(ctx.profile)
    return pct is not None and pct >= float(ctx.rule.get("threshold", 25))


def _rapid_revenue_growth(ctx: SignalContext) -> bool:
    growth = ctx.profile.number("revenue_growth")
    return growth is not None and growth >= float(ctx.rule.get("threshold_pct", 80))


def _strong_recurring_revenue(ctx: SignalContext) -> bool:
    arr = ctx.profile.number("arr")
    return arr is not None and arr >= float(ctx.rule.get("threshold_arr", 1_000_000))


def _multiple_paying_customers(ctx: SignalContext) -> bool:
    paying = ctx.profile.number("paying_customers") or sig.customers(ctx.profile)
    return paying is not None and paying >= float(ctx.rule.get("threshold", 10))


def _high_gross_margin(ctx: SignalContext) -> bool:
    margin = ctx.profile.number("gross_margin")
    return margin is not None and margin >= float(ctx.rule.get("threshold_pct", 70))


def _strong_retention(ctx: SignalContext) -> bool:
    renewal = ctx.profile.number("renewal_rate")
    return renewal is not None and renewal >= float(ctx.rule.get("threshold_pct", 90))


def _strong_runway(ctx: SignalContext) -> bool:
    runway = sig.runway_months(ctx.profile)
    return runway is not None and runway >= float(ctx.rule.get("threshold_months", 12))


def _round_size_appropriate(ctx: SignalContext) -> bool:
    return sig.within_benchmark(
        ctx.profile.number("target_raise"), ctx.cfg, ctx.stage, "raise_typical", ctx.sector
    ) is True


# ---------------------------------------------------------------------------
# Negative detectors
# ---------------------------------------------------------------------------
def _no_revenue_late_stage(ctx: SignalContext) -> bool:
    """Only a negative once the sector's own norms expect revenue."""
    revenue = sig.revenue(ctx.profile)
    if revenue is not None and revenue > 0:
        return False
    expected_from = stage_from_text(ctx.sector.revenue_expected_from)
    return at_least(ctx.stage, expected_from)


def _no_customers(ctx: SignalContext) -> bool:
    """No visible customer, pilot, LOI or contract evidence at Seed or later."""
    if not at_least(ctx.stage, Stage.SEED):
        return False
    engagements = sig.commercial_engagements(ctx.profile)
    if engagements is not None and engagements > 0:
        return False
    revenue = sig.revenue(ctx.profile)
    return not (revenue and revenue > 0)


def _short_runway(ctx: SignalContext) -> bool:
    runway = sig.runway_months(ctx.profile)
    return runway is not None and runway < float(ctx.rule.get("threshold_months", 6))


def _excessive_valuation(ctx: SignalContext) -> bool:
    valuation = (
        ctx.profile.number("pre_money_valuation")
        or ctx.profile.number("note_cap")
        or ctx.profile.number("safe_cap")
    )
    multiple = sig.benchmark_multiple(valuation, ctx.cfg, ctx.stage, "pre_money_typical", ctx.sector)
    return multiple is not None and multiple >= float(ctx.rule.get("threshold_multiple", 2.0))


def _round_too_large(ctx: SignalContext) -> bool:
    multiple = sig.benchmark_multiple(
        ctx.profile.number("target_raise"), ctx.cfg, ctx.stage, "raise_typical", ctx.sector
    )
    return multiple is not None and multiple >= float(ctx.rule.get("threshold_multiple", 2.0))


def _weak_gross_margin(ctx: SignalContext) -> bool:
    margin = ctx.profile.number("gross_margin")
    return margin is not None and margin < float(ctx.rule.get("threshold_pct", 30))


def _customer_concentration(ctx: SignalContext) -> bool:
    concentration = ctx.profile.number("customer_concentration")
    return concentration is not None and concentration >= float(ctx.rule.get("threshold_pct", 50))


def _no_lead_investor(ctx: SignalContext) -> bool:
    """Fires when no lead is committed, including when none is mentioned."""
    return sig.has_lead_investor(ctx.profile) is not True


def _limited_investor_traction(ctx: SignalContext) -> bool:
    pct = sig.committed_pct(ctx.profile)
    return pct is not None and pct < float(ctx.rule.get("threshold", 10))


def _note_maturity_pressure(ctx: SignalContext) -> bool:
    months = sig.months_to_maturity(ctx.profile)
    return months is not None and months < float(ctx.rule.get("threshold_months", 6))


SIGNAL_DETECTORS: dict[str, Detector] = {
    # positive
    "lead_investor_committed": _lead_investor_committed,
    "round_25_committed": _committed_at_least,
    "round_50_committed": _committed_at_least,
    "round_75_committed": _committed_at_least,
    "rapid_revenue_growth": _rapid_revenue_growth,
    "strong_recurring_revenue": _strong_recurring_revenue,
    "multiple_paying_customers": _multiple_paying_customers,
    "high_gross_margin": _high_gross_margin,
    "strategic_investor": lambda c: sig.has_strategic_investor(c.profile),
    "repeat_founder": lambda c: sig.is_repeat_founder(c.profile),
    "prior_exit": lambda c: sig.has_founder_prior_exit(c.profile),
    "strong_retention": _strong_retention,
    "regulatory_milestone": lambda c: sig.has_regulatory_milestone(c.profile),
    "major_partnership": lambda c: sig.has_partnership(c.profile),
    "strong_runway": _strong_runway,
    "round_size_appropriate": _round_size_appropriate,
    "non_dilutive_funding": lambda c: sig.has_non_dilutive_funding(c.profile),
    # negative
    "no_revenue_late_stage": _no_revenue_late_stage,
    "no_customers": _no_customers,
    "short_runway": _short_runway,
    "excessive_valuation": _excessive_valuation,
    "round_too_large": _round_too_large,
    "weak_gross_margin": _weak_gross_margin,
    "customer_concentration": _customer_concentration,
    "unclear_use_of_funds": lambda c: not c.profile.has("use_of_funds"),
    "no_lead_investor": _no_lead_investor,
    "limited_investor_traction": _limited_investor_traction,
    "capital_intensive": lambda c: c.sector.capital_intensity == "high",
    "long_regulatory_timeline": lambda c: sig.has_long_regulatory_path(c.profile, c.sector),
    "unclear_business_model": lambda c: not c.profile.has("business_model"),
    "weak_differentiation": lambda c: not sig.has_differentiation(c.profile),
    "missing_financial_model": lambda c: not sig.has_financial_model(c.profile),
    "founder_not_full_time": lambda c: sig.founder_is_full_time(c.profile) is False,
    "legal_regulatory_dependency": lambda c: sig.has_legal_dependency(c.profile),
    "note_maturity_pressure": _note_maturity_pressure,
}

#: Rules that would otherwise double-count the same underlying fact. Only the
#: first rule in each group that fires is applied.
MUTUALLY_EXCLUSIVE: tuple[tuple[str, ...], ...] = (
    ("round_75_committed", "round_50_committed", "round_25_committed"),
    ("lead_investor_committed", "no_lead_investor"),
)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def baseline_probability(score: float, cfg: AppConfig) -> Probability:
    """Interpolate the readiness score inside its band."""
    for band in cfg.baseline_bands:
        low, high = float(band["min"]), float(band["max"])
        if low <= score <= high:
            span = high - low
            position = (score - low) / span if span else 0.0
            values = {}
            for horizon in HORIZONS:
                p_low, p_high = band[horizon]
                values[horizon] = float(p_low) + position * (float(p_high) - float(p_low))
            return Probability(**values)
    # Below every band (should not happen given a 0-floor band) - use the last.
    band = cfg.baseline_bands[-1]
    return Probability(**{h: float(band[h][0]) for h in HORIZONS})


def _damped(delta: float, running: float, half_saturation: float) -> float:
    """Scale a percentage-point delta by the room left in its direction.

    Keeps a long list of same-signed adjustments from pushing a probability
    through the floor or ceiling, without changing behaviour in the middle of
    the range where the configured magnitudes were calibrated.
    """
    room = max((100.0 - running) if delta >= 0 else running, 0.0)
    damped = delta * room / (room + half_saturation)
    # Never consume the whole remaining room: a delta that exactly reaches 0 or
    # 100 would report a financing outcome as impossible or certain.
    limit = room * MAX_ROOM_CONSUMED
    return max(damped, -limit) if delta < 0 else min(damped, limit)


def resolve_market_condition(cfg: AppConfig, requested: str | None = None) -> MarketCondition:
    """Resolve the capital market environment.

    There is no live market feed, so the result is always reported as an
    assumption unless a future data source sets ``is_verified``.
    """
    environments = cfg.market_conditions["environments"]
    key = (requested or cfg.market_conditions["default"]).strip().lower().replace(" ", "_")
    if key not in environments:
        LOGGER.warning("Unknown market condition %r; falling back to the default.", requested)
        key = cfg.market_conditions["default"]
    env = environments[key]
    return MarketCondition(
        key=key,
        label=str(env["label"]),
        multiplier={h: float(env["multiplier"][h]) for h in HORIZONS},
        is_verified=False,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def predict(
    profile: DeckProfile,
    sector: SectorProfile,
    stage: Stage,
    readiness_total: float,
    config: AppConfig | None = None,
    market_condition: str | None = None,
) -> PredictionResult:
    """Produce the three horizon probabilities and the bridge explaining them."""
    cfg = config or load_config()
    baseline = baseline_probability(readiness_total, cfg)
    running = baseline.as_dict()

    bridge: list[BridgeStep] = [
        BridgeStep(label="Baseline from readiness score", kind="baseline", delta={}, running=dict(running))
    ]
    positives: list[str] = []
    negatives: list[str] = []

    fired = _detect(profile, sector, stage, cfg)

    damping = cfg.probability.get("adjustment_damping", {})
    damping_enabled = bool(damping.get("enabled", True))
    half_saturation = float(damping.get("half_saturation", 8.0))

    for rule_id, rule, is_positive in fired:
        delta = {}
        for horizon in HORIZONS:
            raw = float(rule["delta"][horizon])
            applied = (
                _damped(raw, running[horizon], half_saturation) if damping_enabled else raw
            )
            delta[horizon] = applied
            running[horizon] += applied
        label = str(rule["label"])
        bridge.append(BridgeStep(label=label, kind="signal", delta=delta, running=dict(running)))
        (positives if is_positive else negatives).append(label)

    # Sector modifier.
    sector_modifier = {h: float(sector.probability_modifier.get(h, 1.0)) for h in HORIZONS}
    if any(abs(v - 1.0) > 1e-9 for v in sector_modifier.values()):
        delta = {}
        for horizon in HORIZONS:
            before = running[horizon]
            running[horizon] = before * sector_modifier[horizon]
            delta[horizon] = running[horizon] - before
        bridge.append(
            BridgeStep(
                label=f"{sector.label} sector financing dynamics",
                kind="sector",
                delta=delta,
                running=dict(running),
            )
        )

    # Capital market environment.
    market = resolve_market_condition(cfg, market_condition)
    if any(abs(v - 1.0) > 1e-9 for v in market.multiplier.values()):
        delta = {}
        for horizon in HORIZONS:
            before = running[horizon]
            running[horizon] = before * market.multiplier[horizon]
            delta[horizon] = running[horizon] - before
        bridge.append(
            BridgeStep(
                label=f"Capital market environment: {market.label}",
                kind="market",
                delta=delta,
                running=dict(running),
            )
        )

    final = _finalise(running, cfg, bridge)

    LOGGER.info(
        "Probabilities: 3m %.0f%%, 6m %.0f%%, 12m %.0f%% (%d positive, %d negative signals)",
        final.p3, final.p6, final.p12, len(positives), len(negatives),
    )
    return PredictionResult(
        baseline=baseline,
        final=final,
        bridge=bridge,
        positive_signals=positives,
        negative_signals=negatives,
        market_condition=market,
        sector_modifier=sector_modifier,
    )


def _detect(
    profile: DeckProfile, sector: SectorProfile, stage: Stage, cfg: AppConfig
) -> list[tuple[str, dict, bool]]:
    """Evaluate every enabled rule, resolving mutually exclusive groups."""
    fired: list[tuple[str, dict, bool]] = []
    for rules, is_positive in ((cfg.positive_signals, True), (cfg.negative_signals, False)):
        for rule_id, rule in rules.items():
            if not rule.get("enabled", True):
                continue
            detector = SIGNAL_DETECTORS.get(rule_id)
            if detector is None:
                LOGGER.warning("No detector registered for signal %r; skipping.", rule_id)
                continue
            ctx = SignalContext(profile=profile, sector=sector, stage=stage, cfg=cfg, rule=rule)
            try:
                if detector(ctx):
                    fired.append((rule_id, rule, is_positive))
            except Exception:  # noqa: BLE001 - one bad rule must not sink the run
                LOGGER.exception("Signal detector %r failed; treating as not fired.", rule_id)
    return _resolve_exclusions(fired)


def _resolve_exclusions(fired: list[tuple[str, dict, bool]]) -> list[tuple[str, dict, bool]]:
    suppressed: set[str] = set()
    fired_ids = {rule_id for rule_id, _, _ in fired}
    for group in MUTUALLY_EXCLUSIVE:
        seen = False
        for rule_id in group:
            if rule_id not in fired_ids:
                continue
            if seen:
                suppressed.add(rule_id)
            seen = True
    if suppressed:
        LOGGER.debug("Suppressed overlapping signals: %s", sorted(suppressed))
    return [item for item in fired if item[0] not in suppressed]


def _finalise(running: dict[str, float], cfg: AppConfig, bridge: list[BridgeStep]) -> Probability:
    """Clamp into range, then enforce 3m <= 6m <= 12m."""
    low, high = cfg.clamp
    clamped = {h: max(low, min(high, running[h])) for h in HORIZONS}
    if clamped != running:
        bridge.append(
            BridgeStep(
                label=f"Clamped to the {low:.0f}%-{high:.0f}% reporting range",
                kind="clamp",
                delta={h: clamped[h] - running[h] for h in HORIZONS},
                running=dict(clamped),
            )
        )

    mono = cfg.monotonicity
    if mono.get("enforce", True):
        gap = float(mono.get("min_gap", 0.0))
        adjusted = dict(clamped)
        # A longer horizon can never be less likely than a shorter one.
        adjusted["p6"] = max(adjusted["p6"], adjusted["p3"] + gap)
        adjusted["p12"] = max(adjusted["p12"], adjusted["p6"] + gap)
        adjusted = {h: min(high, adjusted[h]) for h in HORIZONS}
        # Re-tightening from the top keeps the ordering intact after the cap.
        adjusted["p6"] = min(adjusted["p6"], adjusted["p12"])
        adjusted["p3"] = min(adjusted["p3"], adjusted["p6"])
        if any(abs(adjusted[h] - clamped[h]) > 1e-9 for h in HORIZONS):
            bridge.append(
                BridgeStep(
                    label="Adjusted so longer horizons are never less likely",
                    kind="monotonicity",
                    delta={h: adjusted[h] - clamped[h] for h in HORIZONS},
                    running=dict(adjusted),
                )
            )
        clamped = adjusted

    return Probability(**{h: round(clamped[h], 1) for h in HORIZONS})
