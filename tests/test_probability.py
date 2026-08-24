"""The probability model, its adjustments, blockers and recommendations."""

from __future__ import annotations

import pytest

from analysis.blockers import identify_blockers
from analysis.probability_model import (
    SIGNAL_DETECTORS,
    _damped,
    baseline_probability,
    predict,
    resolve_market_condition,
)
from analysis.recommendations import REQUIRED_ACTIONS, recommend_actions
from analysis.sector_classifier import classify_sector
from analysis.stage_inference import infer_stage
from models.company import SectorProfile, Stage
from models.evidence import ValueStatus
from models.prediction import HORIZONS


def _predict(profile, config, readiness=60.0, market=None):
    sector = classify_sector(profile, config)
    stage = infer_stage(profile, config).effective
    return predict(profile, sector, stage, readiness, config, market_condition=market)


# ---------------------------------------------------------------------------
# Baselines and bands
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("score", [0, 15, 39, 40, 54, 55, 69, 70, 84, 85, 100])
def test_baseline_is_defined_and_ordered_for_every_score(score, config):
    baseline = baseline_probability(score, config)
    assert 0 < baseline.p3 <= baseline.p6 <= baseline.p12 <= 100


def test_baseline_rises_with_readiness(config):
    low = baseline_probability(30, config)
    high = baseline_probability(90, config)
    for horizon in HORIZONS:
        assert high.get(horizon) > low.get(horizon)


def test_every_configured_signal_has_a_detector(config):
    configured = set(config.positive_signals) | set(config.negative_signals)
    missing = configured - set(SIGNAL_DETECTORS)
    assert not missing, f"signals without detectors: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Required output properties
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["saas_result", "biotech_result", "sparse_result"])
def test_probabilities_are_clamped_and_monotonic(fixture, request, config):
    result = request.getfixturevalue(fixture)
    final = result.prediction.final
    low, high = config.clamp
    for horizon in HORIZONS:
        assert low <= final.get(horizon) <= high
    assert final.p3 <= final.p6 <= final.p12


def test_probabilities_never_reach_certainty(saas_result):
    """The strongest sample deck must still not be reported as a certainty."""
    assert saas_result.prediction.final.p12 < 100
    assert saas_result.prediction.final.p3 > 0


def test_stronger_company_gets_higher_probability(saas_result, sparse_result):
    assert saas_result.prediction.final.p6 > sparse_result.prediction.final.p6


# ---------------------------------------------------------------------------
# Signal direction
# ---------------------------------------------------------------------------
def test_positive_signals_raise_and_negative_signals_lower(saas_profile, config):
    result = _predict(saas_profile, config, readiness=60.0)
    positive_total = sum(
        step.delta["p6"] for step in result.bridge if step.kind == "signal" and step.delta["p6"] > 0
    )
    negative_total = sum(
        step.delta["p6"] for step in result.bridge if step.kind == "signal" and step.delta["p6"] < 0
    )
    assert positive_total > 0
    assert result.positive_signals
    # The SaaS deck is strong, so positives must dominate.
    assert positive_total + negative_total > 0


def test_removing_the_lead_investor_lowers_the_probability(saas_profile, config):
    before = _predict(saas_profile, config).final.p6

    without_lead = saas_profile.model_copy(deep=True)
    without_lead.values.pop("lead_investor", None)
    after = _predict(without_lead, config).final.p6

    assert after < before
    assert "No lead investor identified" in _predict(without_lead, config).negative_signals


def test_short_runway_lowers_the_probability(saas_profile, config):
    before = _predict(saas_profile, config).final.p6

    tight = saas_profile.model_copy(deep=True)
    tight.put("runway_months", 3.0, ValueStatus.EXPLICIT, 0.95)
    after = _predict(tight, config)

    assert after.final.p6 < before
    assert "Under 6 months runway" in after.negative_signals


def test_committed_thresholds_do_not_double_count(saas_profile, config):
    """25/50/75% committed describe one fact and must not stack."""
    result = _predict(saas_profile, config)
    fired = [label for label in result.positive_signals if "% of round committed" in label]
    assert len(fired) <= 1


def test_lead_investor_signals_are_mutually_exclusive(saas_profile, config):
    result = _predict(saas_profile, config)
    assert not (
        "Lead investor committed" in result.positive_signals
        and "No lead investor identified" in result.negative_signals
    )


def test_seeking_a_lead_is_not_a_committed_lead(saas_profile, config):
    seeking = saas_profile.model_copy(deep=True)
    seeking.put("lead_investor", "Seeking a lead investor", ValueStatus.EXPLICIT, 0.9)
    result = _predict(seeking, config)
    assert "Lead investor committed" not in result.positive_signals


# ---------------------------------------------------------------------------
# Modifiers
# ---------------------------------------------------------------------------
def test_market_condition_defaults_to_an_unverified_neutral(config):
    market = resolve_market_condition(config)
    assert market.key == "neutral"
    assert market.is_verified is False
    assert market.disclosure.startswith("MARKET CONDITION ASSUMPTION")


def test_unknown_market_condition_falls_back_to_the_default(config):
    assert resolve_market_condition(config, "wildly_optimistic").key == "neutral"


def test_difficult_market_lowers_probability(saas_profile, config):
    neutral = _predict(saas_profile, config, market="neutral").final.p6
    difficult = _predict(saas_profile, config, market="highly_difficult").final.p6
    assert difficult < neutral


def test_sector_modifier_is_applied_and_recorded(biotech_result):
    assert biotech_result.prediction.sector_modifier["p6"] < 1.0
    assert any(step.kind == "sector" for step in biotech_result.prediction.bridge)


# ---------------------------------------------------------------------------
# Damping
# ---------------------------------------------------------------------------
def test_damping_shrinks_deltas_near_the_bounds():
    mid = _damped(-7.0, 50.0, 8.0)
    low = _damped(-7.0, 5.0, 8.0)
    assert abs(low) < abs(mid) < 7.0


def test_damping_never_overshoots_the_bounds():
    """Repeated same-signed adjustments approach the bounds without crossing.

    They can converge onto 0.0 or 100.0 once the remaining room falls below
    float precision; what must never happen is passing them. The reported
    range is guaranteed separately by the clamp.
    """
    running = 4.0
    for _ in range(40):
        running += _damped(-8.0, running, 8.0)
        assert running >= 0.0

    running = 96.0
    for _ in range(40):
        running += _damped(9.0, running, 8.0)
        assert running <= 100.0


def test_clamp_keeps_reported_probabilities_inside_the_range(saas_profile, config):
    """Whatever the adjustments do, the reported figure stays inside 1-98."""
    low, high = config.clamp
    for readiness in (0.0, 25.0, 50.0, 75.0, 100.0):
        final = _predict(saas_profile, config, readiness=readiness).final
        for horizon in HORIZONS:
            assert low <= final.get(horizon) <= high


# ---------------------------------------------------------------------------
# Bridge transparency
# ---------------------------------------------------------------------------
def test_bridge_arithmetic_matches_the_reported_result(saas_result):
    bridge = saas_result.prediction.bridge
    assert bridge[0].kind == "baseline"
    running = bridge[0].running["p6"]
    for step in bridge[1:]:
        if step.kind in ("signal", "sector", "market"):
            running += step.delta["p6"]
        else:  # clamp / monotonicity restate the value outright
            running = step.running["p6"]
        assert running == pytest.approx(step.running["p6"], abs=0.01)
    assert round(running, 1) == pytest.approx(saas_result.prediction.final.p6, abs=0.11)


def test_bridge_renders_readable_text(saas_result):
    text = saas_result.prediction.bridge_text("p6")
    assert "Baseline probability" in text
    assert "Final 6 months probability" in text


# ---------------------------------------------------------------------------
# Blockers
# ---------------------------------------------------------------------------
def test_blockers_are_ranked_by_severity(sparse_result):
    severities = [b.severity for b in sparse_result.blockers]
    ranks = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2}
    assert severities == sorted(severities, key=lambda s: ranks[s])


def test_blocker_count_is_capped(sparse_result, biotech_result):
    for result in (sparse_result, biotech_result):
        assert len(result.blockers) <= 5


def test_a_clean_deck_has_no_critical_blockers(saas_result):
    assert not [b for b in saas_result.blockers if b.severity == "CRITICAL"]


def test_oversized_round_is_flagged(saas_profile, config):
    inflated = saas_profile.model_copy(deep=True)
    inflated.put("target_raise", 90_000_000.0, ValueStatus.EXPLICIT, 0.95)
    sector = classify_sector(inflated, config)
    stage = Stage.SERIES_A
    blockers = identify_blockers(inflated, sector, stage, config)
    assert any(b.key == "round_too_large" for b in blockers)


def test_blocker_detail_quotes_the_company_numbers(sparse_result):
    for blocker in sparse_result.blockers:
        assert blocker.detail and len(blocker.detail) > 40


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["saas_result", "biotech_result", "sparse_result"])
def test_exactly_five_actions_are_produced(fixture, request):
    result = request.getfixturevalue(fixture)
    assert len(result.actions) == REQUIRED_ACTIONS


def test_actions_are_ranked_by_impact_then_timeframe(saas_result, sparse_result):
    for result in (saas_result, sparse_result):
        ranks = [action.rank for action in result.actions]
        assert ranks == sorted(ranks)


def test_actions_avoid_generic_advice(sparse_result, biotech_result):
    banned = ("improve traction", "work harder", "get more customers", "be better")
    for result in (sparse_result, biotech_result):
        for action in result.actions:
            assert action.action.lower() not in banned
            assert len(action.action) > 45, "action is too vague to be actionable"
            assert action.why
            assert action.impact in ("HIGH", "MEDIUM", "LOW")
            assert action.timeframe in ("Immediate", "30 Days", "60 Days", "90 Days")


@pytest.mark.parametrize("fixture", ["saas_result", "biotech_result", "sparse_result"])
def test_generated_prose_reads_cleanly_even_with_unknowns(fixture, request):
    """Missing sector or stage must not leak placeholders into the prose.

    A deck that discloses nothing still produces text a founder reads, so
    "Unknown Not Provided companies" or "the the current round round" are
    defects, not cosmetic issues.
    """
    result = request.getfixturevalue(fixture)
    prose = " ".join(
        [a.action for a in result.actions]
        + [a.why for a in result.actions]
        + [b.detail for b in result.blockers]
        + [b.title for b in result.blockers]
        + [result.comparable.rationale, result.ten_capital.explanation]
    )
    for artefact in ("the the", "Not Provided", "Unknown", "None", "  ", " ,", " .", ", and ,"):
        assert artefact not in prose, f"generated prose contains {artefact!r}"


def test_actions_reference_the_company_specifics(biotech_result):
    joined = " ".join(action.action for action in biotech_result.actions)
    assert "Biotech" in joined or "$3.5M" in joined


def test_actions_are_generated_for_a_deck_with_no_gaps(saas_profile, config):
    sector = classify_sector(saas_profile, config)
    stage = infer_stage(saas_profile, config).effective
    actions = recommend_actions(saas_profile, sector, stage, config)
    assert len(actions) == REQUIRED_ACTIONS
    assert len({a.action for a in actions}) == REQUIRED_ACTIONS


# ---------------------------------------------------------------------------
# Comparable profile and TEN Capital fit
# ---------------------------------------------------------------------------
def test_comparable_profile_is_classified_not_fabricated(saas_result, biotech_result):
    assert saas_result.comparable.label
    assert saas_result.comparable.rationale
    # A qualitative archetype must not claim a specific transaction.
    for result in (saas_result, biotech_result):
        assert "raised $" not in result.comparable.rationale


def test_milestone_sector_gets_a_life_science_profile(biotech_result):
    assert biotech_result.comparable.key in (
        "milestone_life_science", "competitive_seed", "highly_fundable_seed"
    )


def test_ten_capital_flags_are_tri_state(saas_result, sparse_result):
    for result in (saas_result, sparse_result):
        for flag in result.ten_capital.flags:
            assert flag.value in ("YES", "NO", "UNKNOWN", "REVIEW")


def test_unknown_flags_are_not_converted_to_no(sparse_result):
    """Silence in the deck must not be reported as a confirmed negative."""
    values = {flag.key: flag.value for flag in sparse_result.ten_capital.flags}
    assert values["raise_over_2m"] == "UNKNOWN"
    assert values["lead_investor"] == "UNKNOWN"


def test_revenue_flags_never_contradict_each_other(saas_result, biotech_result, sparse_result):
    for result in (saas_result, biotech_result, sparse_result):
        values = {flag.key: flag.value for flag in result.ten_capital.flags}
        if values["revenue_generating"] == "NO":
            assert values["revenue_over_700k"] == "NO"


def test_fit_score_is_bounded_and_ordered(saas_result, sparse_result):
    for result in (saas_result, sparse_result):
        assert 0 <= result.ten_capital.fit_score <= 100
    assert saas_result.ten_capital.fit_score > sparse_result.ten_capital.fit_score


def test_recommended_next_step_is_one_of_the_defined_actions(saas_result, biotech_result, sparse_result):
    allowed = (
        "HIGH PRIORITY", "QUALIFIED", "NURTURE", "EARLY", "NOT CURRENTLY QUALIFIED",
    )
    for result in (saas_result, biotech_result, sparse_result):
        assert result.ten_capital.recommended_next_step.startswith(allowed)
        assert result.ten_capital.explanation
