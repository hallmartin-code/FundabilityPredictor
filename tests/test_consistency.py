"""Contradiction detection and the derived-value rules."""

from __future__ import annotations

import pytest

from analysis.consistency import check_consistency
from analysis.derive import derive_fields
from models.evidence import ValueStatus
from models.profile import DeckProfile, SourceDocument


def _profile(**values) -> DeckProfile:
    profile = DeckProfile(source=SourceDocument(path="x", filename="x.pdf", file_type="pdf"))
    for name, value in values.items():
        profile.put(name, value, ValueStatus.EXPLICIT, 0.9, source_page=1, source_text=str(value))
    return profile


def _codes(profile: DeckProfile) -> set[str]:
    return {inc.code for inc in check_consistency(profile)}


# ---------------------------------------------------------------------------
# Contradictions
# ---------------------------------------------------------------------------
def test_committed_above_target_is_flagged():
    profile = _profile(target_raise=2_000_000.0, amount_committed=3_000_000.0)
    assert "committed_exceeds_target" in _codes(profile)


def test_percent_committed_inconsistent_with_amounts_is_flagged():
    profile = _profile(
        target_raise=2_000_000.0, amount_committed=200_000.0, percent_committed=75.0
    )
    assert "percent_committed_mismatch" in _codes(profile)


def test_matching_percent_and_amounts_are_not_flagged():
    profile = _profile(
        target_raise=2_000_000.0, amount_committed=1_000_000.0, percent_committed=50.0
    )
    assert "percent_committed_mismatch" not in _codes(profile)


def test_post_money_below_pre_money_is_flagged():
    profile = _profile(pre_money_valuation=20_000_000.0, post_money_valuation=15_000_000.0)
    assert "post_below_pre_money" in _codes(profile)


def test_runway_inconsistent_with_cash_and_burn_is_flagged():
    profile = _profile(cash_balance=600_000.0, burn_rate=100_000.0, runway_months=24.0)
    assert "runway_mismatch" in _codes(profile)


def test_runway_consistent_with_cash_and_burn_is_not_flagged():
    profile = _profile(cash_balance=600_000.0, burn_rate=100_000.0, runway_months=6.0)
    assert "runway_mismatch" not in _codes(profile)


def test_pre_revenue_claim_alongside_revenue_is_flagged():
    profile = _profile(annual_revenue=1_250_000.0)
    profile.raw_text_by_page = {1: "We are a pre-revenue company with strong pipeline."}
    assert "pre_revenue_with_revenue" in _codes(profile)


def test_series_a_without_prior_funding_is_flagged():
    profile = _profile(stage_declared="Series A", target_raise=12_000_000.0)
    assert "late_stage_no_prior_funding" in _codes(profile)


def test_market_sizing_out_of_order_is_flagged():
    profile = _profile(tam=100_000_000.0, sam=500_000_000.0)
    assert "market_sizing_order" in _codes(profile)


def test_past_maturity_date_is_flagged():
    profile = _profile(maturity_date="2020-01-01")
    codes = _codes(profile)
    assert "maturity_passed" in codes
    assert next(i for i in profile.inconsistencies if i.code == "maturity_passed").severity == "HIGH"


def test_a_clean_profile_raises_nothing():
    profile = _profile(
        target_raise=2_000_000.0,
        amount_committed=1_000_000.0,
        pre_money_valuation=8_000_000.0,
        post_money_valuation=10_000_000.0,
        cash_balance=600_000.0,
        burn_rate=100_000.0,
        runway_months=6.0,
        tam=1_000_000_000.0,
        sam=200_000_000.0,
        som=20_000_000.0,
    )
    assert _codes(profile) == set()


def test_contradictions_are_flagged_not_silently_corrected():
    """The values themselves must survive the check untouched."""
    profile = _profile(target_raise=2_000_000.0, amount_committed=3_000_000.0)
    check_consistency(profile)
    assert profile.number("amount_committed") == 3_000_000.0
    assert profile.number("target_raise") == 2_000_000.0


def test_inconsistencies_are_not_duplicated():
    profile = _profile(target_raise=1_000.0, amount_committed=9_000.0)
    check_consistency(profile)
    check_consistency(profile)
    assert len(profile.inconsistencies) == len({i.code for i in profile.inconsistencies})


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------
def test_percent_committed_is_derived_and_labelled():
    profile = _profile(target_raise=4_000_000.0, amount_committed=1_000_000.0)
    derive_fields(profile)
    value = profile.get("percent_committed")
    assert value.value == pytest.approx(25.0)
    assert value.status is ValueStatus.CALCULATED
    assert value.note


def test_amount_committed_is_derived_from_a_percentage():
    profile = _profile(target_raise=4_000_000.0, percent_committed=25.0)
    derive_fields(profile)
    assert profile.number("amount_committed") == pytest.approx(1_000_000.0)


def test_runway_is_derived_from_cash_and_burn():
    profile = _profile(cash_balance=900_000.0, burn_rate=100_000.0)
    derive_fields(profile)
    assert profile.number("runway_months") == pytest.approx(9.0)
    assert profile.get("runway_months").status is ValueStatus.CALCULATED


def test_arr_is_derived_from_mrr():
    profile = _profile(mrr=50_000.0)
    derive_fields(profile)
    assert profile.number("arr") == pytest.approx(600_000.0)
    assert profile.number("annual_revenue") == pytest.approx(600_000.0)


def test_derivation_never_overwrites_a_stated_value():
    profile = _profile(target_raise=4_000_000.0, amount_committed=1_000_000.0, percent_committed=40.0)
    derive_fields(profile)
    stated = profile.get("percent_committed")
    assert stated.value == 40.0
    assert stated.status is ValueStatus.EXPLICIT


def test_derivation_is_skipped_when_inputs_are_missing():
    profile = _profile(target_raise=4_000_000.0)
    derive_fields(profile)
    assert profile.get("percent_committed").is_missing
    assert profile.get("runway_months").is_missing


def test_derivation_rejects_implausible_results():
    """Cash divided by a trivial burn would imply a century of runway."""
    profile = _profile(cash_balance=900_000.0, burn_rate=1_000.0)
    derive_fields(profile)
    assert profile.get("runway_months").is_missing


def test_derived_confidence_never_exceeds_its_inputs():
    profile = DeckProfile(source=SourceDocument(path="x", filename="x.pdf", file_type="pdf"))
    profile.put("target_raise", 4_000_000.0, ValueStatus.EXPLICIT, 0.60, source_page=1)
    profile.put("amount_committed", 1_000_000.0, ValueStatus.EXPLICIT, 0.55, source_page=1)
    derive_fields(profile)
    assert profile.get("percent_committed").confidence <= 0.55
