"""JSON outputs: schema shape, consent handling and evidence traceability."""

from __future__ import annotations

import json

import pytest

from models.evidence import ValueStatus
from models.fields import FIELD_SPECS
from reports.json_report import (
    ConsentRecord,
    build_evidence_map,
    build_lead_record,
    write_json,
)

REQUIRED_TOP_LEVEL = {
    "company", "financing", "traction", "prediction", "drivers", "blockers",
    "recommended_actions", "ten_capital", "evidence", "consent",
}


# ---------------------------------------------------------------------------
# Lead record shape
# ---------------------------------------------------------------------------
def test_lead_record_matches_the_defined_schema(saas_result):
    record = build_lead_record(saas_result)
    assert REQUIRED_TOP_LEVEL <= set(record)

    assert set(record["company"]) >= {
        "name", "website", "sector", "subsector", "stage_declared", "stage_inferred", "geography",
    }
    assert set(record["financing"]) >= {
        "target_raise", "amount_committed", "percent_committed", "valuation",
        "instrument", "prior_funding", "runway_months",
    }
    assert set(record["traction"]) >= {
        "annual_revenue", "arr", "revenue_growth", "gross_margin", "customers",
    }
    assert set(record["prediction"]) >= {
        "probability_3_months", "probability_6_months", "probability_12_months",
        "financing_readiness_score", "confidence_score", "confidence_level",
        "comparable_profile",
    }
    assert set(record["ten_capital"]) >= {
        "fit_score", "raise_over_2m", "twenty_five_percent_committed",
        "fifty_percent_committed", "revenue_over_700k", "revenue_generating",
        "lead_investor", "institutional_round", "recommended_next_step",
    }
    assert set(record["evidence"]) >= {
        "data_completeness_score", "missing_critical_fields", "estimated_fields", "source_file",
    }


def test_lead_record_is_valid_json(saas_result, tmp_path):
    path = write_json(build_lead_record(saas_result), tmp_path / "record.json")
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["company"]["name"] == saas_result.company_name


def test_probabilities_in_json_match_the_analysis(saas_result):
    record = build_lead_record(saas_result)
    final = saas_result.prediction.final
    assert record["prediction"]["probability_3_months"] == final.p3
    assert record["prediction"]["probability_6_months"] == final.p6
    assert record["prediction"]["probability_12_months"] == final.p12


def test_missing_values_serialise_as_null_not_zero(sparse_result):
    record = build_lead_record(sparse_result)
    assert record["financing"]["target_raise"] is None
    assert record["traction"]["annual_revenue"] is None
    assert record["financing"]["runway_months"] is None


def test_record_carries_readiness_flags_and_evidence_quality(saas_result):
    record = build_lead_record(saas_result)
    assert record["ten_capital"]["raise_over_2m"] in ("YES", "NO", "UNKNOWN")
    assert 0 <= record["evidence"]["data_completeness_score"] <= 100
    assert isinstance(record["evidence"]["missing_critical_fields"], list)
    assert isinstance(record["evidence"]["estimated_fields"], list)


def test_record_includes_the_probability_bridge(saas_result):
    bridge = build_lead_record(saas_result)["prediction"]["probability_bridge"]
    assert bridge and bridge[0]["kind"] == "baseline"
    assert all({"label", "kind", "delta", "running"} <= set(step) for step in bridge)


def test_market_condition_is_marked_unverified(saas_result):
    prediction = build_lead_record(saas_result)["prediction"]
    assert prediction["market_condition_verified"] is False


def test_disclaimer_is_present(saas_result):
    assert "does not constitute investment" in build_lead_record(saas_result)["disclaimer"]


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------
def test_consent_defaults_to_null(saas_result):
    consent = build_lead_record(saas_result)["consent"]
    assert consent["analysis_consent"] is None
    assert consent["ten_capital_contact_consent"] is None
    assert consent["timestamp"] is None


def test_absent_consent_is_never_converted_to_true(saas_result):
    """The critical privacy property: silence is not agreement."""
    for record in (
        build_lead_record(saas_result),
        build_lead_record(saas_result, ConsentRecord()),
        build_lead_record(saas_result, ConsentRecord(analysis_consent=None)),
    ):
        assert record["consent"]["analysis_consent"] is not True
        assert record["consent"]["ten_capital_contact_consent"] is not True


def test_explicit_consent_is_recorded_with_a_timestamp(saas_result):
    consent = build_lead_record(
        saas_result, ConsentRecord(analysis_consent=True, ten_capital_contact_consent=True)
    )["consent"]
    assert consent["analysis_consent"] is True
    assert consent["ten_capital_contact_consent"] is True
    assert consent["timestamp"]


def test_declined_consent_is_recorded_as_false(saas_result):
    consent = build_lead_record(
        saas_result, ConsentRecord(analysis_consent=True, ten_capital_contact_consent=False)
    )["consent"]
    assert consent["ten_capital_contact_consent"] is False


# ---------------------------------------------------------------------------
# Evidence map
# ---------------------------------------------------------------------------
def test_evidence_map_covers_every_registry_field(saas_profile):
    evidence = build_evidence_map(saas_profile)
    assert len(evidence["fields"]) == len(FIELD_SPECS)
    assert {entry["field"] for entry in evidence["fields"]} == {s.name for s in FIELD_SPECS}


def test_every_evidence_entry_declares_a_value_type(saas_profile):
    allowed = {status.value for status in ValueStatus}
    for entry in build_evidence_map(saas_profile)["fields"]:
        assert entry["status"] in allowed
        assert 0.0 <= entry["confidence"] <= 1.0


def test_explicit_entries_carry_a_source_page_and_text(saas_profile):
    for entry in build_evidence_map(saas_profile)["fields"]:
        if entry["status"] == ValueStatus.EXPLICIT.value:
            assert entry["source_page"] is not None
            assert entry["source_text"]


def test_missing_entries_have_no_value(saas_profile):
    for entry in build_evidence_map(saas_profile)["fields"]:
        if entry["status"] == ValueStatus.MISSING.value:
            assert entry["value"] is None


def test_evidence_map_reports_extraction_quality(saas_profile):
    quality = build_evidence_map(saas_profile)["extraction_quality"]
    assert quality["pages_total"] > 0
    assert "warnings" in quality


def test_evidence_map_is_serialisable(saas_profile, tmp_path):
    path = write_json(build_evidence_map(saas_profile), tmp_path / "evidence.json")
    assert json.loads(path.read_text(encoding="utf-8"))["fields"]


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------
def test_override_is_recorded_separately_from_extraction(saas_profile):
    profile = saas_profile.model_copy(deep=True)
    original = profile.number("target_raise")
    profile.apply_override("target_raise", 12_000_000.0)

    updated = profile.get("target_raise")
    assert updated.value == 12_000_000.0
    assert updated.overridden is True
    assert updated.extracted_value == original
    assert updated.extracted_status is ValueStatus.EXPLICIT

    entry = next(
        e for e in build_evidence_map(profile)["fields"] if e["field"] == "target_raise"
    )
    assert entry["overridden"] is True
    assert entry["extracted_value"] == original


def test_overrides_flow_into_the_analysis(saas_deck, config):
    from analysis.pipeline import analyse_profile, extract_profile

    baseline = analyse_profile(extract_profile(saas_deck), config)
    overridden = analyse_profile(
        extract_profile(saas_deck, overrides={"target_raise": 60_000_000.0}), config
    )
    # A round six times the stage norm must reduce the probability of closing.
    assert overridden.prediction.final.p6 < baseline.prediction.final.p6
    assert "target_raise" in overridden.profile.overridden_fields()


def test_override_of_an_unknown_field_is_ignored(saas_profile):
    from analysis.pipeline import apply_overrides

    applied = apply_overrides(saas_profile.model_copy(deep=True), {"not_a_field": 1})
    assert applied == []
