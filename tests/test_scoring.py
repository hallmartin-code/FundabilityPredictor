"""Readiness scoring, stage inference, sector rules and confidence."""

from __future__ import annotations

import pytest

from analysis.confidence_model import score_confidence
from analysis.readiness_score import score_readiness
from analysis.sector_classifier import classify_sector
from analysis.stage_inference import infer_stage
from analysis.signals import piecewise
from config.settings import ConfigError, load_config
from models.company import SectorProfile, Stage, at_least, stage_from_text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_category_weights_sum_to_100(config):
    assert sum(float(c["weight"]) for c in config.categories.values()) == pytest.approx(100.0)


def test_config_validates_itself(config):
    config.validate()  # must not raise


def test_readiness_labels_cover_the_whole_range(config):
    assert config.readiness_label(92) == "HIGHLY FINANCEABLE"
    assert config.readiness_label(75) == "FINANCING READY"
    assert config.readiness_label(60) == "NEARLY READY"
    assert config.readiness_label(45) == "MATERIAL GAPS"
    assert config.readiness_label(10) == "EARLY / HIGH RISK"


def test_invalid_config_is_rejected(tmp_path, config):
    import yaml

    for name, data in (
        ("scoring_weights.yaml", config.scoring),
        ("sector_rules.yaml", config.sectors),
        ("probability_rules.yaml", config.probability),
    ):
        (tmp_path / name).write_text(yaml.safe_dump(data), encoding="utf-8")

    broken = yaml.safe_load((tmp_path / "scoring_weights.yaml").read_text(encoding="utf-8"))
    broken["categories"]["team"]["weight"] = 40  # weights no longer total 100
    (tmp_path / "scoring_weights.yaml").write_text(yaml.safe_dump(broken), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(str(tmp_path))


# ---------------------------------------------------------------------------
# Scoring maths
# ---------------------------------------------------------------------------
def test_piecewise_interpolates_and_clamps():
    points = [(0.0, 0.0), (10.0, 0.5), (20.0, 1.0)]
    assert piecewise(-5, points) == 0.0
    assert piecewise(5, points) == pytest.approx(0.25)
    assert piecewise(10, points) == pytest.approx(0.5)
    assert piecewise(100, points) == 1.0


def test_readiness_total_is_bounded_and_matches_its_parts(saas_result):
    readiness = saas_result.readiness
    assert 0 <= readiness.total <= 100
    assert readiness.total == pytest.approx(sum(c.points for c in readiness.categories), abs=0.1)
    for category in readiness.categories:
        assert 0 <= category.points <= category.weight + 1e-9


def test_strong_deck_scores_above_weak_deck(saas_result, sparse_result):
    assert saas_result.readiness.total > sparse_result.readiness.total + 25


def test_missing_data_earns_partial_credit_not_zero(sparse_result, config):
    """A silent deck must not be scored as an actively bad company."""
    assert sparse_result.readiness.total > 0
    traction = sparse_result.readiness.category("traction_revenue")
    assert traction is not None
    # Nothing is evidenced, so every signal sits at the missing-data credit.
    assert traction.evidenced_signals == 0
    assert traction.points == pytest.approx(traction.weight * config.missing_data_credit, abs=0.3)


# ---------------------------------------------------------------------------
# Sector classification
# ---------------------------------------------------------------------------
def test_saas_deck_is_classified_as_revenue_mode(saas_result):
    assert saas_result.sector.key == "saas"
    assert saas_result.sector.traction_mode == "revenue"


def test_biotech_deck_is_classified_as_milestone_mode(biotech_result):
    assert biotech_result.sector.key == "biotech"
    assert biotech_result.sector.traction_mode == "milestone"
    assert biotech_result.sector.is_capital_intensive


def test_sector_falls_back_to_neutral_without_evidence(sparse_result):
    assert sparse_result.sector.key == "unknown"
    assert sparse_result.sector.probability_modifier == {"p3": 1.0, "p6": 1.0, "p12": 1.0}


def test_pre_revenue_biotech_is_not_penalised_like_saas(biotech_result, saas_result):
    """The core sector-fairness requirement.

    The biotech deck has no revenue at all. Its traction category must still
    score respectably, because milestone evidence is the right proof for the
    sector.
    """
    traction = biotech_result.readiness.category("traction_revenue")
    assert traction is not None
    assert traction.pct > 50, "pre-revenue biotech was scored as if it should have revenue"
    assert biotech_result.profile.number("annual_revenue") is None


def test_investor_fit_recognises_normal_pre_revenue(biotech_result):
    fit = biotech_result.readiness.category("investor_fit")
    maturity = next(s for s in fit.signals if s.key == "maturity_match")
    assert "normal" in maturity.detail.lower()


# ---------------------------------------------------------------------------
# Stage inference
# ---------------------------------------------------------------------------
def test_stage_text_mapping():
    assert stage_from_text("Series A round") is Stage.SERIES_A
    assert stage_from_text("pre-seed") is Stage.PRE_SEED
    assert stage_from_text("Series D") is Stage.SERIES_C_PLUS
    assert stage_from_text("") is Stage.UNKNOWN


def test_at_least_never_passes_for_unknown():
    assert not at_least(Stage.UNKNOWN, Stage.SEED)
    assert at_least(Stage.SERIES_B, Stage.SEED)


def test_stage_is_inferred_from_financials(saas_result):
    assert saas_result.stage.inferred is Stage.SERIES_A
    assert saas_result.stage.confidence in ("HIGH", "MEDIUM")
    assert saas_result.stage.signals_used >= 4


def test_stage_is_unknown_when_nothing_supports_it(sparse_result):
    assert sparse_result.stage.inferred is Stage.UNKNOWN
    assert sparse_result.stage.confidence == "LOW"


def test_declared_and_inferred_stage_are_reported_separately(saas_result):
    assert saas_result.stage.declared.value
    assert saas_result.stage.inferred.value


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------
def test_confidence_is_bounded(saas_result, sparse_result):
    for result in (saas_result, sparse_result):
        assert 0 <= result.confidence.score <= 100
        assert result.confidence.level in ("HIGH", "MEDIUM", "LOW")


def test_complete_deck_earns_higher_confidence(saas_result, sparse_result):
    assert saas_result.confidence.score > sparse_result.confidence.score + 30


def test_missing_inputs_are_named_and_explained(sparse_result):
    assert sparse_result.confidence.missing_critical_fields
    assert sparse_result.confidence.notes
    assert any("commitments" in note or "revenue" in note for note in sparse_result.confidence.notes)


def test_high_readiness_does_not_force_high_confidence(saas_profile, config):
    """Readiness and confidence must be able to disagree."""
    stripped = saas_profile.model_copy(deep=True)
    for field in ("annual_revenue", "arr", "mrr", "runway_months", "burn_rate",
                  "customers", "paying_customers", "gross_margin", "prior_funding"):
        stripped.values.pop(field, None)

    sector = classify_sector(stripped, config)
    stage = infer_stage(stripped, config).effective
    readiness = score_readiness(stripped, sector, stage, config)
    confidence = score_confidence(stripped, config)

    assert readiness.total > 40
    assert confidence.level in ("LOW", "MEDIUM")
    assert confidence.score < 80


def test_estimated_values_count_for_less_than_stated_ones(saas_profile, config):
    from models.evidence import ValueStatus

    baseline = score_confidence(saas_profile, config).score
    downgraded = saas_profile.model_copy(deep=True)
    for field in ("target_raise", "annual_revenue"):
        value = downgraded.get(field)
        if value.is_known:
            downgraded.set(value.model_copy(update={"status": ValueStatus.ESTIMATED}))
    assert score_confidence(downgraded, config).score < baseline


# ---------------------------------------------------------------------------
# Sector profile helpers
# ---------------------------------------------------------------------------
def test_sector_profile_defaults_are_neutral():
    profile = SectorProfile()
    assert profile.key == "unknown"
    assert not profile.is_capital_intensive
    assert not profile.is_milestone_driven
