"""End-to-end CLI behaviour, including the review-and-override workflow."""

from __future__ import annotations

import json

import pytest

from app import EXIT_OK, EXIT_USAGE, _coerce, collect_overrides, main, safe_name
from services.llm import NullLLMService, apply_llm_extraction


class _Args:
    """Minimal stand-in for the parsed namespace."""

    def __init__(self, **kwargs):
        self.overrides = kwargs.get("overrides")
        self.set = kwargs.get("set")


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------
def test_safe_name_strips_unsafe_characters():
    assert safe_name("Acme / Widgets, Inc.") == "Acme_Widgets_Inc"
    assert safe_name("") == "Company"
    assert safe_name(None, fallback="Deck") == "Deck"


def test_safe_name_is_length_bounded():
    assert len(safe_name("A" * 200)) <= 60


# ---------------------------------------------------------------------------
# Override coercion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("target_raise", "$2.5M", 2_500_000.0),
        ("target_raise", "2500000", 2_500_000.0),
        ("percent_committed", "40%", 40.0),
        ("percent_committed", "40", 40.0),
        ("runway_months", "18", 18.0),
        ("runway_months", "2 years", 24.0),
        ("customers", "42", 42.0),
        ("founder_full_time", "yes", True),
        ("maturity_date", "March 2027", "2027-03-01"),
        ("founders", "Ada Lovelace, Alan Turing", ["Ada Lovelace", "Alan Turing"]),
        ("lead_investor", "Calder Ridge", "Calder Ridge"),
    ],
)
def test_override_values_are_coerced_to_the_field_type(field, raw, expected):
    assert _coerce(field, raw) == expected


def test_collect_overrides_from_set_flags():
    overrides = collect_overrides(_Args(set=["target_raise=$3M", "percent_committed=25%"]))
    assert overrides == {"target_raise": 3_000_000.0, "percent_committed": 25.0}


def test_collect_overrides_rejects_malformed_flags():
    from extractors.base import ExtractionError

    with pytest.raises(ExtractionError):
        collect_overrides(_Args(set=["target_raise"]))


def test_collect_overrides_ignores_unknown_fields():
    assert collect_overrides(_Args(set=["not_a_field=7"])) == {}


def test_collect_overrides_reads_the_annotated_template(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(
            {
                "target_raise": {"label": "Target Raise", "value": 5_000_000, "status": "EXPLICIT"},
                "lead_investor": {"label": "Lead Investor", "value": None},
            }
        ),
        encoding="utf-8",
    )
    overrides = collect_overrides(_Args(overrides=str(path)))
    assert overrides["target_raise"] == 5_000_000
    assert "lead_investor" not in overrides  # nulls are not overrides


def test_missing_overrides_file_is_an_error(tmp_path):
    from extractors.base import ExtractionError

    with pytest.raises(ExtractionError):
        collect_overrides(_Args(overrides=str(tmp_path / "nope.json")))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def test_analyse_writes_all_three_output_files(saas_deck, tmp_path):
    code = main(["--quiet", "analyse", str(saas_deck), "--out", str(tmp_path)])
    assert code == EXIT_OK

    pdfs = list(tmp_path.glob("*_Financing_Probability_Report.pdf"))
    data = list(tmp_path.glob("*_Financing_Probability_Data.json"))
    evidence = list(tmp_path.glob("*_Extraction_Evidence.json"))
    assert len(pdfs) == len(data) == len(evidence) == 1
    assert "Northwind" in pdfs[0].name


def test_analyse_can_also_write_html(saas_deck, tmp_path):
    main(["--quiet", "analyse", str(saas_deck), "--out", str(tmp_path), "--html"])
    assert list(tmp_path.glob("*_Financing_Probability_Report.html"))


def test_analyse_records_consent_only_when_flagged(saas_deck, tmp_path):
    main(["--quiet", "analyse", str(saas_deck), "--out", str(tmp_path)])
    record = json.loads(next(tmp_path.glob("*_Data.json")).read_text(encoding="utf-8"))
    assert record["consent"]["analysis_consent"] is None

    main([
        "--quiet", "analyse", str(saas_deck), "--out", str(tmp_path / "consented"),
        "--consent-analysis", "--consent-contact",
    ])
    consented = json.loads(
        next((tmp_path / "consented").glob("*_Data.json")).read_text(encoding="utf-8")
    )
    assert consented["consent"]["analysis_consent"] is True
    assert consented["consent"]["ten_capital_contact_consent"] is True


def test_review_writes_an_editable_overrides_template(saas_deck, tmp_path, capsys):
    template = tmp_path / "overrides.json"
    assert main(["--quiet", "review", str(saas_deck), "--save-overrides", str(template)]) == EXIT_OK

    data = json.loads(template.read_text(encoding="utf-8"))
    assert data["target_raise"]["value"] == 9_000_000
    assert data["target_raise"]["source_page"]
    assert "Extracted profile" in capsys.readouterr().out


def test_review_then_analyse_applies_the_corrections(saas_deck, tmp_path):
    template = tmp_path / "overrides.json"
    main(["--quiet", "review", str(saas_deck), "--save-overrides", str(template)])

    data = json.loads(template.read_text(encoding="utf-8"))
    data["target_raise"]["value"] = 4_000_000
    template.write_text(json.dumps(data), encoding="utf-8")

    main([
        "--quiet", "analyse", str(saas_deck), "--out", str(tmp_path / "out"),
        "--overrides", str(template),
    ])
    record = json.loads(next((tmp_path / "out").glob("*_Data.json")).read_text(encoding="utf-8"))
    assert record["financing"]["target_raise"] == 4_000_000
    assert "target_raise" in record["evidence"]["overridden_fields"]


def test_set_flag_overrides_an_extracted_value(saas_deck, tmp_path):
    main([
        "--quiet", "analyse", str(saas_deck), "--out", str(tmp_path),
        "--set", "lead_investor=Seeking a lead",
    ])
    record = json.loads(next(tmp_path.glob("*_Data.json")).read_text(encoding="utf-8"))
    assert record["ten_capital"]["lead_investor"] == "NO"


def test_explain_prints_the_bridge_and_breakdown(saas_deck, capsys):
    assert main(["--quiet", "explain", str(saas_deck)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Baseline probability" in out
    assert "Readiness score breakdown" in out
    assert "Traction & Revenue" in out


def test_market_condition_flag_changes_the_result(saas_deck, tmp_path):
    for market in ("favorable", "highly_difficult"):
        main([
            "--quiet", "analyse", str(saas_deck),
            "--out", str(tmp_path / market), "--market", market,
        ])
    favorable = json.loads(
        next((tmp_path / "favorable").glob("*_Data.json")).read_text(encoding="utf-8")
    )
    difficult = json.loads(
        next((tmp_path / "highly_difficult").glob("*_Data.json")).read_text(encoding="utf-8")
    )
    assert (
        difficult["prediction"]["probability_6_months"]
        < favorable["prediction"]["probability_6_months"]
    )
    assert difficult["prediction"]["market_condition"] == "Highly Difficult"


def test_unsupported_file_exits_with_a_usage_code(unsupported_file, tmp_path):
    assert main(["--quiet", "analyse", str(unsupported_file), "--out", str(tmp_path)]) == EXIT_USAGE


def test_missing_file_exits_with_a_usage_code(tmp_path):
    assert main(["--quiet", "analyse", str(tmp_path / "nope.pdf"), "--out", str(tmp_path)]) == EXIT_USAGE


def test_analyze_spelling_is_accepted(saas_deck, tmp_path):
    assert main(["--quiet", "analyze", str(saas_deck), "--out", str(tmp_path)]) == EXIT_OK


# ---------------------------------------------------------------------------
# LLM service abstraction
# ---------------------------------------------------------------------------
def test_default_pipeline_makes_no_llm_calls(saas_profile):
    """The deterministic path must not depend on a network service."""
    run = apply_llm_extraction(saas_profile.model_copy(deep=True), NullLLMService())
    assert run.requests == 0
    assert run.accepted == 0


def test_llm_is_off_by_default_even_when_a_key_is_present(monkeypatch):
    from config.settings import RuntimeSettings

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.delenv("FP_LLM_ENABLED", raising=False)
    assert RuntimeSettings.from_env().llm_enabled is False


def test_llm_can_be_enabled_explicitly(monkeypatch):
    from config.settings import RuntimeSettings

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("FP_LLM_ENABLED", "auto")
    assert RuntimeSettings.from_env().llm_enabled is True


def test_llm_values_never_displace_stronger_deck_evidence(saas_profile):
    """A low-confidence model proposal must not overwrite a stated value."""
    from services.llm import ExtractionRun

    class _Stub:
        name = "stub"
        model = "stub-model"

        def available(self) -> bool:
            return True

        def extract(self, profile):
            return (
                [
                    {
                        "field": "business_model",
                        "value": "SOMETHING ELSE",
                        "source_page": 3,
                        "quote": "not a real quote from the deck",
                        "confidence": 0.4,
                    }
                ],
                ExtractionRun(model=self.model, requests=1),
            )

    profile = saas_profile.model_copy(deep=True)
    original = profile.text("business_model")
    assert original, "the sample deck states a business model"

    apply_llm_extraction(profile, _Stub())
    assert profile.text("business_model") == original
