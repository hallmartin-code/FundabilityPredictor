"""Claude extraction: quote verification, merge policy and failure handling.

Every test here uses a stub service. The suite must never make a network call
or need an API key.
"""

from __future__ import annotations

import pytest

from models.evidence import ValueStatus
from models.profile import DeckProfile, SourceDocument
from services.llm import (
    EXTRACTION_SCHEMA,
    LLM_EXCLUDED_FIELDS,
    ClaudeExtractionService,
    ExtractionRun,
    NullLLMService,
    _chunk_pages,
    apply_llm_extraction,
    build_llm_service,
    verify_literal,
    verify_quote,
)
from models.fields import FIELD_SPECS


PAGES = {
    1: "Northwind Analytics\nRevenue intelligence for industrial distributors",
    4: "Traction\nARR: $2,400,000\nGross margin: 82%\n42 paying customers",
    9: "The Round\nCurrently raising $9,000,000 Series A\nLead investor: Calder Ridge Ventures",
    12: "SAFE: Convertible Debt:\nSeries A:\nClosed Tranche 1 Tranche 2\nTBD\n$350K $750k $1.25M",
}


def _profile() -> DeckProfile:
    profile = DeckProfile(
        source=SourceDocument(path="x", filename="deck.pdf", file_type="pdf", page_count=12)
    )
    profile.raw_text_by_page = dict(PAGES)
    return profile


class StubService:
    """A service that returns whatever proposals the test supplies."""

    name = "stub"
    model = "stub-model"

    def __init__(self, proposals: list[dict]) -> None:
        self.proposals = proposals

    def available(self) -> bool:
        return True

    def extract(self, profile):  # noqa: ANN001
        run = ExtractionRun(model=self.model, requests=1, input_tokens=100, output_tokens=50)
        return self.proposals, run


def _proposal(field: str, value: str, page: int, quote: str, confidence: float = 0.9) -> dict:
    return {
        "field": field,
        "value": value,
        "source_page": page,
        "quote": quote,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------
def test_exact_quote_is_verified():
    assert verify_quote(_profile(), "ARR: $2,400,000", 4) == 4


def test_quote_is_found_on_a_different_page_than_cited():
    """A misattributed page is corrected, not rejected."""
    assert verify_quote(_profile(), "Currently raising $9,000,000", 3) == 9


def test_whitespace_differences_do_not_defeat_verification():
    assert verify_quote(_profile(), "ARR:    $2,400,000  ", 4) == 4


def test_invented_quote_is_not_verified():
    assert verify_quote(_profile(), "ARR: $14,000,000 and growing fast", 4) is None


def test_trivially_short_quote_is_not_verified():
    assert verify_quote(_profile(), "$", 4) is None


def test_literal_verification_rescues_table_layouts():
    """A figure split across table cells has no contiguous quote.

    The quote "SAFE: Closed $350K" is not a substring of the page, but $350K
    genuinely is on it, so the value is still evidenced.
    """
    assert verify_quote(_profile(), "SAFE: Closed $350K", 12) is None
    assert verify_literal(_profile(), "$350K", 12) == 12


def test_literal_verification_rejects_a_figure_that_is_absent():
    assert verify_literal(_profile(), "$7,400,000", 12) is None


# ---------------------------------------------------------------------------
# Merge policy
# ---------------------------------------------------------------------------
def test_verified_value_is_recorded_as_explicit():
    profile = _profile()
    run = apply_llm_extraction(
        profile, StubService([_proposal("arr", "$2,400,000", 4, "ARR: $2,400,000")])
    )
    value = profile.get("arr")
    assert value.value == pytest.approx(2_400_000)
    assert value.status is ValueStatus.EXPLICIT
    assert value.source_page == 4
    assert "stub" in (value.note or "")
    assert run.accepted == 1


def test_unverified_number_is_discarded_entirely():
    """The central anti-hallucination guarantee."""
    profile = _profile()
    run = apply_llm_extraction(
        profile,
        StubService([_proposal("arr", "$14,000,000", 4, "ARR reached $14M last quarter")]),
    )
    assert profile.get("arr").is_missing
    assert "arr" in run.rejected_unverified
    assert run.accepted == 0


def test_unverified_text_is_downgraded_rather_than_dropped():
    profile = _profile()
    run = apply_llm_extraction(
        profile,
        StubService([_proposal("business_model", "Per-seat SaaS", 4, "a paraphrase, not a quote")]),
    )
    value = profile.get("business_model")
    assert value.is_known
    assert value.status is ValueStatus.ESTIMATED
    assert value.confidence <= 0.5
    assert "business_model" in run.downgraded


def test_llm_does_not_overwrite_a_more_confident_existing_value():
    profile = _profile()
    profile.put("arr", 2_400_000.0, ValueStatus.EXPLICIT, 0.95, source_page=4)
    apply_llm_extraction(
        profile,
        StubService([_proposal("arr", "$2,400,000", 4, "ARR: $2,400,000", confidence=0.6)]),
    )
    value = profile.get("arr")
    assert value.confidence == pytest.approx(0.95)
    assert value.note is None  # untouched by the merge


def test_llm_replaces_a_less_confident_existing_value():
    profile = _profile()
    profile.put("arr", 999.0, ValueStatus.EXPLICIT, 0.30, source_page=1)
    apply_llm_extraction(
        profile, StubService([_proposal("arr", "$2,400,000", 4, "ARR: $2,400,000", 0.95)])
    )
    assert profile.get("arr").value == pytest.approx(2_400_000)


def test_excluded_fields_are_ignored():
    """Derived fields must be computed, never proposed."""
    profile = _profile()
    apply_llm_extraction(
        profile, StubService([_proposal("percent_committed", "60%", 9, "60% committed")])
    )
    assert profile.get("percent_committed").is_missing


def test_unknown_field_names_are_ignored():
    profile = _profile()
    apply_llm_extraction(profile, StubService([_proposal("not_a_field", "x", 1, "Northwind")]))
    assert profile.known_fields() == []


def test_unparseable_value_is_skipped():
    profile = _profile()
    apply_llm_extraction(
        profile, StubService([_proposal("arr", "lots of revenue", 4, "ARR: $2,400,000")])
    )
    assert profile.get("arr").is_missing


def test_values_are_coerced_with_the_same_parsers_as_the_rules():
    profile = _profile()
    apply_llm_extraction(
        profile,
        StubService([
            _proposal("gross_margin", "82%", 4, "Gross margin: 82%"),
            _proposal("customers", "42", 4, "42 paying customers"),
            _proposal("lead_investor", "Calder Ridge Ventures", 9, "Lead investor: Calder Ridge"),
        ]),
    )
    assert profile.number("gross_margin") == pytest.approx(82)
    assert profile.number("customers") == pytest.approx(42)
    assert profile.text("lead_investor") == "Calder Ridge Ventures"


# ---------------------------------------------------------------------------
# Service wiring
# ---------------------------------------------------------------------------
def test_null_service_makes_no_proposals():
    profile = _profile()
    run = apply_llm_extraction(profile, NullLLMService())
    assert run.requests == 0
    assert profile.known_fields() == []


def test_build_service_returns_null_when_disabled():
    assert isinstance(build_llm_service(False), NullLLMService)


def test_build_service_returns_null_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(build_llm_service(True), NullLLMService)


def test_service_reports_unavailable_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ClaudeExtractionService(api_key=None).available() is False


def test_api_failure_degrades_to_rules_only():
    """An API outage must not fail the analysis."""

    class Failing:
        name = "failing"
        model = "x"

        def available(self) -> bool:
            return True

        def extract(self, profile):  # noqa: ANN001
            run = ExtractionRun()
            run.errors.append("simulated outage")
            return [], run

    profile = _profile()
    profile.put("arr", 2_400_000.0, ValueStatus.EXPLICIT, 0.9, source_page=4)
    run = apply_llm_extraction(profile, Failing())
    assert run.errors
    assert profile.number("arr") == pytest.approx(2_400_000)


# ---------------------------------------------------------------------------
# Prompt and schema integrity
# ---------------------------------------------------------------------------
def test_schema_enumerates_exactly_the_allowed_fields():
    allowed = set(EXTRACTION_SCHEMA["properties"]["fields"]["items"]["properties"]["field"]["enum"])
    expected = {s.name for s in FIELD_SPECS} - set(LLM_EXCLUDED_FIELDS)
    assert allowed == expected


def test_schema_requires_a_quote_and_page_for_every_field():
    required = EXTRACTION_SCHEMA["properties"]["fields"]["items"]["required"]
    assert {"field", "value", "source_page", "quote", "confidence"} == set(required)


def test_schema_avoids_unsupported_numeric_constraints():
    """The structured-output validator rejects minimum/maximum."""
    item = EXTRACTION_SCHEMA["properties"]["fields"]["items"]
    for prop in item["properties"].values():
        assert "minimum" not in prop
        assert "maximum" not in prop


def test_cost_is_computed_from_token_usage():
    run = ExtractionRun(model="claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert run.cost_usd == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def test_small_deck_fits_one_chunk():
    assert len(_chunk_pages(PAGES)) == 1


def test_large_deck_splits_on_page_boundaries():
    big = {n: "x" * 20_000 for n in range(1, 8)}
    chunks = _chunk_pages(big)
    assert len(chunks) > 1
    # Every page appears exactly once, with its marker intact.
    joined = "".join(chunks)
    for page in big:
        assert joined.count(f"--- PAGE {page} ---") == 1


def test_chunking_preserves_absolute_page_numbers():
    chunks = _chunk_pages({5: "five", 6: "six"})
    assert "--- PAGE 5 ---" in chunks[0]
    assert "--- PAGE 6 ---" in chunks[0]
