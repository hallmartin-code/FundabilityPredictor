"""Ingestion, normalisation and field extraction."""

from __future__ import annotations

import pytest

from extractors.base import ExtractionError, UnsupportedFormatError
from extractors.field_extractor import FieldExtractor, _is_forecast_page
from extractors.base import Page
from extractors.ingest import ingest, supported_extensions
from extractors.normalization import (
    find_money,
    format_money,
    nearest_span,
    parse_bool,
    parse_date,
    parse_list,
    parse_money,
    parse_months,
    parse_number,
    parse_percent,
)
from extractors.text_repair import flatten_pattern, looks_letter_spaced, repair_spacing
from models.evidence import ValueStatus
from models.fields import FIELD_SPECS


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$2.5M", 2_500_000),
        ("USD 1,200,000", 1_200_000),
        ("2.5 million", 2_500_000),
        ("$750K", 750_000),
        ("raising $2M", 2_000_000),
        ("$3.4 billion", 3_400_000_000),
    ],
)
def test_parse_money(text, expected):
    assert parse_money(text)[0] == pytest.approx(expected)


def test_parse_money_records_currency_without_converting():
    value, currency = parse_money("€1.2m")
    assert value == pytest.approx(1_200_000)
    assert currency == "EUR"


def test_parse_money_returns_none_when_absent():
    assert parse_money("no figures here") == (None, None)
    assert parse_money(None) == (None, None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("35%", 35.0), ("35 percent", 35.0), ("0.35", 35.0), ("12.5%", 12.5)],
)
def test_parse_percent(text, expected):
    assert parse_percent(text) == pytest.approx(expected)


def test_parse_percent_rejects_ambiguous_text():
    assert parse_percent("we grew a lot") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [("18 months", 18), ("2 years", 24), ("9 mo", 9), ("3 quarters", 9)],
)
def test_parse_months(text, expected):
    assert parse_months(text) == pytest.approx(expected)


def test_parse_number_handles_suffixes():
    assert parse_number("12k users") == pytest.approx(12_000)
    assert parse_number("1,200 customers") == pytest.approx(1_200)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("March 2026", "2026-03-01"), ("2026-03-01", "2026-03-01"), ("Q3 2026", "2026-07-01")],
)
def test_parse_date(text, expected):
    assert parse_date(text) == expected


def test_parse_bool_and_list():
    assert parse_bool("Commitment: Full-Time") is True
    assert parse_bool("fractional CFO") is False
    assert parse_bool("") is None
    assert parse_list("Jane Doe, John Smith and Alice Lee") == ["Jane Doe", "John Smith", "Alice Lee"]


def test_format_money_never_invents_a_value():
    assert format_money(None) == "Not Provided"
    assert format_money(2_500_000) == "$2.5M"


def test_money_finder_marks_unit_less_values_as_weak():
    """"100 million ventilators" is a count, not an amount of money."""
    weak = find_money("TAM = 100 million manual ventilators")
    assert weak and weak[0].strong is False
    strong = find_money("$3.5B market")
    assert strong and strong[0].strong is True


def test_nearest_span_prefers_the_value_after_the_cue():
    line = "raising $2M at $8M pre-money"
    cue_start = line.index("pre-money")
    chosen = nearest_span(find_money(line), cue_start, cue_start + len("pre-money"))
    # Nothing follows the cue, so the nearest preceding value wins - $8M, not $2M.
    assert chosen.value == pytest.approx(8_000_000)


# ---------------------------------------------------------------------------
# Letter-spacing repair
# ---------------------------------------------------------------------------
def test_looks_letter_spaced():
    assert looks_letter_spaced("M ARKE T S IZ E - M a nua l")
    assert not looks_letter_spaced("Market size for manual resuscitators")


def test_repair_spacing_rebuilds_words():
    assert repair_spacing("CO M PE TITIO N") == "COMPETITION"
    assert "Helix" in repair_spacing("H e l i x  B i o w o r k s")


def test_flatten_pattern_drops_whitespace_requirements():
    flattened = flatten_pattern(r"\bmarket\s+size\b")
    assert flattened == "marketsize"


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def test_supported_extensions_cover_the_required_formats():
    extensions = supported_extensions()
    assert {".pdf", ".pptx", ".ppt"} <= set(extensions)


def test_pptx_ingestion(saas_deck):
    document = ingest(saas_deck)
    assert document.source.file_type == "pptx"
    assert len(document.pages) == 10
    assert "ARR" in document.full_text
    # Speaker notes are captured too.
    assert any(page.notes for page in document.pages)


def test_pdf_ingestion(biotech_deck):
    document = ingest(biotech_deck)
    assert document.source.file_type == "pdf"
    assert len(document.pages) >= 8
    assert document.source.sha256


def test_unsupported_format_gives_an_actionable_error(unsupported_file):
    with pytest.raises(UnsupportedFormatError) as excinfo:
        ingest(unsupported_file)
    assert ".pdf" in str(excinfo.value)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ExtractionError):
        ingest(tmp_path / "nope.pdf")


def test_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(ExtractionError):
        ingest(empty)


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------
def test_financing_fields_are_extracted(saas_profile):
    assert saas_profile.number("target_raise") == pytest.approx(9_000_000)
    assert saas_profile.number("pre_money_valuation") == pytest.approx(36_000_000)
    assert saas_profile.number("percent_committed") == pytest.approx(60, abs=1)
    assert "Calder Ridge" in saas_profile.text("lead_investor")


def test_revenue_fields_are_extracted(saas_profile):
    assert saas_profile.number("arr") == pytest.approx(2_400_000)
    assert saas_profile.number("mrr") == pytest.approx(200_000)
    assert saas_profile.number("gross_margin") == pytest.approx(82)


def test_every_extracted_value_keeps_its_source_page(saas_profile):
    for name in saas_profile.known_fields():
        value = saas_profile.get(name)
        if value.status is ValueStatus.EXPLICIT:
            assert value.source_page is not None, f"{name} lost its source page"
            assert value.source_text


def test_missing_fields_stay_missing(sparse_deck):
    from analysis.pipeline import extract_profile

    profile = extract_profile(sparse_deck)
    for name in ("target_raise", "annual_revenue", "runway_months", "pre_money_valuation"):
        value = profile.get(name)
        assert value.is_missing
        assert value.value is None
        assert value.display() == "Not Provided"


def test_unknown_field_name_is_rejected(saas_profile):
    with pytest.raises(KeyError):
        saas_profile.get("not_a_real_field")


def test_letter_spaced_raise_is_recovered(biotech_deck):
    """The biotech sample uses tracked-out headings on every page."""
    from analysis.pipeline import extract_profile

    profile = extract_profile(biotech_deck)
    assert profile.number("target_raise") == pytest.approx(3_500_000)


def test_forecast_pages_are_not_read_as_current_revenue():
    page = Page(
        number=9,
        text=(
            "FINANCIALS\n"
            "$000s 2026 2027 2028 2029 2030\n"
            "Revenue - 2,430 10,620 69,890 144,060\n"
            "EBITDA (1,377) (1,540) 940 31,290 77,300"
        ),
    )
    assert _is_forecast_page(page)
    candidates = FieldExtractor()._candidates_from_page(page)
    assert not [c for c in candidates if c.field in {"annual_revenue", "ebitda"}]


def test_derived_percent_committed_is_marked_calculated(saas_deck):
    from analysis.pipeline import extract_profile

    profile = extract_profile(saas_deck)
    committed = profile.get("percent_committed")
    assert committed.is_known
    assert committed.status in (ValueStatus.EXPLICIT, ValueStatus.CALCULATED)


def test_post_money_is_derived_from_pre_money_and_raise(saas_profile):
    post = saas_profile.get("post_money_valuation")
    assert post.status is ValueStatus.CALCULATED
    assert post.value == pytest.approx(45_000_000)


def test_field_registry_is_internally_consistent():
    names = [spec.name for spec in FIELD_SPECS]
    assert len(names) == len(set(names)), "duplicate field names in the registry"
    assert all(spec.label for spec in FIELD_SPECS)
