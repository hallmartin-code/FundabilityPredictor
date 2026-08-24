"""The document-structure template.

The template is a specification, so the tests assert two things: that it still
renders through the real layout code, and that it contains no company data.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
sys.path.insert(0, str(DOCS))

from generate_template import build_template_result  # noqa: E402
from models.fields import FIELD_SPECS  # noqa: E402
from reports.json_report import build_evidence_map, build_lead_record  # noqa: E402
from reports.pdf_report import TemplateTokens, render_pdf  # noqa: E402

#: Anything that would indicate real company data survived into the template.
COMPANY_MARKERS = (
    "AccuBreath", "Northwind", "Helix", "Orbital", "Calder Ridge",
    "Keiretsu", "Madsen", "Bioworks", "Analytics", "Freight",
)


@pytest.fixture(scope="module")
def template_pdf(tmp_path_factory) -> Path:
    result = build_template_result()
    path = tmp_path_factory.mktemp("template") / "report_template.pdf"
    return render_pdf(result, path, tokens=TemplateTokens())


def _text(path: Path) -> str:
    """Extracted text with whitespace flattened (see test_pdf._page_text)."""
    raw = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return re.sub(r"\s+", " ", raw)


# ---------------------------------------------------------------------------
# The template renders through the production layout
# ---------------------------------------------------------------------------
def test_template_is_one_page(template_pdf):
    assert len(PdfReader(str(template_pdf)).pages) == 1


def test_template_contains_no_company_data(template_pdf):
    text = _text(template_pdf)
    found = [marker for marker in COMPANY_MARKERS if marker in text]
    assert not found, f"company data leaked into the template: {found}"


def test_template_shows_placeholder_tokens_not_numbers(template_pdf):
    text = _text(template_pdf)
    for token in ("XX%", "XX/100", "$X.XM", "[DATE]"):
        assert token in text, f"missing placeholder token {token}"


def test_template_keeps_every_fixed_section_heading(template_pdf):
    text = _text(template_pdf)
    for heading in (
        "FUNDABILITY REPORT",
        "PROBABILITY OF COMPLETING ROUND",
        "PROBABILITY DRIVERS",
        "CRITICAL BLOCKERS",
        "COMPARABLE FINANCING PROFILE",
        "FIVE ACTIONS TO INCREASE FUNDING PROBABILITY",
        "TEN CAPITAL ASSESSMENT",
        "FUNDABILITY",
        "CONFIDENCE",
        "3 MONTHS",
        "6 MONTHS",
        "12 MONTHS",
    ):
        assert heading in text, f"template lost fixed heading: {heading}"


def test_template_keeps_the_required_disclaimers(template_pdf):
    text = _text(template_pdf)
    assert "CONFIDENTIAL - FOR INTENDED RECIPIENT ONLY" in text
    assert "not guarantees of financing outcomes" in text
    assert "does not constitute investment, legal, tax, or securities advice" in text
    assert "MARKET CONDITION ASSUMPTION" in text


def test_template_shows_all_ten_ten_capital_flags(template_pdf):
    text = _text(template_pdf)
    for label in (
        "Raise >= $2M", ">= 25% committed", ">= 50% committed", "Revenue >= $700K",
        "Revenue generating", "Institutional round", "Lead investor identified",
        "Runway >= 9 months", "Investor materials complete", "Likely fundraising candidate",
    ):
        assert label in text, f"template lost flag: {label}"


def test_template_shows_the_full_flag_value_domain(template_pdf):
    text = _text(template_pdf)
    for value in ("YES", "NO", "UNKNOWN", "REVIEW"):
        assert value in text


def test_template_shows_five_actions_and_five_blockers(template_pdf):
    text = _text(template_pdf)
    for n in range(1, 6):
        assert f"[Action {n}:" in text
        assert f"[Blocker {n} title]" in text


# ---------------------------------------------------------------------------
# Committed artifacts stay in sync
# ---------------------------------------------------------------------------
def test_committed_template_artifacts_exist():
    for name in (
        "TEMPLATE.md",
        "report_template.pdf",
        "report_template.html",
        "lead_record_template.json",
        "evidence_map_template.json",
    ):
        assert (DOCS / name).exists(), f"missing template artifact: {name}"


def test_committed_pdf_template_is_current(template_pdf):
    """A regenerated template must still be one page and data-free."""
    committed = DOCS / "report_template.pdf"
    assert len(PdfReader(str(committed)).pages) == 1
    text = _text(committed)
    assert not [m for m in COMPANY_MARKERS if m in text]


def test_evidence_template_lists_every_registry_field():
    evidence = json.loads(
        (DOCS / "evidence_map_template.json").read_text(encoding="utf-8")
    )
    assert len(evidence["fields"]) == len(FIELD_SPECS)
    assert {entry["field"] for entry in evidence["fields"]} == {s.name for s in FIELD_SPECS}
    # Structure only: no field may carry a value.
    assert all(entry["value"] is None for entry in evidence["fields"])
    assert all(entry["source_page"] is None for entry in evidence["fields"])
    assert all(count is None for count in evidence["status_counts"].values())


def test_lead_record_template_matches_the_live_schema():
    """The template must keep the same keys the app actually emits."""
    template = json.loads(
        (DOCS / "lead_record_template.json").read_text(encoding="utf-8")
    )
    live = build_lead_record(build_template_result())
    assert set(template) == set(live)
    for section in ("company", "financing", "traction", "prediction", "ten_capital", "evidence", "consent"):
        assert set(template[section]) == set(live[section]), f"{section} drifted"


def test_lead_record_template_values_are_blank():
    template = json.loads(
        (DOCS / "lead_record_template.json").read_text(encoding="utf-8")
    )
    for section in ("company", "financing", "traction"):
        assert all(value is None for value in template[section].values())


def test_template_consent_defaults_to_null():
    """The template must not model consent as granted or refused."""
    template = json.loads(
        (DOCS / "lead_record_template.json").read_text(encoding="utf-8")
    )
    consent = template["consent"]
    assert consent == {
        "analysis_consent": None,
        "ten_capital_contact_consent": None,
        "timestamp": None,
    }


def test_template_retains_fixed_text_rather_than_nulling_it():
    template = json.loads(
        (DOCS / "lead_record_template.json").read_text(encoding="utf-8")
    )
    assert template["schema_version"]
    assert "does not constitute investment" in template["disclaimer"]


def test_evidence_map_template_is_derived_from_the_registry():
    """Adding a field to the registry must show up in the template."""
    evidence = build_evidence_map(build_template_result().profile)
    assert len(evidence["fields"]) == len(FIELD_SPECS)


# ---------------------------------------------------------------------------
# Template mode must not affect real reports
# ---------------------------------------------------------------------------
def test_real_reports_are_unaffected_by_template_mode(saas_result, tmp_path):
    plain = render_pdf(saas_result, tmp_path / "plain.pdf")
    text = _text(plain)
    assert "XX%" not in text
    assert "$X.XM" not in text
    assert "[DATE]" not in text
    assert saas_result.company_name in text
