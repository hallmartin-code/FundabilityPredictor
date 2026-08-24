"""PDF rendering: exactly one page, nothing clipped, nothing outside margins."""

from __future__ import annotations

import re

import pytest
from pypdf import PdfReader

from models.evidence import ValueStatus
from reports.html_report import render_html
from reports.pdf_report import (
    FinancingReportRenderer,
    _clip,
    _fit_text,
    _measure,
    render_pdf,
)
from reports.theme import DENSITY_LADDER, page_spec

pdfium = pytest.importorskip("pypdfium2")


def _render(result, tmp_path, name="report.pdf", page="LETTER"):
    return render_pdf(result, tmp_path / name, page=page)


def _page_text(path) -> str:
    """Extracted text with whitespace flattened.

    A required phrase that happens to wrap across two lines is still present
    in the document; asserting against raw extraction would fail on a layout
    change alone, which is not what these tests are checking.
    """
    raw = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return re.sub(r"\s+", " ", raw)


# ---------------------------------------------------------------------------
# The one-page requirement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["saas_result", "biotech_result", "sparse_result"])
def test_report_is_exactly_one_page(fixture, request, tmp_path):
    result = request.getfixturevalue(fixture)
    path = _render(result, tmp_path, f"{fixture}.pdf")
    assert len(PdfReader(str(path)).pages) == 1


@pytest.mark.parametrize("size", ["LETTER", "A4"])
def test_both_page_sizes_stay_on_one_page(saas_result, tmp_path, size):
    path = _render(saas_result, tmp_path, f"{size}.pdf", page=size)
    reader = PdfReader(str(path))
    assert len(reader.pages) == 1
    spec = page_spec(size)
    box = reader.pages[0].mediabox
    assert float(box.width) == pytest.approx(spec.width, abs=1)
    assert float(box.height) == pytest.approx(spec.height, abs=1)


def test_long_company_name_does_not_break_the_layout(saas_result, tmp_path):
    stretched = saas_result.model_copy(deep=True)
    stretched.profile.put(
        "company_name",
        "Northwind Analytics International Holdings and Industrial Data Systems Corporation",
        ValueStatus.EXPLICIT,
        0.9,
    )
    path = _render(stretched, tmp_path, "long_name.pdf")
    assert len(PdfReader(str(path)).pages) == 1


def test_very_long_blockers_and_actions_still_fit(saas_result, tmp_path):
    """Force the renderer down the density ladder."""
    stressed = saas_result.model_copy(deep=True)
    filler = "This is a deliberately long sentence used to stress the layout engine. " * 6
    for blocker in stressed.blockers:
        blocker.detail = filler
    for action in stressed.actions:
        action.action = filler
        action.why = filler
    stressed.comparable.rationale = filler
    stressed.confidence.notes = [filler] * 8

    path = _render(stressed, tmp_path, "stressed.pdf")
    assert len(PdfReader(str(path)).pages) == 1


def test_missing_data_does_not_break_rendering(sparse_result, tmp_path):
    emptied = sparse_result.model_copy(deep=True)
    emptied.profile.values.clear()
    emptied.blockers.clear()
    emptied.prediction.positive_signals.clear()
    path = _render(emptied, tmp_path, "empty.pdf")
    assert len(PdfReader(str(path)).pages) == 1
    assert "Not Provided" in _page_text(path)


# ---------------------------------------------------------------------------
# Required content
# ---------------------------------------------------------------------------
def test_all_required_sections_appear(saas_result, tmp_path):
    text = _page_text(_render(saas_result, tmp_path))
    for heading in (
        "PROBABILITY OF COMPLETING ROUND",
        "PROBABILITY DRIVERS",
        "CRITICAL BLOCKERS",
        "COMPARABLE FINANCING PROFILE",
        "FIVE ACTIONS TO INCREASE FUNDING PROBABILITY",
        "TEN CAPITAL ASSESSMENT",
        "FUNDABILITY",
        "CONFIDENCE",
    ):
        assert heading in text, f"missing section: {heading}"


def test_required_disclaimers_appear(saas_result, tmp_path):
    text = _page_text(_render(saas_result, tmp_path))
    assert "CONFIDENTIAL - FOR INTENDED RECIPIENT ONLY" in text
    assert "not guarantees of financing outcomes" in text
    assert "does not constitute investment, legal, tax, or securities advice" in text


def test_headline_numbers_are_printed(saas_result, tmp_path):
    text = _page_text(_render(saas_result, tmp_path))
    final = saas_result.prediction.final
    for value in (final.p3, final.p6, final.p12):
        assert f"{value:.0f}%" in text
    assert "3 MONTHS" in text and "6 MONTHS" in text and "12 MONTHS" in text


def test_market_condition_is_disclosed_as_an_assumption(saas_result, tmp_path):
    assert "MARKET CONDITION ASSUMPTION" in _page_text(_render(saas_result, tmp_path))


# ---------------------------------------------------------------------------
# Layout mechanics
# ---------------------------------------------------------------------------
def test_content_fits_within_the_measured_body_height(saas_result):
    renderer = FinancingReportRenderer(saas_result)
    spec = renderer.spec
    body_top = spec.height - spec.margin_top - 52.0 - 10
    body_height = body_top - (spec.margin_bottom + 30.0)

    density, flowables = renderer._fit(body_height)
    assert density in DENSITY_LADDER
    assert _measure(flowables, spec.content_width) <= body_height + 1.0


def test_density_ladder_is_monotonically_tighter():
    for previous, following in zip(DENSITY_LADDER, DENSITY_LADDER[1:]):
        assert following.body_size <= previous.body_size
        assert following.body_leading <= previous.body_leading
        assert following.action_chars <= previous.action_chars


def test_fit_text_never_exceeds_its_width():
    from reportlab.pdfbase.pdfmetrics import stringWidth

    long_text = "Northwind Analytics International Holdings Corporation Limited"
    fitted = _fit_text(long_text, "Helvetica-Bold", 17, 120)
    assert stringWidth(fitted, "Helvetica-Bold", 17) <= 120
    assert fitted.endswith("…")


def test_fit_text_leaves_short_text_alone():
    assert _fit_text("Acme", "Helvetica", 10, 500) == "Acme"


def test_clip_breaks_on_word_boundaries():
    clipped = _clip("documented customer commitments and signed contracts", 30)
    assert clipped.endswith("…")
    assert not clipped[:-1].endswith(" ")
    # The truncation must not land mid-word.
    assert clipped.rstrip("…").split()[-1] in "documented customer commitments".split()


def test_clip_returns_short_text_unchanged():
    assert _clip("short", 40) == "short"


# ---------------------------------------------------------------------------
# HTML companion
# ---------------------------------------------------------------------------
def test_html_report_is_written_and_self_describing(saas_result, tmp_path):
    path = render_html(saas_result, tmp_path / "report.html")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert saas_result.company_name in text
    assert "CONFIDENTIAL" in text
    assert "MARKET CONDITION ASSUMPTION" in text


def test_html_escapes_company_supplied_text(saas_result, tmp_path):
    injected = saas_result.model_copy(deep=True)
    injected.profile.put("company_name", "<script>alert(1)</script>", ValueStatus.EXPLICIT, 0.9)
    text = render_html(injected, tmp_path / "escaped.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text
