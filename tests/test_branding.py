"""The TEN Capital brand system in the PDF: fonts, palette and assets."""

from __future__ import annotations

import pytest
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics

from reports import fonts as fontmod
from reports.theme import (
    FONT_BOLD,
    FONT_DISPLAY,
    FONT_DISPLAY_HEAVY,
    FONT_MONO,
    FONT_REGULAR,
    PALETTE,
    blend,
)


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
def test_brand_fonts_are_bundled():
    """The statics must be committed; a deployment cannot fetch them."""
    missing = [name for name in fontmod.BRAND_FONTS.values()
               if not (fontmod.FONT_DIR / name).exists()]
    assert not missing, f"missing bundled fonts: {missing} (run assets/make_fonts.py)"


def test_brand_fonts_register_with_reportlab():
    resolved = fontmod.register_fonts()
    assert resolved["Inter"] == "Inter"
    assert resolved["Sora-Bold"] == "Sora-Bold"
    for name in resolved.values():
        assert pdfmetrics.getFont(name) is not None


def test_registration_is_idempotent():
    assert fontmod.register_fonts() is fontmod.register_fonts()


def test_weights_are_genuinely_different():
    """Variable fonts render at one weight; the statics must not.

    If instancing silently failed, Bold and Regular would have identical
    metrics and the design would lose its hierarchy without any error.
    """
    from reportlab.pdfbase.pdfmetrics import stringWidth

    sample = "Fundability 79/100"
    regular = stringWidth(sample, FONT_REGULAR, 12)
    bold = stringWidth(sample, FONT_BOLD, 12)
    assert regular != pytest.approx(bold, abs=0.01), "Inter Regular and SemiBold are identical"

    display = stringWidth(sample, FONT_DISPLAY, 12)
    heavy = stringWidth(sample, FONT_DISPLAY_HEAVY, 12)
    assert display != pytest.approx(heavy, abs=0.01), "Sora Bold and ExtraBold are identical"


def test_missing_fonts_fall_back_to_helvetica(monkeypatch, tmp_path):
    """A deployment that lost its assets still renders, just unbranded."""
    monkeypatch.setattr(fontmod, "FONT_DIR", tmp_path / "nowhere")
    monkeypatch.setattr(fontmod, "_resolved", None)
    resolved = fontmod.register_fonts()
    assert resolved["Inter"] == "Helvetica"
    assert resolved["Sora-Bold"] == "Helvetica-Bold"
    assert fontmod.using_brand_fonts() is False
    monkeypatch.setattr(fontmod, "_resolved", None)  # restore for other tests


def test_every_brand_font_has_a_fallback():
    assert set(fontmod.BRAND_FONTS) == set(fontmod.FALLBACKS)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
def test_report_surface_is_light():
    """The report is printed and forwarded, so it stays ink on paper."""
    page = PALETTE.page
    assert (page.red + page.green + page.blue) / 3 > 0.9
    assert PALETTE.navy.hexval()[2:].upper() == "14213D"


def test_brand_accents_are_available_for_the_lockup():
    """The tri-colour rule pulls its stops straight from the logo mark."""
    assert PALETTE.coral.hexval()[2:].upper() == "EE5A4E"
    assert PALETTE.amber.hexval()[2:].upper() == "F3A22A"
    assert PALETTE.teal.hexval()[2:].upper() == "35BEBB"


def test_semantic_colours_map_to_the_light_palette():
    assert PALETTE.for_probability(85) is PALETTE.strong
    assert PALETTE.for_probability(45) is PALETTE.moderate
    assert PALETTE.for_probability(10) is PALETTE.weak

    assert PALETTE.for_flag("YES") is PALETTE.strong
    assert PALETTE.for_flag("REVIEW") is PALETTE.moderate
    assert PALETTE.for_flag("NO") is PALETTE.weak

    assert PALETTE.for_severity("CRITICAL") is PALETTE.critical
    assert PALETTE.for_severity("MODERATE") is PALETTE.moderate


def test_tile_colours_carry_white_text():
    """Tiles are filled solid, so the figure on them must stay legible."""
    for value in (10.0, 45.0, 90.0):
        fill = PALETTE.for_probability(value)
        luminance = 0.299 * fill.red + 0.587 * fill.green + 0.114 * fill.blue
        assert luminance < 0.6, f"tile fill too light for white text at {value}%"


def test_blend_interpolates_and_clamps():
    black, white = colors.Color(0, 0, 0), colors.Color(1, 1, 1)
    assert blend(black, white, 0.0).red == pytest.approx(0.0)
    assert blend(black, white, 1.0).red == pytest.approx(1.0)
    assert blend(black, white, 0.5).red == pytest.approx(0.5)
    assert blend(black, white, 5.0).red == pytest.approx(1.0)   # clamped
    assert blend(black, white, -3.0).red == pytest.approx(0.0)  # clamped


def test_role_aliases_resolve_to_the_light_palette():
    assert PALETTE.page is PALETTE.paper
    assert PALETTE.band is PALETTE.navy
    assert PALETTE.heading is PALETTE.navy


# ---------------------------------------------------------------------------
# Logo assets
# ---------------------------------------------------------------------------
def test_both_logo_variants_are_available():
    """One for the navy header band, one for the white body."""
    from reports.pdf_report import LOGO_DARK, LOGO_LIGHT

    assert LOGO_DARK.exists(), "the supplied TEN Capital mark is missing"
    assert LOGO_LIGHT.exists(), "run assets/make_light_logo.py"


def test_light_logo_is_visible_on_navy():
    """The supplied mark is dark ink and would vanish on the header band."""
    from PIL import Image

    from reports.pdf_report import LOGO_LIGHT

    image = Image.open(LOGO_LIGHT).convert("RGBA")
    access = image.load()
    width, height = image.size
    bright = 0
    for y in range(0, height, 3):
        for x in range(0, width, 3):
            r, g, b, a = access[x, y]
            if a > 60 and (r + g + b) / 3 > 140:
                bright += 1
    assert bright > 500, "logo has too little light ink to read on navy"


def test_report_renders_with_the_brand_fonts(saas_result, tmp_path):
    from pypdf import PdfReader

    from reports.pdf_report import render_pdf

    path = render_pdf(saas_result, tmp_path / "brand.pdf")
    reader = PdfReader(str(path))
    assert len(reader.pages) == 1

    # The embedded BaseFont names prove the brand faces reached the document
    # rather than silently degrading to Helvetica. The entries are indirect
    # references, so they have to be resolved first.
    resources = reader.pages[0]["/Resources"]["/Font"]
    names = " ".join(
        str(resources[key].get_object().get("/BaseFont", "")) for key in resources
    )
    assert "Inter" in names, names
    assert "Sora" in names, names
    assert "JetBrainsMono" in names, names
