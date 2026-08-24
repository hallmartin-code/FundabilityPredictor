"""Visual design tokens for the PDF report.

Carries the TEN Capital brand system: dark navy surface, the tri-colour
coral/amber/teal accent from the logo mark, Sora for display, Inter for body
and JetBrains Mono for labels. The tokens match ``web/static/app.css`` so the
downloadable PDF and the web app read as one product.

Those three brand colours also carry the semantics - teal strong, amber
moderate, coral weak - which is why the probability tiles, blocker severities
and readiness flags need no second palette.

Kept apart from both the analysis and the layout code so the report can be
restyled without touching either. ``Density`` is the mechanism that keeps the
report on exactly one page: the layout retries with progressively tighter
densities until the content fits.
"""

from __future__ import annotations

from dataclasses import dataclass

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER

from reports.fonts import brand_font


def _hex(value: str) -> colors.Color:
    return colors.HexColor(value)


def blend(base: colors.Color, tint: colors.Color, amount: float) -> colors.Color:
    """Mix ``tint`` into ``base``.

    PDF fills have no compositing model as forgiving as CSS ``rgba()``, so the
    translucent tile backgrounds from the web design are pre-blended here
    against the surface they sit on.
    """
    amount = max(0.0, min(1.0, amount))
    return colors.Color(
        base.red + (tint.red - base.red) * amount,
        base.green + (tint.green - base.green) * amount,
        base.blue + (tint.blue - base.blue) * amount,
    )


@dataclass(frozen=True)
class Palette:
    """The light report palette.

    A printed or forwarded investor one-pager wants ink on paper, so the
    document surface stays white. The TEN Capital brand shows up where it
    reads best on light stock: the navy header band, the tri-colour accent
    rule from the logo mark, and the logo itself.
    """

    # Document surfaces
    ink: colors.Color = _hex("#101820")
    navy: colors.Color = _hex("#14213D")
    slate: colors.Color = _hex("#4A5568")
    muted: colors.Color = _hex("#7A8699")
    rule: colors.Color = _hex("#D5DAE2")
    panel: colors.Color = _hex("#F4F6F9")
    accent: colors.Color = _hex("#1F4E79")
    paper: colors.Color = colors.white

    # Semantic colours for tiles, severities and flags.
    strong: colors.Color = _hex("#1B7F4B")
    moderate: colors.Color = _hex("#B7791F")
    weak: colors.Color = _hex("#B03A2E")
    critical: colors.Color = _hex("#9B1C1C")

    # Brand accents, used for the logo lockup and the tri-colour rule.
    coral: colors.Color = _hex("#EE5A4E")
    amber: colors.Color = _hex("#F3A22A")
    teal: colors.Color = _hex("#35BEBB")

    # ---------------------------------------------------------- semantics
    @property
    def page(self) -> colors.Color:
        return self.paper

    @property
    def band(self) -> colors.Color:
        return self.navy

    @property
    def heading(self) -> colors.Color:
        return self.navy

    def for_probability(self, value: float) -> colors.Color:
        if value >= 60:
            return self.strong
        if value >= 30:
            return self.moderate
        return self.weak

    def for_severity(self, severity: str) -> colors.Color:
        return {
            "CRITICAL": self.critical,
            "HIGH": self.weak,
            "MODERATE": self.moderate,
        }.get(severity, self.slate)

    def for_flag(self, value: str) -> colors.Color:
        return {
            "YES": self.strong,
            "NO": self.weak,
            "REVIEW": self.moderate,
            "UNKNOWN": self.muted,
        }.get(value, self.muted)

    def for_confidence(self, level: str) -> colors.Color:
        return {
            "HIGH": self.strong,
            "MEDIUM": self.moderate,
        }.get(level, self.weak)


@dataclass(frozen=True)
class Density:
    """One rung of the shrink ladder used to force a single page."""

    name: str
    body_size: float
    body_leading: float
    small_size: float
    small_leading: float
    section_size: float
    gap: float
    max_drivers: int
    max_blockers: int
    max_actions: int
    action_chars: int
    blocker_chars: int
    driver_chars: int
    show_action_why: bool
    show_blocker_detail: bool
    #: The probability bridge and confidence notes are shown when there is
    #: room for them; they are the first thing dropped when there is not.
    show_bridge: bool


#: Tried in order. The first that fits on one page wins, so a sparse deck gets
#: the roomy layout and a dense one still lands on a single page.
DENSITY_LADDER: tuple[Density, ...] = (
    Density("comfortable", 7.6, 9.6, 6.8, 8.3, 8.0, 7.0, 5, 5, 5, 235, 165, 95, True, True, True),
    Density("standard", 7.2, 9.0, 6.5, 7.9, 7.7, 6.0, 5, 5, 5, 205, 145, 88, True, True, True),
    Density("tight", 6.9, 8.5, 6.2, 7.5, 7.4, 5.0, 5, 5, 5, 185, 135, 84, True, True, False),
    Density("compact", 6.6, 8.0, 5.9, 7.1, 7.1, 4.2, 4, 4, 5, 150, 108, 72, False, True, False),
    Density("dense", 6.3, 7.6, 5.7, 6.8, 6.9, 3.6, 3, 3, 5, 128, 92, 64, False, False, False),
)


@dataclass(frozen=True)
class PageSpec:
    width: float
    height: float
    margin_x: float
    margin_top: float
    margin_bottom: float

    @property
    def content_width(self) -> float:
        return self.width - 2 * self.margin_x

    @property
    def content_height(self) -> float:
        return self.height - self.margin_top - self.margin_bottom


def page_spec(name: str = "LETTER") -> PageSpec:
    size = A4 if name.upper() == "A4" else LETTER
    width, height = size
    return PageSpec(
        width=width,
        height=height,
        margin_x=34.0,
        margin_top=30.0,
        margin_bottom=34.0,
    )


PALETTE = Palette()

# ---------------------------------------------------------------------------
# Typography. Resolved lazily so a missing font file degrades to Helvetica
# rather than raising at import time.
# ---------------------------------------------------------------------------
FONT_DISPLAY = brand_font("Sora-Bold")
FONT_DISPLAY_HEAVY = brand_font("Sora-ExtraBold")
FONT_REGULAR = brand_font("Inter")
FONT_BOLD = brand_font("Inter-SemiBold")
FONT_MONO = brand_font("JetBrainsMono")
FONT_MONO_MEDIUM = brand_font("JetBrainsMono-Medium")

#: Inter has no bundled italic, so emphasis uses weight instead of slant.
FONT_OBLIQUE = FONT_REGULAR
