"""Visual design tokens for the PDF report.

Kept apart from both the analysis and the layout code so the report can be
restyled without touching either. ``Density`` is the mechanism that keeps the
report on exactly one page: the layout retries with progressively tighter
densities until the content fits.
"""

from __future__ import annotations

from dataclasses import dataclass

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER


@dataclass(frozen=True)
class Palette:
    ink: colors.Color = colors.HexColor("#101820")
    navy: colors.Color = colors.HexColor("#14213D")
    slate: colors.Color = colors.HexColor("#4A5568")
    muted: colors.Color = colors.HexColor("#7A8699")
    rule: colors.Color = colors.HexColor("#D5DAE2")
    panel: colors.Color = colors.HexColor("#F4F6F9")
    accent: colors.Color = colors.HexColor("#1F4E79")
    paper: colors.Color = colors.white

    # Semantic colours, used for probability tiles, severities and flags.
    strong: colors.Color = colors.HexColor("#1B7F4B")
    moderate: colors.Color = colors.HexColor("#B7791F")
    weak: colors.Color = colors.HexColor("#B03A2E")
    critical: colors.Color = colors.HexColor("#9B1C1C")

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
    Density("comfortable", 7.8, 9.8, 6.9, 8.4, 8.4, 7.0, 5, 5, 5, 235, 165, 95, True, True, True),
    Density("standard", 7.4, 9.2, 6.6, 8.0, 8.1, 6.0, 5, 5, 5, 205, 145, 88, True, True, True),
    Density("tight", 7.0, 8.6, 6.3, 7.6, 7.8, 5.0, 5, 5, 5, 185, 135, 84, True, True, False),
    Density("compact", 6.7, 8.1, 6.0, 7.2, 7.5, 4.2, 4, 4, 5, 150, 108, 72, False, True, False),
    Density("dense", 6.4, 7.7, 5.8, 6.9, 7.2, 3.6, 3, 3, 5, 128, 92, 64, False, False, False),
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

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_OBLIQUE = "Helvetica-Oblique"
