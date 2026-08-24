"""The one-page Financing Probability Report.

Staying on exactly one page is a hard requirement, so the renderer does not
trust a single layout pass. It builds the report at the roomiest density,
measures the result, and steps down the density ladder until the content fits.
A final ``KeepInFrame`` is the backstop that makes overflow impossible even if
an unexpectedly long input defeats every rung.

This module owns presentation only. It reads a finished
:class:`AnalysisResult` and never computes anything the reader could mistake
for analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    Flowable,
    Frame,
    KeepInFrame,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from config.settings import ASSETS_DIR
from extractors.normalization import format_money
from models.prediction import AnalysisResult
from reports.json_report import DISCLAIMER
from reports.theme import (
    DENSITY_LADDER,
    FONT_BOLD,
    FONT_OBLIQUE,
    FONT_REGULAR,
    PALETTE,
    Density,
    PageSpec,
    page_spec,
)

LOGGER = logging.getLogger(__name__)

LOGO_PATH = ASSETS_DIR / "TEN_Capital_logo_footer.png"

HEADER_HEIGHT = 52.0
FOOTER_HEIGHT = 30.0
NOT_PROVIDED = "Not Provided"


@dataclass(frozen=True)
class TemplateTokens:
    """Placeholder strings for the blank structure template.

    Numeric slots cannot carry placeholder text through the normal typed
    models, so the renderer substitutes these tokens when it is asked for a
    template. Everything else - headings, labels, flag names, disclaimers -
    is fixed structure and renders exactly as it does in a real report, which
    is what makes the template a faithful specification of the output.
    """

    percent: str = "XX%"
    score: str = "XX"
    money: str = "$X.XM"
    delta: str = "+X.X%"
    count: str = "N"


# ---------------------------------------------------------------------------
# Custom flowables
# ---------------------------------------------------------------------------
class ProbabilityTiles(Flowable):
    """The three headline probability tiles, drawn as one unit."""

    def __init__(
        self,
        width: float,
        values: list[tuple[str, float]],
        height: float = 54.0,
        token: str | None = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.values = values
        self.token = token

    def wrap(self, availWidth: float, availHeight: float) -> tuple[float, float]:  # noqa: N803
        self.width = availWidth
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        gap = 7.0
        count = len(self.values)
        tile_width = (self.width - gap * (count - 1)) / count
        for index, (label, value) in enumerate(self.values):
            x = index * (tile_width + gap)
            colour = PALETTE.for_probability(value)
            canvas.setFillColor(colour)
            canvas.roundRect(x, 0, tile_width, self.height, 3.5, stroke=0, fill=1)

            canvas.setFillColor(colors.white)
            canvas.setFont(FONT_BOLD, 21)
            figure = self.token or f"{value:.0f}%"
            canvas.drawCentredString(x + tile_width / 2, self.height - 27.5, figure)

            canvas.setFont(FONT_BOLD, 6.6)
            canvas.drawCentredString(x + tile_width / 2, 9.0, label.upper())


class ScoreBadge(Flowable):
    """Readiness score and confidence, set beside the probability tiles."""

    def __init__(
        self,
        width: float,
        readiness: float,
        readiness_label: str,
        confidence: float,
        confidence_level: str,
        height: float = 54.0,
        tokens: TemplateTokens | None = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.readiness = readiness
        self.readiness_label = readiness_label
        self.confidence = confidence
        self.confidence_level = confidence_level
        self.tokens = tokens

    def wrap(self, availWidth: float, availHeight: float) -> tuple[float, float]:  # noqa: N803
        self.width = availWidth
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        canvas.setFillColor(PALETTE.panel)
        canvas.setStrokeColor(PALETTE.rule)
        canvas.setLineWidth(0.6)
        canvas.roundRect(0, 0, self.width, self.height, 3.5, stroke=1, fill=1)

        canvas.setFillColor(PALETTE.muted)
        canvas.setFont(FONT_BOLD, 6.0)
        canvas.drawString(8, self.height - 12, "FINANCING READINESS")

        score = self.tokens.score if self.tokens else f"{self.readiness:.0f}"
        canvas.setFillColor(PALETTE.navy)
        canvas.setFont(FONT_BOLD, 15)
        canvas.drawString(8, self.height - 28, score)
        canvas.setFont(FONT_REGULAR, 8)
        canvas.setFillColor(PALETTE.slate)
        canvas.drawString(8 + stringWidth(score, FONT_BOLD, 15) + 2,
                          self.height - 28, "/100")

        canvas.setFont(FONT_BOLD, 6.4)
        canvas.setFillColor(PALETTE.accent)
        canvas.drawString(8, self.height - 38, _fit_text(self.readiness_label, FONT_BOLD, 6.4, self.width - 16))

        canvas.setStrokeColor(PALETTE.rule)
        canvas.line(8, 15.5, self.width - 8, 15.5)

        canvas.setFillColor(PALETTE.muted)
        canvas.setFont(FONT_BOLD, 6.0)
        canvas.drawString(8, 6.0, "CONFIDENCE")
        level_colour = {
            "HIGH": PALETTE.strong,
            "MEDIUM": PALETTE.moderate,
        }.get(self.confidence_level, PALETTE.weak)
        canvas.setFillColor(level_colour)
        canvas.setFont(FONT_BOLD, 7.4)
        figure = self.tokens.score if self.tokens else f"{self.confidence:.0f}"
        canvas.drawRightString(self.width - 8, 6.0, f"{self.confidence_level} ({figure})")


class FlagGrid(Flowable):
    """TEN Capital readiness flags in a compact two-column grid."""

    def __init__(self, width: float, flags: list[tuple[str, str]], font_size: float) -> None:
        super().__init__()
        self.width = width
        self.flags = flags
        self.font_size = font_size
        self.row_height = font_size + 3.4
        self.rows = (len(flags) + 1) // 2
        self.height = self.rows * self.row_height

    def wrap(self, availWidth: float, availHeight: float) -> tuple[float, float]:  # noqa: N803
        self.width = availWidth
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        column_width = self.width / 2
        for index, (label, value) in enumerate(self.flags):
            column, row = index % 2, index // 2
            x = column * column_width
            y = self.height - (row + 1) * self.row_height + 2.0

            canvas.setFillColor(PALETTE.slate)
            canvas.setFont(FONT_REGULAR, self.font_size)
            value_width = stringWidth(value, FONT_BOLD, self.font_size)
            available = column_width - value_width - 10
            canvas.drawString(x, y, _fit_text(label, FONT_REGULAR, self.font_size, available))

            canvas.setFillColor(PALETTE.for_flag(value))
            canvas.setFont(FONT_BOLD, self.font_size)
            canvas.drawRightString(x + column_width - 6, y, value)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class FinancingReportRenderer:
    """Builds and writes the one-page report."""

    def __init__(
        self,
        result: AnalysisResult,
        page: str = "LETTER",
        tokens: TemplateTokens | None = None,
    ) -> None:
        self.result = result
        self.spec: PageSpec = page_spec(page)
        #: Set only when rendering the blank structure template.
        self.tokens = tokens

    def _num(self, formatted: str, token: str) -> str:
        """Real value normally; a placeholder token in template mode."""
        return token if self.tokens else formatted

    # -------------------------------------------------------------- public
    def render(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        body_top = self.spec.height - self.spec.margin_top - HEADER_HEIGHT - 10
        body_height = body_top - (self.spec.margin_bottom + FOOTER_HEIGHT)

        chosen, flowables = self._fit(body_height)
        LOGGER.info("Rendering at '%s' density.", chosen.name)

        canvas = pdfcanvas.Canvas(str(path), pagesize=(self.spec.width, self.spec.height))
        canvas.setTitle(f"{self.result.company_name} - Financing Probability Report")
        canvas.setAuthor("TEN Capital Network")
        canvas.setSubject("Startup Financing Probability Report")

        self._draw_header(canvas)
        self._draw_footer(canvas)

        # KeepInFrame is the guarantee: even if every density rung overflowed,
        # the content is scaled to fit rather than spilling onto page two.
        block = KeepInFrame(
            self.spec.content_width,
            body_height,
            flowables,
            mode="shrink",
            hAlign="LEFT",
        )
        frame = Frame(
            self.spec.margin_x,
            body_top - body_height,
            self.spec.content_width,
            body_height,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            showBoundary=0,
        )
        remaining = [block]
        frame.addFromList(remaining, canvas)
        if remaining:  # pragma: no cover - KeepInFrame should always consume it
            LOGGER.error("Report content did not fit the single-page frame.")

        canvas.showPage()
        canvas.save()
        LOGGER.info("Wrote %s", path)
        return path

    # ------------------------------------------------------------- fitting
    def _fit(self, available_height: float) -> tuple[Density, list]:
        """Step down the density ladder until the content fits the page."""
        last: tuple[Density, list] | None = None
        for density in DENSITY_LADDER:
            flowables = self._build_body(density)
            height = _measure(flowables, self.spec.content_width)
            last = (density, flowables)
            if height <= available_height:
                return density, flowables
            LOGGER.debug(
                "Density '%s' needs %.1fpt of %.1fpt; trying tighter.",
                density.name, height, available_height,
            )
        LOGGER.warning(
            "Content exceeds the page at every density; KeepInFrame will scale it to fit."
        )
        assert last is not None
        return last

    # -------------------------------------------------------------- header
    def _draw_header(self, canvas: pdfcanvas.Canvas) -> None:
        spec = self.spec
        top = spec.height - spec.margin_top
        canvas.setFillColor(PALETTE.navy)
        canvas.rect(spec.margin_x, top - HEADER_HEIGHT, spec.content_width, HEADER_HEIGHT,
                    stroke=0, fill=1)

        pad = 11.0
        right_label = "FINANCING PROBABILITY REPORT"
        canvas.setFont(FONT_BOLD, 6.8)
        canvas.setFillColor(colors.HexColor("#9FB3D1"))
        canvas.drawRightString(spec.margin_x + spec.content_width - pad, top - 15, right_label)

        # The company name is auto-sized so a long one cannot collide with the
        # label on its right or run off the band.
        name = self.result.company_name or NOT_PROVIDED
        max_name_width = (
            spec.content_width - 2 * pad - stringWidth(right_label, FONT_BOLD, 6.8) - 14
        )
        size = 17.0
        while size > 9.0 and stringWidth(name, FONT_BOLD, size) > max_name_width:
            size -= 0.5
        canvas.setFillColor(colors.white)
        canvas.setFont(FONT_BOLD, size)
        canvas.drawString(spec.margin_x + pad, top - 20,
                          _fit_text(name, FONT_BOLD, size, max_name_width))

        canvas.setFont(FONT_REGULAR, 7.2)
        canvas.setFillColor(colors.HexColor("#C6D3E6"))
        canvas.drawString(spec.margin_x + pad, top - 32, self._identity_line())

        canvas.setFont(FONT_BOLD, 7.2)
        canvas.setFillColor(colors.white)
        canvas.drawString(spec.margin_x + pad, top - 43.5, self._financing_line())

    def _identity_line(self) -> str:
        profile = self.result.profile
        stage = self.result.stage
        if self.tokens:
            stage_text = "[STAGE] (declared [DECLARED STAGE])"
        else:
            stage_text = stage.inferred.value if stage.inferred.is_known else NOT_PROVIDED
            if stage.declared.is_known and stage.declared is not stage.inferred:
                stage_text = f"{stage_text} (declared {stage.declared.value})"
        geography = (
            profile.text("headquarters")
            or profile.text("primary_geography")
            or profile.text("geographic_market")
            or profile.text("country")
            or NOT_PROVIDED
        )
        sector = self.result.sector.label if self.result.sector.key != "unknown" else NOT_PROVIDED
        return _fit_text(
            f"{sector}   |   {stage_text}   |   {geography}",
            FONT_REGULAR, 7.2, self.spec.content_width - 24,
        )

    def _financing_line(self) -> str:
        profile = self.result.profile
        target = profile.number("target_raise")
        committed = profile.number("amount_committed")
        pct = profile.number("percent_committed")
        valuation = (
            profile.number("pre_money_valuation")
            or profile.number("note_cap")
            or profile.number("safe_cap")
        )
        if self.tokens:
            committed_text = f"{self.tokens.money} ({self.tokens.percent})"
        else:
            committed_text = format_money(committed) if committed is not None else NOT_PROVIDED
            if pct is not None:
                committed_text = (
                    f"{committed_text} ({pct:.0f}%)" if committed is not None else f"{pct:.0f}%"
                )
        parts = [
            "Target Raise "
            + self._num(
                format_money(target) if target is not None else NOT_PROVIDED, self.tokens.money
                if self.tokens else ""
            ),
            f"Committed {committed_text}",
            "Valuation "
            + self._num(
                format_money(valuation) if valuation is not None else NOT_PROVIDED,
                self.tokens.money if self.tokens else "",
            ),
        ]
        return _fit_text("   ·   ".join(parts), FONT_BOLD, 7.2, self.spec.content_width - 24)

    # -------------------------------------------------------------- footer
    def _draw_footer(self, canvas: pdfcanvas.Canvas) -> None:
        spec = self.spec
        y = spec.margin_bottom + FOOTER_HEIGHT
        canvas.setStrokeColor(PALETTE.rule)
        canvas.setLineWidth(0.6)
        canvas.line(spec.margin_x, y, spec.margin_x + spec.content_width, y)

        canvas.setFillColor(PALETTE.muted)
        canvas.setFont(FONT_REGULAR, 5.6)
        note = (
            "Financing probability estimates are model-based assessments, not guarantees of "
            "financing outcomes.   ·   " + DISCLAIMER
        )
        for index, line in enumerate(_wrap_text(note, FONT_REGULAR, 5.6, spec.content_width - 86)):
            canvas.drawString(spec.margin_x, y - 8.5 - index * 6.4, line)

        canvas.setFont(FONT_BOLD, 6.0)
        canvas.setFillColor(PALETTE.critical)
        canvas.drawString(spec.margin_x, spec.margin_bottom - 2, "CONFIDENTIAL - FOR INTENDED RECIPIENT ONLY")

        canvas.setFont(FONT_REGULAR, 5.6)
        canvas.setFillColor(PALETTE.muted)
        stamp = (
            "Compiled on "
            + self._num(self.result.generated_at.strftime("%d %B %Y"), "[DATE]")
            + " by TEN Capital Network"
        )
        logo_width = 46.0
        canvas.drawRightString(
            spec.margin_x + spec.content_width - logo_width - 6, spec.margin_bottom - 2, stamp
        )
        self._draw_logo(canvas, spec.margin_x + spec.content_width - logo_width,
                        spec.margin_bottom - 7, logo_width)

    @staticmethod
    def _draw_logo(canvas: pdfcanvas.Canvas, x: float, y: float, width: float) -> None:
        if not LOGO_PATH.exists():
            return
        try:
            image = ImageReader(str(LOGO_PATH))
            iw, ih = image.getSize()
            height = width * ih / iw
            canvas.drawImage(image, x, y, width=width, height=height, mask="auto")
        except Exception:  # noqa: BLE001 - a missing logo must not fail the report
            LOGGER.debug("Could not draw the footer logo.", exc_info=True)

    # ---------------------------------------------------------------- body
    def _build_body(self, density: Density) -> list:
        styles = _styles(density)
        left = self._left_column(density, styles)
        right = self._right_column(density, styles)

        gutter = 15.0
        column_width = (self.spec.content_width - gutter) / 2

        prediction_block = self._prediction_block(density, styles)

        columns = Table(
            [[left, right]],
            colWidths=[column_width, column_width],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (0, 0), gutter),
                    ("RIGHTPADDING", (1, 0), (1, 0), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            ),
        )
        body = prediction_block + [Spacer(1, density.gap + 2), columns]
        if density.show_bridge:
            body += [Spacer(1, density.gap + 3), self._transparency_block(density, styles, gutter)]
        return body

    def _transparency_block(self, density: Density, styles: dict, gutter: float) -> Table:
        """How the probability was reached, and what limits confidence in it.

        Shown only when the page has room. It answers the question a founder
        asks first - "why this number?" - without which the headline
        percentages read as an oracle.
        """
        column_width = (self.spec.content_width - gutter) / 2

        left: list = [
            Paragraph("HOW THE 6-MONTH PROBABILITY WAS REACHED", styles["section"]),
            Spacer(1, 2.0),
        ]
        for step in self.result.prediction.bridge[:9]:
            if step.kind == "baseline":
                score = self._num(
                    f"{self.result.readiness.total:.0f}", self.tokens.score if self.tokens else ""
                )
                value = self._num(
                    f"{step.running.get('p6', 0):.0f}%", self.tokens.percent if self.tokens else ""
                )
                text = f"<b>Baseline from readiness score {score}/100: {value}</b>"
            else:
                delta = step.delta.get("p6", 0.0)
                colour = PALETTE.strong if delta >= 0 else PALETTE.weak
                sign = "+" if delta >= 0 else "−"
                # The template keeps the sign so both directions read correctly.
                rendered = self._num(
                    f"{sign}{abs(delta):.1f}%",
                    f"{sign}X.X%" if self.tokens else "",
                )
                text = (
                    f'<font color="{_hex(colour)}"><b>{rendered}</b></font>  '
                    f"{_clip(step.label, 74)}"
                )
            left.append(Paragraph(text, styles["small"]))
        left.append(Spacer(1, 1.5))
        final = self._num(
            f"{self.result.prediction.final.p6:.0f}%", self.tokens.percent if self.tokens else ""
        )
        left.append(Paragraph(f"<b>Final 6-month probability: {final}</b>", styles["body"]))

        right: list = [
            Paragraph("WHAT LIMITS CONFIDENCE", styles["section"]),
            Spacer(1, 2.0),
        ]
        notes = self.result.confidence.notes[:5]
        if notes:
            for note in notes:
                right.append(Paragraph(f"▪ {_clip(note, 150)}", styles["small"]))
        else:
            right.append(
                Paragraph("The deck disclosed every probability-critical field.", styles["small"])
            )

        missing = self.result.confidence.missing_critical_fields
        if missing:
            right.append(Spacer(1, 2.0))
            right.append(
                Paragraph(
                    f"<b>{self._num(str(len(missing)), self.tokens.count if self.tokens else '')}</b>"
                    " of the model's critical fields were not provided; "
                    "values shown as Not Provided were never inferred.",
                    styles["muted"],
                )
            )

        return Table(
            [[left, right]],
            colWidths=[column_width, column_width],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (0, 0), gutter),
                    ("RIGHTPADDING", (1, 0), (1, 0), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("LINEABOVE", (0, 0), (-1, 0), 0.6, PALETTE.rule),
                ]
            ),
        )

    def _prediction_block(self, density: Density, styles: dict) -> list:
        final = self.result.prediction.final
        tiles = ProbabilityTiles(
            0,
            [("3 Months", final.p3), ("6 Months", final.p6), ("12 Months", final.p12)],
            token=self.tokens.percent if self.tokens else None,
        )
        badge = ScoreBadge(
            0,
            self.result.readiness.total,
            self.result.readiness.label,
            self.result.confidence.score,
            self.result.confidence.level,
            tokens=self.tokens,
        )
        tile_width = self.spec.content_width * 0.635
        row = Table(
            [[tiles, badge]],
            colWidths=[tile_width, self.spec.content_width - tile_width],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (0, 0), 9),
                    ("RIGHTPADDING", (1, 0), (1, 0), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            ),
        )
        return [
            Paragraph("PROBABILITY OF COMPLETING ROUND", styles["section"]),
            Spacer(1, 3.0),
            row,
        ]

    def _left_column(self, density: Density, styles: dict) -> list:
        out: list = [Paragraph("PROBABILITY DRIVERS", styles["section"]), Spacer(1, 2.0)]

        drivers = self.result.prediction.positive_signals[: density.max_drivers]
        if drivers:
            for driver in drivers:
                out.append(Paragraph(f"▪ {_clip(driver, density.driver_chars)}", styles["bullet"]))
        else:
            out.append(
                Paragraph(
                    "No positive financing signals were evidenced in the deck.", styles["muted"]
                )
            )

        out += [Spacer(1, density.gap), Paragraph("CRITICAL BLOCKERS", styles["section"]), Spacer(1, 2.0)]
        blockers = self.result.blockers[: density.max_blockers]
        if blockers:
            for blocker in blockers:
                colour = PALETTE.for_severity(blocker.severity)
                text = (
                    f'<font color="{_hex(colour)}"><b>{blocker.severity}</b></font>  '
                    f"<b>{blocker.title}</b>"
                )
                if density.show_blocker_detail:
                    text += f"<br/>{_clip(blocker.detail, density.blocker_chars)}"
                out.append(Paragraph(text, styles["bullet"]))
        else:
            out.append(Paragraph("No structural blockers identified.", styles["muted"]))

        out += [
            Spacer(1, density.gap),
            Paragraph("COMPARABLE FINANCING PROFILE", styles["section"]),
            Spacer(1, 2.0),
            Paragraph(f"<b>{self.result.comparable.label}</b>", styles["body"]),
            Paragraph(_clip(self.result.comparable.rationale, 210), styles["small"]),
            Spacer(1, 2.5),
            Paragraph(self.result.prediction.market_condition.disclosure, styles["muted"]),
        ]
        return out

    def _right_column(self, density: Density, styles: dict) -> list:
        out: list = [
            Paragraph("FIVE ACTIONS TO INCREASE FUNDING PROBABILITY", styles["section"]),
            Spacer(1, 2.0),
        ]
        for index, action in enumerate(self.result.actions[: density.max_actions], start=1):
            text = f"<b>{index}.</b> {_clip(action.action, density.action_chars)}"
            text += (
                f'  <font color="{_hex(PALETTE.muted)}">'
                f"[{action.impact} · {action.timeframe}]</font>"
            )
            if density.show_action_why:
                text += (
                    f'<br/><font color="{_hex(PALETTE.slate)}">'
                    f"{_clip(action.why, 120)}</font>"
                )
            out.append(Paragraph(text, styles["bullet"]))

        ten = self.result.ten_capital
        out += [
            Spacer(1, density.gap),
            Paragraph("TEN CAPITAL ASSESSMENT", styles["section"]),
            Spacer(1, 2.0),
            Paragraph(
                f"<b>Fit Score {self._num(f'{ten.fit_score:.0f}', self.tokens.score if self.tokens else '')}/100</b> &nbsp;·&nbsp; "
                f"<b>{_clip(ten.recommended_next_step, 60)}</b>",
                styles["body"],
            ),
            Paragraph(_clip(ten.explanation, 165), styles["small"]),
            Spacer(1, 3.0),
            FlagGrid(
                0,
                [(flag.label, flag.value) for flag in ten.flags],
                density.small_size,
            ),
        ]
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _styles(density: Density) -> dict[str, ParagraphStyle]:
    return {
        "section": ParagraphStyle(
            "section",
            fontName=FONT_BOLD,
            fontSize=density.section_size,
            leading=density.section_size + 1.6,
            textColor=PALETTE.navy,
            spaceAfter=0,
            alignment=TA_LEFT,
        ),
        "body": ParagraphStyle(
            "body",
            fontName=FONT_REGULAR,
            fontSize=density.body_size,
            leading=density.body_leading,
            textColor=PALETTE.ink,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName=FONT_REGULAR,
            fontSize=density.body_size,
            leading=density.body_leading,
            textColor=PALETTE.ink,
            spaceAfter=2.6,
        ),
        "small": ParagraphStyle(
            "small",
            fontName=FONT_REGULAR,
            fontSize=density.small_size,
            leading=density.small_leading,
            textColor=PALETTE.slate,
        ),
        "muted": ParagraphStyle(
            "muted",
            fontName=FONT_OBLIQUE,
            fontSize=density.small_size,
            leading=density.small_leading,
            textColor=PALETTE.muted,
        ),
        "center": ParagraphStyle(
            "center",
            fontName=FONT_REGULAR,
            fontSize=density.small_size,
            leading=density.small_leading,
            textColor=PALETTE.muted,
            alignment=TA_CENTER,
        ),
    }


def _measure(flowables: list, width: float) -> float:
    """Total height the flowables need at the given width."""
    total = 0.0
    for flowable in flowables:
        _, height = flowable.wrap(width, 1_000_000)
        total += height
        total += getattr(flowable, "getSpaceBefore", lambda: 0)()
        total += getattr(flowable, "getSpaceAfter", lambda: 0)()
    return total


def _hex(colour: colors.Color) -> str:
    """CSS-style hex for inline <font color="..."> markup."""
    return "#" + colour.hexval()[2:]


def _clip(text: str, limit: int) -> str:
    """Truncate on a word boundary so the report never cuts mid-word."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    window = text[: max(0, limit - 1)]
    space = window.rfind(" ")
    # Only fall back to a hard cut if the last word is implausibly long.
    if space > limit * 0.6:
        window = window[:space]
    return window.rstrip(" ,;:.-") + "…"


def _fit_text(text: str, font: str, size: float, max_width: float) -> str:
    """Truncate to fit a fixed width, so nothing can overrun its box."""
    if max_width <= 0:
        return ""
    if stringWidth(text, font, size) <= max_width:
        return text
    ellipsis = "…"
    trimmed = text
    while trimmed and stringWidth(trimmed + ellipsis, font, size) > max_width:
        trimmed = trimmed[:-1]
    return (trimmed.rstrip() + ellipsis) if trimmed else ""


def _wrap_text(text: str, font: str, size: float, max_width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:2]


def render_pdf(
    result: AnalysisResult,
    path: Path,
    page: str = "LETTER",
    tokens: TemplateTokens | None = None,
) -> Path:
    """Render the one-page report for ``result`` to ``path``.

    Passing ``tokens`` renders the blank structure template instead of a real
    report, using this same layout code so the template can never drift from
    the output it documents.
    """
    return FinancingReportRenderer(result, page=page, tokens=tokens).render(path)
