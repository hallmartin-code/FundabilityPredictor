"""Generate the blank document-structure template.

    python docs/generate_template.py

Writes into ``docs/``:

    report_template.pdf            the one-page layout with placeholder tokens
    report_template.png            a preview image of the same
    report_template.html           the HTML companion layout
    lead_record_template.json      the lead-record structure, values nulled
    evidence_map_template.json     the evidence-map structure, values nulled

The PDF is produced by the *real* renderer with placeholder tokens rather than
by a separate mock, so the template cannot drift from the document the
application actually produces. Re-run this whenever the layout or the field
registry changes.

Nothing here contains company data. Every value is either a placeholder token,
a bracketed field description, or fixed structural text (headings, flag
labels, disclaimers) that appears verbatim in every report.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.company import SectorProfile, Stage, StageAssessment  # noqa: E402
from models.evidence import Inconsistency, ValueStatus  # noqa: E402
from models.fields import FIELD_SPECS  # noqa: E402
from models.prediction import (  # noqa: E402
    Action,
    AnalysisResult,
    Blocker,
    BridgeStep,
    CategoryScore,
    ComparableProfile,
    ConfidenceResult,
    MarketCondition,
    PredictionResult,
    Probability,
    ReadinessScore,
    SignalScore,
    TenCapitalAssessment,
    TenCapitalFlag,
)
from models.profile import DeckProfile, SourceDocument  # noqa: E402
from reports.html_report import render_html  # noqa: E402
from reports.json_report import build_evidence_map, build_lead_record  # noqa: E402
from reports.pdf_report import TemplateTokens, render_pdf  # noqa: E402

DOCS = Path(__file__).resolve().parent

#: Field descriptions used in place of company data. Square brackets mark a
#: slot the application fills; anything outside them is fixed text.
PLACEHOLDER = "[{}]"


def _p(text: str) -> str:
    return PLACEHOLDER.format(text)


def _template_profile() -> DeckProfile:
    """A profile whose every value describes the field rather than a company."""
    profile = DeckProfile(
        source=SourceDocument(
            path=_p("SOURCE FILE PATH"),
            filename=_p("SOURCE DECK FILENAME"),
            file_type="pdf | pptx | ppt",
            page_count=0,
            sha256=_p("SHA-256 OF THE SOURCE FILE"),
        )
    )
    profile.put(
        "company_name", _p("COMPANY NAME"), ValueStatus.EXPLICIT, 0.0,
        source_page=None, source_text=None,
    )
    profile.put(
        "headquarters", _p("GEOGRAPHY"), ValueStatus.EXPLICIT, 0.0,
        source_page=None, source_text=None,
    )
    return profile


def _template_readiness() -> ReadinessScore:
    """The nine categories, in the fixed order the report always uses."""
    from config.settings import load_config

    cfg = load_config()
    exemplar = [
        SignalScore(
            key=_p("signal key"),
            label=_p("Signal label"),
            weight=0.0,
            awarded=0.0,
            evidenced=False,
            detail=_p("What the signal found, or why it could not be evaluated"),
        )
    ]
    return ReadinessScore(
        total=0.0,
        label=_p("READINESS LABEL"),
        categories=[
            CategoryScore(
                key=key,
                label=str(meta["label"]),
                weight=float(meta["weight"]),
                points=0.0,
                # Only the first category carries an exemplar signal; the
                # nested shape is identical for all nine.
                signals=exemplar if index == 0 else [],
            )
            for index, (key, meta) in enumerate(cfg.categories.items())
        ],
    )


def _template_prediction() -> PredictionResult:
    zero = Probability(p3=0.0, p6=0.0, p12=0.0)
    bridge = [
        BridgeStep(
            label="Baseline from readiness score",
            kind="baseline",
            delta={"p3": 0.0, "p6": 0.0, "p12": 0.0},
            running={"p3": 0.0, "p6": 0.0, "p12": 0.0},
        )
    ]
    # One positive and one negative row so both colour treatments are visible.
    bridge += [
        BridgeStep(
            label=_p("Positive signal detected in the deck"),
            kind="signal",
            delta={"p3": 1.0, "p6": 1.0, "p12": 1.0},
            running={"p3": 0.0, "p6": 0.0, "p12": 0.0},
        ),
        BridgeStep(
            label=_p("Negative signal detected in the deck"),
            kind="signal",
            delta={"p3": -1.0, "p6": -1.0, "p12": -1.0},
            running={"p3": 0.0, "p6": 0.0, "p12": 0.0},
        ),
        BridgeStep(
            label=_p("SECTOR") + " sector financing dynamics",
            kind="sector",
            delta={"p3": -1.0, "p6": -1.0, "p12": -1.0},
            running={"p3": 0.0, "p6": 0.0, "p12": 0.0},
        ),
        BridgeStep(
            label="Capital market environment: " + _p("ENVIRONMENT"),
            kind="market",
            delta={"p3": 1.0, "p6": 1.0, "p12": 1.0},
            running={"p3": 0.0, "p6": 0.0, "p12": 0.0},
        ),
    ]
    return PredictionResult(
        baseline=zero,
        final=zero,
        bridge=bridge,
        positive_signals=[_p(f"Probability driver {n} of up to 5") for n in range(1, 6)],
        negative_signals=[],
        market_condition=MarketCondition(
            key="neutral", label=_p("ENVIRONMENT"), is_verified=False
        ),
        sector_modifier={"p3": 1.0, "p6": 1.0, "p12": 1.0},
    )


def _template_flags() -> list[TenCapitalFlag]:
    """The ten readiness flags, with their fixed labels and value domain."""
    labels = [
        ("raise_over_2m", "Raise >= $2M"),
        ("twenty_five_percent_committed", ">= 25% committed"),
        ("fifty_percent_committed", ">= 50% committed"),
        ("revenue_over_700k", "Revenue >= $700K"),
        ("revenue_generating", "Revenue generating"),
        ("institutional_round", "Institutional round"),
        ("lead_investor", "Lead investor identified"),
        ("runway_over_9_months", "Runway >= 9 months"),
        ("materials_complete", "Investor materials complete"),
        ("likely_fundraising_candidate", "Likely fundraising candidate"),
    ]
    # Cycle the value domain so every colour treatment appears in the template.
    domain = ["YES", "NO", "UNKNOWN", "REVIEW"]
    return [
        TenCapitalFlag(key=key, label=label, value=domain[index % len(domain)])
        for index, (key, label) in enumerate(labels)
    ]


def build_template_result() -> AnalysisResult:
    """An AnalysisResult containing only field descriptions and placeholders."""
    return AnalysisResult(
        profile=_template_profile(),
        stage=StageAssessment(
            declared=Stage.UNKNOWN,
            inferred=Stage.UNKNOWN,
            confidence=_p("HIGH | MEDIUM | LOW"),
            rationale=[],
            signals_used=0,
        ),
        sector=SectorProfile(key="template", label=_p("SECTOR")),
        readiness=_template_readiness(),
        prediction=_template_prediction(),
        confidence=ConfidenceResult(
            score=0.0,
            level=_p("LEVEL"),
            components={
                "critical_field_completeness": 0.0,
                "evidence_quality": 0.0,
                "extraction_quality": 0.0,
                "consistency": 0.0,
            },
            missing_critical_fields=[_p("missing critical field")],
            notes=[
                _p(f"Confidence note {n}: what a specific missing input cost")
                for n in range(1, 6)
            ],
        ),
        blockers=[
            Blocker(
                key=f"blocker_{n}",
                severity=severity,
                title=_p(f"Blocker {n} title"),
                detail=_p(
                    f"Blocker {n} detail: one or two sentences naming the issue and quoting "
                    "the company's own figures"
                ),
            )
            for n, severity in enumerate(
                ["CRITICAL", "HIGH", "HIGH", "MODERATE", "MODERATE"], start=1
            )
        ],
        actions=[
            Action(
                action=_p(
                    f"Action {n}: a specific instruction naming the company's own numbers, "
                    "milestones or counterparties"
                ),
                why=_p(f"Why action {n} matters to closing this round"),
                impact=impact,
                timeframe=timeframe,
            )
            for n, (impact, timeframe) in enumerate(
                [
                    ("HIGH", "Immediate"),
                    ("HIGH", "30 Days"),
                    ("HIGH", "60 Days"),
                    ("MEDIUM", "60 Days"),
                    ("MEDIUM", "90 Days"),
                ],
                start=1,
            )
        ],
        comparable=ComparableProfile(
            key="template",
            label=_p("COMPARABLE FINANCING PROFILE"),
            rationale=_p(
                "One or two sentences explaining why the company fits this profile"
            ),
        ),
        ten_capital=TenCapitalAssessment(
            fit_score=0.0,
            flags=_template_flags(),
            recommended_next_step=_p("RECOMMENDED NEXT STEP"),
            explanation=_p("One sentence explaining the recommended next step"),
        ),
        inconsistencies=[
            Inconsistency(
                code=_p("inconsistency code"),
                severity=_p("REVIEW | HIGH"),
                message=_p("What contradicts what, quoting both values"),
                fields=[_p("field name")],
            )
        ],
        generated_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
    )


def _null_json(value):
    """Recursively blank a JSON structure, keeping keys and list shapes."""
    if isinstance(value, dict):
        return {key: _null_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_null_json(value[0])] if value else []
    return None


def main() -> int:
    import json

    result = build_template_result()
    tokens = TemplateTokens()

    pdf_path = render_pdf(result, DOCS / "report_template.pdf", tokens=tokens)
    html_path = render_html(result, DOCS / "report_template.html")

    real = build_lead_record(result)
    lead = _null_json(real)
    # These are fixed structure, not company data, so they stay verbatim.
    lead["schema_version"] = real["schema_version"]
    lead["disclaimer"] = real["disclaimer"]
    lead_path = DOCS / "lead_record_template.json"
    lead_path.write_text(json.dumps(lead, indent=2) + "\n", encoding="utf-8")

    # The evidence map lists every registry field, so the template keeps the
    # full field list rather than a single blanked example.
    evidence = build_evidence_map(_template_profile())
    evidence["fields"] = [
        {
            "field": spec.name,
            "label": spec.label,
            "category": spec.category.value,
            "critical": spec.critical,
            "value": None,
            "status": "EXPLICIT | CALCULATED | ESTIMATED | MISSING",
            "confidence": None,
            "source_page": None,
            "source_text": None,
        }
        for spec in FIELD_SPECS
    ]
    evidence["source_file"] = None
    evidence["source_sha256"] = None
    evidence["pages"] = None
    # Counts describe a specific document; the template shows only the shape.
    evidence["status_counts"] = {key: None for key in evidence["status_counts"]}
    evidence["extraction_quality"] = {
        key: (None if not isinstance(value, list) else [])
        for key, value in evidence["extraction_quality"].items()
    }
    evidence_path = DOCS / "evidence_map_template.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    written = [pdf_path, html_path, lead_path, evidence_path]

    try:
        import pypdfium2 as pdfium

        png_path = DOCS / "report_template.png"
        pdfium.PdfDocument(str(pdf_path))[0].render(scale=2.0).to_pil().save(png_path)
        written.append(png_path)
    except ImportError:
        print("pypdfium2 not installed; skipping the PNG preview.", file=sys.stderr)

    for path in written:
        print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
