"""JSON outputs: the TEN Capital lead record and the extraction evidence map.

Consent is never inferred. Unknown consent stays ``null``; the only way a
consent field becomes ``true`` is for a caller to pass it explicitly, which
means a human said so.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.evidence import ValueStatus
from models.fields import FIELD_SPECS
from models.prediction import AnalysisResult
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


class ConsentRecord:
    """Explicit consent capture.

    Defaults are ``None`` - not ``False`` - because "we never asked" and "they
    said no" are different facts, and only the second is a decision.
    """

    def __init__(
        self,
        analysis_consent: bool | None = None,
        ten_capital_contact_consent: bool | None = None,
        timestamp: str | None = None,
    ) -> None:
        self.analysis_consent = analysis_consent
        self.ten_capital_contact_consent = ten_capital_contact_consent
        self.timestamp = timestamp

    def as_dict(self) -> dict[str, Any]:
        recorded = self.analysis_consent is not None or self.ten_capital_contact_consent is not None
        return {
            "analysis_consent": self.analysis_consent,
            "ten_capital_contact_consent": self.ten_capital_contact_consent,
            "timestamp": self.timestamp
            or (datetime.now(timezone.utc).isoformat() if recorded else None),
        }


def build_lead_record(result: AnalysisResult, consent: ConsentRecord | None = None) -> dict[str, Any]:
    """Assemble the structured TEN Capital lead record."""
    profile = result.profile
    ten = result.ten_capital
    consent = consent or ConsentRecord()

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": result.generated_at.isoformat(),
        "company": {
            "name": profile.value("company_name"),
            "website": profile.value("website"),
            "sector": result.sector.label if result.sector.key != "unknown" else profile.value("sector"),
            "subsector": profile.value("subsector"),
            "stage_declared": result.stage.declared.value if result.stage.declared.is_known else None,
            "stage_inferred": result.stage.inferred.value if result.stage.inferred.is_known else None,
            "geography": (
                profile.value("headquarters")
                or profile.value("primary_geography")
                or profile.value("geographic_market")
                or profile.value("country")
            ),
        },
        "financing": {
            "target_raise": profile.value("target_raise"),
            "amount_committed": profile.value("amount_committed"),
            "percent_committed": profile.value("percent_committed"),
            "valuation": profile.value("pre_money_valuation")
            or profile.value("note_cap")
            or profile.value("safe_cap"),
            "instrument": profile.value("security_type"),
            "prior_funding": profile.value("prior_funding"),
            "runway_months": profile.value("runway_months"),
        },
        "traction": {
            "annual_revenue": profile.value("annual_revenue"),
            "arr": profile.value("arr"),
            "revenue_growth": profile.value("revenue_growth"),
            "gross_margin": profile.value("gross_margin"),
            "customers": profile.value("customers"),
        },
        "prediction": {
            "probability_3_months": result.prediction.final.p3,
            "probability_6_months": result.prediction.final.p6,
            "probability_12_months": result.prediction.final.p12,
            "financing_readiness_score": result.readiness.total,
            "financing_readiness_label": result.readiness.label,
            "confidence_score": result.confidence.score,
            "confidence_level": result.confidence.level,
            "comparable_profile": result.comparable.label,
            "comparable_profile_rationale": result.comparable.rationale,
            "stage_confidence": result.stage.confidence,
            "market_condition": result.prediction.market_condition.label,
            "market_condition_verified": result.prediction.market_condition.is_verified,
            "baseline": result.prediction.baseline.as_dict(),
            "probability_bridge": [
                {
                    "label": step.label,
                    "kind": step.kind,
                    "delta": {k: round(v, 2) for k, v in step.delta.items()},
                    "running": {k: round(v, 2) for k, v in step.running.items()},
                }
                for step in result.prediction.bridge
            ],
        },
        "readiness_breakdown": [
            {
                "category": category.key,
                "label": category.label,
                "weight": category.weight,
                "points": category.points,
                "signals": [
                    {
                        "key": s.key,
                        "label": s.label,
                        "weight": s.weight,
                        "awarded": s.awarded,
                        "evidenced": s.evidenced,
                        "detail": s.detail,
                    }
                    for s in category.signals
                ],
            }
            for category in result.readiness.categories
        ],
        "drivers": result.prediction.positive_signals,
        "blockers": [
            {"severity": b.severity, "title": b.title, "detail": b.detail} for b in result.blockers
        ],
        "recommended_actions": [
            {
                "action": a.action,
                "why_it_matters": a.why,
                "estimated_financing_impact": a.impact,
                "target_timeframe": a.timeframe,
            }
            for a in result.actions
        ],
        "ten_capital": {
            "fit_score": ten.fit_score,
            "raise_over_2m": ten.flag("raise_over_2m"),
            "twenty_five_percent_committed": ten.flag("twenty_five_percent_committed"),
            "fifty_percent_committed": ten.flag("fifty_percent_committed"),
            "revenue_over_700k": ten.flag("revenue_over_700k"),
            "revenue_generating": ten.flag("revenue_generating"),
            "lead_investor": ten.flag("lead_investor"),
            "institutional_round": ten.flag("institutional_round"),
            "runway_over_9_months": ten.flag("runway_over_9_months"),
            "materials_complete": ten.flag("materials_complete"),
            "likely_fundraising_candidate": ten.flag("likely_fundraising_candidate"),
            "recommended_next_step": ten.recommended_next_step,
            "explanation": ten.explanation,
        },
        "evidence": {
            "data_completeness_score": round(profile.completeness() * 100, 1),
            "missing_critical_fields": result.confidence.missing_critical_fields,
            "estimated_fields": sorted(profile.estimated_fields()),
            "overridden_fields": profile.overridden_fields(),
            "confidence_components": result.confidence.components,
            "confidence_notes": result.confidence.notes,
            "extraction_warnings": profile.extraction_quality.warnings(),
            "inconsistencies": [
                {
                    "code": i.code,
                    "severity": i.severity,
                    "message": i.message,
                    "fields": i.fields,
                }
                for i in result.inconsistencies
            ],
            "source_file": profile.source.filename,
            "source_sha256": profile.source.sha256,
            "pages": profile.source.page_count,
            "extraction_engine": "rules+claude" if profile.llm_run else "rules",
            "llm_extraction": profile.llm_run,
        },
        "consent": consent.as_dict(),
        "disclaimer": DISCLAIMER,
    }


DISCLAIMER = (
    "This report estimates financing probability based on company-supplied information, "
    "model assumptions, and available market context. It does not predict investor behavior "
    "with certainty and does not constitute investment, legal, tax, or securities advice."
)


def build_evidence_map(profile: DeckProfile) -> dict[str, Any]:
    """Full per-field provenance, including fields that stayed MISSING."""
    entries = []
    for spec in FIELD_SPECS:
        value = profile.get(spec.name)
        entry: dict[str, Any] = {
            "field": spec.name,
            "label": spec.label,
            "category": spec.category.value,
            "critical": spec.critical,
            "value": value.value,
            "status": value.status.value,
            "confidence": value.confidence,
            "source_page": value.source_page,
            "source_text": value.source_text,
        }
        if value.note:
            entry["note"] = value.note
        if value.overridden:
            entry["overridden"] = True
            entry["extracted_value"] = value.extracted_value
            entry["extracted_status"] = (
                value.extracted_status.value if value.extracted_status else None
            )
        entries.append(entry)

    counts = {status.value: 0 for status in ValueStatus}
    for entry in entries:
        counts[entry["status"]] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "source_file": profile.source.filename,
        "source_sha256": profile.source.sha256,
        "file_type": profile.source.file_type,
        "pages": profile.source.page_count,
        "extraction_quality": {
            "pages_total": profile.extraction_quality.pages_total,
            "pages_with_text": profile.extraction_quality.pages_with_text,
            "image_based_pages": profile.extraction_quality.image_based_pages,
            "total_characters": profile.extraction_quality.total_characters,
            "tables_found": profile.extraction_quality.tables_found,
            "notes_found": profile.extraction_quality.notes_found,
            "warnings": profile.extraction_quality.warnings(),
        },
        "status_counts": counts,
        "fields": entries,
    }


def write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
    LOGGER.info("Wrote %s", path)
    return path
