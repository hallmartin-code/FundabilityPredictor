"""Prediction confidence (0-100).

Confidence answers a different question from readiness. Readiness asks "how
financeable does this company look?"; confidence asks "how much should you
trust that answer given what the deck actually disclosed?".

The two are deliberately independent: a thin deck describing a strong company
can produce a high readiness score with low confidence, and the report says so
rather than presenting a guess as a finding.
"""

from __future__ import annotations

import logging

from config.settings import AppConfig, load_config
from models.evidence import EVIDENCE_WEIGHT, ValueStatus
from models.fields import spec_for
from models.prediction import ConfidenceResult
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)

#: Confidence lost per inconsistency found, within the consistency component.
INCONSISTENCY_PENALTY = 0.25

#: Plain-language note explaining what each missing critical field costs.
MISSING_FIELD_IMPACT: dict[str, str] = {
    "target_raise": "The target raise was not stated, so round-size and stage fit could not be assessed.",
    "amount_committed": "Current round commitments were not provided, reducing confidence in the 3-month financing estimate.",
    "percent_committed": "Round progress was not disclosed, which is the single strongest near-term closing signal.",
    "annual_revenue": "No revenue figure was found, so commercial traction is inferred rather than measured.",
    "arr": "No recurring revenue figure was found, limiting assessment of revenue quality.",
    "prior_funding": "Prior capital raised was not stated, weakening the stage inference.",
    "pre_money_valuation": "No valuation was stated, so terms could not be compared with stage benchmarks.",
    "runway_months": "Runway was not disclosed, so financing urgency could not be assessed.",
    "customers": "No customer count was found, so demand evidence relies on qualitative statements.",
    "gross_margin": "Gross margin was not stated, limiting the view of unit economics.",
    "burn_rate": "Burn rate was not disclosed, so capital efficiency could not be evaluated.",
    "founders": "The founding team was not clearly identified, limiting team assessment.",
    "lead_investor": "No lead investor position was stated, which materially affects near-term closing odds.",
    "use_of_funds": "Use of proceeds was not specified, weakening the credibility of the ask.",
    "sector": "The sector was inferred from deck language rather than stated.",
    "stage_declared": "The round stage was not declared, so it was inferred from the financials.",
}


def score_confidence(profile: DeckProfile, config: AppConfig | None = None) -> ConfidenceResult:
    """Compute confidence and explain what is holding it down."""
    cfg = config or load_config()
    conf_cfg = cfg.confidence
    weights = conf_cfg["components"]

    completeness, missing = _critical_completeness(profile, conf_cfg["critical_fields"])
    evidence = _evidence_quality(profile)
    extraction = _extraction_quality(profile)
    consistency = _consistency(profile)

    components = {
        "critical_field_completeness": round(completeness, 3),
        "evidence_quality": round(evidence, 3),
        "extraction_quality": round(extraction, 3),
        "consistency": round(consistency, 3),
    }
    score = sum(components[name] * float(weight) for name, weight in weights.items()) * 100.0
    score = round(max(0.0, min(100.0, score)), 1)

    level = "LOW"
    for band in conf_cfg["levels"]:
        if score >= float(band["min"]):
            level = str(band["level"])
            break

    notes = _notes(profile, missing)
    LOGGER.info("Prediction confidence: %.1f (%s)", score, level)
    return ConfidenceResult(
        score=score,
        level=level,
        components=components,
        missing_critical_fields=missing,
        notes=notes,
    )


def _critical_completeness(
    profile: DeckProfile, critical: dict[str, float]
) -> tuple[float, list[str]]:
    """Weighted share of the probability-critical fields that are present."""
    total = 0.0
    earned = 0.0
    missing: list[str] = []
    for field, weight in critical.items():
        weight = float(weight)
        total += weight
        value = profile.get(field)
        if value.is_known:
            # A guessed value is worth less than a stated one, even here.
            earned += weight * (0.6 if value.status is ValueStatus.ESTIMATED else 1.0)
        else:
            missing.append(field)
    return (earned / total if total else 0.0), missing


def _evidence_quality(profile: DeckProfile) -> float:
    """Mean provenance strength across everything actually extracted."""
    known = [profile.get(name) for name in profile.known_fields()]
    if not known:
        return 0.0
    scores = [
        EVIDENCE_WEIGHT.get(field.status, 0.0) * max(field.confidence, 0.3) for field in known
    ]
    return sum(scores) / len(scores)


def _extraction_quality(profile: DeckProfile) -> float:
    """How readable the source document was."""
    quality = profile.extraction_quality
    if not quality.pages_total:
        return 0.0
    readable = 1.0 - quality.image_based_ratio
    density = min(1.0, (quality.total_characters / quality.pages_total) / 400.0)
    return max(0.0, min(1.0, 0.65 * readable + 0.35 * density))


def _consistency(profile: DeckProfile) -> float:
    if not profile.inconsistencies:
        return 1.0
    return max(0.0, 1.0 - INCONSISTENCY_PENALTY * len(profile.inconsistencies))


def _notes(profile: DeckProfile, missing: list[str]) -> list[str]:
    """Explain the effect of each gap, most damaging first."""
    notes: list[str] = []
    for field in missing:
        note = MISSING_FIELD_IMPACT.get(field)
        if note:
            notes.append(note)
        else:
            notes.append(f"{spec_for(field).label} was not provided.")

    for warning in profile.extraction_quality.warnings():
        notes.append(warning)

    estimated = profile.estimated_fields()
    if estimated:
        labels = ", ".join(spec_for(f).label for f in sorted(estimated)[:5])
        notes.append(f"Some values are model estimates rather than deck statements: {labels}.")

    for inconsistency in profile.inconsistencies:
        notes.append(f"Flagged for review: {inconsistency.message}")

    return notes
