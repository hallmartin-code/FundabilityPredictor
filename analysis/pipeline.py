"""The analysis pipeline: a deck path in, a complete AnalysisResult out.

Ordering matters and is enforced here:

    ingest -> extract -> (optional LLM enrichment) -> overrides -> derive
           -> sector -> stage -> consistency -> readiness -> probability
           -> confidence -> blockers -> actions -> comparable -> TEN Capital

Overrides are applied *before* derivation so a corrected raise amount flows
into every downstream calculation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from analysis.blockers import identify_blockers
from analysis.comparable_profile import classify_comparable
from analysis.confidence_model import score_confidence
from analysis.consistency import check_consistency
from analysis.derive import derive_fields
from analysis.probability_model import predict
from analysis.readiness_score import score_readiness
from analysis.recommendations import recommend_actions
from analysis.sector_classifier import classify_sector
from analysis.stage_inference import infer_stage
from analysis.ten_capital_fit import assess_ten_capital_fit
from config.settings import AppConfig, load_config
from extractors.field_extractor import FieldExtractor
from extractors.ingest import ingest
from models.prediction import AnalysisResult
from models.profile import DeckProfile
from services.llm import ExtractionRun, LLMService, NullLLMService, apply_llm_extraction

LOGGER = logging.getLogger(__name__)


def extract_profile(
    path: str | Path,
    llm: LLMService | None = None,
    overrides: dict[str, Any] | None = None,
) -> DeckProfile:
    """Ingest and extract, up to and including derived fields.

    Exposed separately from :func:`analyse_profile` so a review-and-correct
    workflow can show the user what was found before any scoring happens.
    """
    document = ingest(path)
    profile = FieldExtractor().extract(document)

    # Claude reads the deck after the rule-based pass, so deterministic
    # extraction is the floor and the model fills what it could not reach.
    run = apply_llm_extraction(profile, llm or NullLLMService())
    profile.llm_run = run.as_dict() if run.requests else None

    if overrides:
        apply_overrides(profile, overrides)

    derive_fields(profile)
    return profile


def apply_overrides(profile: DeckProfile, overrides: dict[str, Any]) -> list[str]:
    """Record human corrections, keeping the extracted originals intact."""
    applied: list[str] = []
    for field, value in overrides.items():
        if value is None:
            continue
        try:
            profile.apply_override(field, value)
            applied.append(field)
        except KeyError:
            LOGGER.warning("Ignoring override for unknown field %r.", field)
    if applied:
        LOGGER.info("Applied %d user override(s): %s", len(applied), ", ".join(applied))
    return applied


def analyse_profile(
    profile: DeckProfile,
    config: AppConfig | None = None,
    market_condition: str | None = None,
) -> AnalysisResult:
    """Run the full model over an already-extracted profile."""
    cfg = config or load_config()

    sector = classify_sector(profile, cfg)
    stage_assessment = infer_stage(profile, cfg)
    stage = stage_assessment.effective

    inconsistencies = check_consistency(profile)

    readiness = score_readiness(profile, sector, stage, cfg)
    prediction = predict(
        profile, sector, stage, readiness.total, cfg, market_condition=market_condition
    )
    confidence = score_confidence(profile, cfg)
    blocker_list = identify_blockers(profile, sector, stage, cfg)
    actions = recommend_actions(profile, sector, stage, cfg)
    comparable = classify_comparable(profile, sector, stage, readiness.total, cfg)
    ten_capital = assess_ten_capital_fit(profile, sector, stage, readiness.total, confidence)

    return AnalysisResult(
        profile=profile,
        stage=stage_assessment,
        sector=sector,
        readiness=readiness,
        prediction=prediction,
        confidence=confidence,
        blockers=blocker_list,
        actions=actions,
        comparable=comparable,
        ten_capital=ten_capital,
        inconsistencies=inconsistencies,
    )


def analyse(
    path: str | Path,
    config: AppConfig | None = None,
    llm: LLMService | None = None,
    overrides: dict[str, Any] | None = None,
    market_condition: str | None = None,
) -> AnalysisResult:
    """Convenience wrapper: ingest, extract and analyse in one call."""
    cfg = config or load_config()
    profile = extract_profile(path, llm=llm, overrides=overrides)
    result = analyse_profile(profile, cfg, market_condition=market_condition)
    LOGGER.info(
        "Analysis complete for %s: readiness %.1f, 6-month probability %.0f%%, confidence %s",
        result.company_name,
        result.readiness.total,
        result.prediction.final.p6,
        result.confidence.level,
    )
    return result
