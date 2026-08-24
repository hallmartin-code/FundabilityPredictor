"""Qualitative comparable financing profile.

This is a *classification*, not a claim about specific transactions. The
system has no comparable-transaction database, so it never asserts that a
named company raised a named amount; it places this raise into a recognisable
financing archetype and says why.
"""

from __future__ import annotations

import logging

from analysis import signals as sig
from config.settings import AppConfig, load_config
from models.company import SectorProfile, Stage
from models.prediction import ComparableProfile
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)


def classify_comparable(
    profile: DeckProfile,
    sector: SectorProfile,
    stage: Stage,
    readiness_total: float,
    config: AppConfig | None = None,
) -> ComparableProfile:
    """Select the first archetype whose conditions all match."""
    cfg = config or load_config()

    for rule in cfg.comparable_profile_rules:
        if _matches(rule.get("when", {}), profile, sector, stage, readiness_total):
            LOGGER.info("Comparable profile: %s", rule["label"])
            return ComparableProfile(
                key=str(rule["id"]),
                label=str(rule["label"]),
                rationale=str(rule["rationale"]),
            )

    default = cfg.comparable_profile_default
    LOGGER.info("Comparable profile: %s (default)", default["label"])
    return ComparableProfile(
        key=str(default["id"]),
        label=str(default["label"]),
        rationale=str(default["rationale"]),
    )


def _matches(
    when: dict,
    profile: DeckProfile,
    sector: SectorProfile,
    stage: Stage,
    readiness: float,
) -> bool:
    if "min_readiness" in when and readiness < float(when["min_readiness"]):
        return False
    if "max_readiness" in when and readiness > float(when["max_readiness"]):
        return False
    if "stages" in when and stage.value not in when["stages"]:
        return False
    if "sectors" in when and sector.key not in when["sectors"]:
        return False
    if "traction_modes" in when and sector.traction_mode not in when["traction_modes"]:
        return False
    if "capital_intensities" in when and sector.capital_intensity not in when["capital_intensities"]:
        return False
    if when.get("requires_partnership") and not sig.has_partnership(profile):
        return False
    if "max_runway_months" in when:
        runway = sig.runway_months(profile)
        # An unknown runway cannot satisfy a runway-bounded archetype.
        if runway is None or runway > float(when["max_runway_months"]):
            return False
    return True
