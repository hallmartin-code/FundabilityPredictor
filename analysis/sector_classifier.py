"""Sector classification from deck vocabulary.

Sector drives how traction is judged: a pre-revenue biotech is normal, a
pre-revenue Series A SaaS company is not. Getting this wrong distorts the
whole score, so classification requires real keyword evidence and otherwise
returns the neutral "unknown" profile rather than guessing.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from config.settings import AppConfig, load_config
from models.company import SectorProfile
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)

#: Keyword hits needed before a sector is claimed at all.
MIN_HITS_FOR_MATCH = 2
#: Hits needed for HIGH confidence.
MIN_HITS_FOR_HIGH = 6
#: How far ahead of the runner-up the winner must be for HIGH confidence.
DOMINANCE_FOR_HIGH = 1.6


def classify_sector(profile: DeckProfile, config: AppConfig | None = None) -> SectorProfile:
    """Resolve the company's sector and the rules that go with it."""
    cfg = config or load_config()
    rules = cfg.sector_rules

    declared = " ".join(
        filter(None, [profile.text("sector"), profile.text("industry"), profile.text("subsector")])
    ).lower()
    corpus = " ".join(profile.raw_text_by_page.values()).lower()

    scores: Counter[str] = Counter()
    matched: dict[str, list[str]] = {}
    for key, rule in rules.items():
        keywords = rule.get("keywords") or []
        hits: list[str] = []
        weight = 0.0
        for keyword in keywords:
            pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)
            occurrences = len(pattern.findall(corpus))
            if occurrences:
                hits.append(keyword)
                # Diminishing returns: ten mentions of one word should not beat
                # broad vocabulary coverage across several distinct terms.
                weight += 1.0 + min(occurrences - 1, 4) * 0.25
            # An explicitly declared sector is worth far more than prose.
            if declared and pattern.search(declared):
                weight += 4.0
                if keyword not in hits:
                    hits.append(keyword)
        if hits:
            scores[key] = weight
            matched[key] = hits

    scores.pop("unknown", None)
    if not scores:
        LOGGER.info("No sector keywords matched; using the neutral profile.")
        return _build(cfg, "unknown", [], "LOW")

    ranked = scores.most_common(2)
    best_key, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    hits = matched[best_key]

    if len(hits) < MIN_HITS_FOR_MATCH and best_score < 4.0:
        LOGGER.info("Sector evidence too thin (%s); using the neutral profile.", best_key)
        return _build(cfg, "unknown", hits, "LOW")

    dominance = best_score / runner_up if runner_up else float("inf")
    if len(hits) >= MIN_HITS_FOR_HIGH and dominance >= DOMINANCE_FOR_HIGH:
        confidence = "HIGH"
    elif len(hits) >= 3 or dominance >= 1.3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    LOGGER.info(
        "Sector classified as %s (score %.1f, %d keyword hits, confidence %s)",
        best_key, best_score, len(hits), confidence,
    )
    return _build(cfg, best_key, hits, confidence)


def _build(cfg: AppConfig, key: str, hits: list[str], confidence: str) -> SectorProfile:
    rule = cfg.sector_rules.get(key, cfg.sector_rules["unknown"])
    return SectorProfile(
        key=key,
        label=rule.get("label", "Not Provided"),
        traction_mode=rule.get("traction_mode", "hybrid"),
        capital_intensity=rule.get("capital_intensity", "medium"),
        regulated=bool(rule.get("regulated", False)),
        revenue_expected_from=rule.get("revenue_expected_from", "Series A"),
        probability_modifier=dict(rule.get("probability_modifier", {"p3": 1.0, "p6": 1.0, "p12": 1.0})),
        note=rule.get("note", ""),
        matched_keywords=hits[:10],
        confidence=confidence,
    )
