"""Configuration loading.

All model tuning lives in the YAML files beside this module. Nothing in the
analysis package hard-codes a weight, threshold or band.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

SCORING_WEIGHTS_FILE = CONFIG_DIR / "scoring_weights.yaml"
SECTOR_RULES_FILE = CONFIG_DIR / "sector_rules.yaml"
PROBABILITY_RULES_FILE = CONFIG_DIR / "probability_rules.yaml"


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing or structurally invalid."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a YAML mapping at the top level.")
    return data


@dataclass(frozen=True)
class AppConfig:
    """The three config documents, loaded and validated together."""

    scoring: dict[str, Any]
    sectors: dict[str, Any]
    probability: dict[str, Any]

    # -------------------------------------------------------------- accessors
    @property
    def categories(self) -> dict[str, Any]:
        return self.scoring["categories"]

    @property
    def missing_data_credit(self) -> float:
        return float(self.scoring.get("missing_data_credit", 0.35))

    @property
    def readiness_labels(self) -> list[dict[str, Any]]:
        return self.scoring["readiness_labels"]

    @property
    def sector_rules(self) -> dict[str, Any]:
        return self.sectors["sectors"]

    @property
    def stage_benchmarks(self) -> dict[str, Any]:
        return self.sectors["stage_benchmarks"]

    @property
    def stage_order(self) -> list[str]:
        return self.sectors["stage_order"]

    @property
    def capital_intensity_multiplier(self) -> dict[str, float]:
        return self.sectors["capital_intensity_multiplier"]

    @property
    def comparable_profile_rules(self) -> list[dict[str, Any]]:
        return self.sectors["comparable_profiles"]["rules"]

    @property
    def comparable_profile_default(self) -> dict[str, Any]:
        return self.sectors["comparable_profiles"]["default"]

    @property
    def baseline_bands(self) -> list[dict[str, Any]]:
        return self.probability["baseline_bands"]

    @property
    def positive_signals(self) -> dict[str, Any]:
        return self.probability.get("positive_signals", {})

    @property
    def negative_signals(self) -> dict[str, Any]:
        return self.probability.get("negative_signals", {})

    @property
    def clamp(self) -> tuple[float, float]:
        c = self.probability["clamp"]
        return float(c["min"]), float(c["max"])

    @property
    def market_conditions(self) -> dict[str, Any]:
        return self.probability["market_conditions"]

    @property
    def monotonicity(self) -> dict[str, Any]:
        return self.probability.get("monotonicity", {"enforce": True, "min_gap": 0.0})

    @property
    def confidence(self) -> dict[str, Any]:
        return self.probability["confidence"]

    def readiness_label(self, score: float) -> str:
        for band in self.readiness_labels:
            if score >= float(band["min"]):
                return str(band["label"])
        return str(self.readiness_labels[-1]["label"])

    def validate(self) -> None:
        """Fail loudly at load time rather than silently mis-scoring later."""
        total = sum(float(c["weight"]) for c in self.categories.values())
        if abs(total - 100.0) > 0.01:
            raise ConfigError(
                f"scoring_weights.yaml category weights sum to {total}, expected 100."
            )
        comp = self.confidence["components"]
        comp_total = sum(float(v) for v in comp.values())
        if abs(comp_total - 1.0) > 0.001:
            raise ConfigError(
                f"probability_rules.yaml confidence components sum to {comp_total}, expected 1.0."
            )
        lo, hi = self.clamp
        if not 0.0 < lo < hi < 100.0:
            raise ConfigError("probability clamp must satisfy 0 < min < max < 100.")
        for band in self.baseline_bands:
            for h in ("p3", "p6", "p12"):
                low, high = band[h]
                if low > high:
                    raise ConfigError(f"Baseline band {band['min']}-{band['max']} has {h} low > high.")
        default_env = self.market_conditions["default"]
        if default_env not in self.market_conditions["environments"]:
            raise ConfigError(f"Default market condition {default_env!r} is not defined.")
        missing_bench = set(self.stage_order) - set(self.stage_benchmarks)
        if missing_bench:
            raise ConfigError(f"stage_benchmarks missing entries for: {sorted(missing_bench)}")


@lru_cache(maxsize=4)
def load_config(config_dir: str | None = None) -> AppConfig:
    """Load and validate the configuration set (cached per directory)."""
    base = Path(config_dir) if config_dir else CONFIG_DIR
    cfg = AppConfig(
        scoring=_load_yaml(base / "scoring_weights.yaml"),
        sectors=_load_yaml(base / "sector_rules.yaml"),
        probability=_load_yaml(base / "probability_rules.yaml"),
    )
    cfg.validate()
    LOGGER.debug("Loaded configuration from %s", base)
    return cfg


# ---------------------------------------------------------------------------
# Environment-driven runtime settings. No credential is ever hard-coded.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RuntimeSettings:
    anthropic_api_key: str | None
    llm_model: str
    llm_enabled: bool
    market_condition: str | None
    page_size: str
    output_dir: Path
    log_level: str
    resend_api_key: str | None
    email_enabled: bool
    email_on_failure: bool
    email_to: list[str]
    email_attachments: tuple[str, ...]

    @property
    def email_configured(self) -> bool:
        return bool(self.resend_api_key)

    @classmethod
    def from_env(cls) -> RuntimeSettings:
        key = os.getenv("ANTHROPIC_API_KEY") or None
        # Off by default. The presence of a key in the environment must not
        # silently change a score: two people running the same deck should get
        # the same numbers unless one of them explicitly asks for LLM assist,
        # via --llm or FP_LLM_ENABLED.
        enabled = os.getenv("FP_LLM_ENABLED", "off").strip().lower()
        if enabled == "auto":
            llm_enabled = key is not None
        else:
            llm_enabled = enabled in ("1", "true", "yes", "on")
        out = os.getenv("FP_OUTPUT_DIR")

        # Email is off by default for the same reason the LLM is: the mere
        # presence of a key must not start sending mail about every analysis.
        resend_key = os.getenv("RESEND_API_KEY") or None
        email_flag = os.getenv("FP_EMAIL_ENABLED", "off").strip().lower()
        if email_flag == "auto":
            email_enabled = resend_key is not None
        else:
            email_enabled = email_flag in ("1", "true", "yes", "on")

        recipients = os.getenv("FP_EMAIL_TO") or "Info@tencapital.group"
        attachments = os.getenv("FP_EMAIL_ATTACH", "pdf,data")

        return cls(
            anthropic_api_key=key,
            llm_model=os.getenv("FP_LLM_MODEL", "claude-opus-5"),
            llm_enabled=llm_enabled,
            market_condition=os.getenv("FP_MARKET_CONDITION") or None,
            page_size=os.getenv("FP_PAGE_SIZE", "LETTER").upper(),
            output_dir=Path(out) if out else DEFAULT_OUTPUT_DIR,
            log_level=os.getenv("FP_LOG_LEVEL", "INFO").upper(),
            resend_api_key=resend_key,
            email_enabled=email_enabled,
            email_on_failure=os.getenv("FP_EMAIL_ON_FAILURE", "on").strip().lower()
            in ("1", "true", "yes", "on"),
            email_to=[a.strip() for a in recipients.split(",") if a.strip()],
            email_attachments=tuple(
                a.strip().lower() for a in attachments.split(",") if a.strip()
            ),
        )
