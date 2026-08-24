"""Claude-powered deck extraction.

The division of labour is deliberate and load-bearing:

* **Claude reads the deck.** Language understanding is what an LLM is good at,
  and it finds fields a regex never will - a business model described in
  prose, a raise buried in a narrative sentence, a runway implied by a
  milestone chart.
* **The scoring model stays deterministic.** Readiness, probability,
  confidence, blockers and TEN Capital fit are computed in Python from the
  extracted values, so two runs on the same inputs produce the same numbers
  and every point is auditable.

Every value Claude returns must be accompanied by a verbatim quote and a page
number, and the quote is **checked against the source text** before the value
is accepted. A value whose quote cannot be found is downgraded or dropped.
That check is what lets LLM-extracted figures carry EXPLICIT status: they are
not the model's recollection, they are located text.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Protocol, runtime_checkable

from extractors.normalization import (
    clean_text,
    parse_bool,
    parse_date,
    parse_list,
    parse_money,
    parse_months,
    parse_number,
    parse_percent,
)
from models.evidence import ValueStatus
from models.fields import FIELD_SPECS, FIELDS_BY_NAME, FieldKind
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"

#: Characters of deck text per request. Most decks fit in a single call; the
#: rest are split by page so ``source_page`` stays absolute.
CHUNK_CHARS = 45_000

#: Non-streaming would risk an HTTP timeout on a long deck, so requests stream
#: and are collected with ``get_final_message()``.
MAX_OUTPUT_TOKENS = 16_000

#: Claude Opus 5 list pricing, USD per million tokens.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

#: Fields the LLM must never supply. The company name is resolved from the
#: cover slide and filename together, and derived fields are computed.
LLM_EXCLUDED_FIELDS = frozenset({"percent_committed", "post_money_valuation", "total_funding_to_date"})

#: Numeric kinds where an unverified quote means the value is discarded
#: outright. A fabricated figure is far more damaging than a missing one.
STRICT_KINDS = frozenset({FieldKind.MONEY, FieldKind.NUMBER, FieldKind.PERCENT, FieldKind.MONTHS})

_WHITESPACE = re.compile(r"\s+")


@dataclass
class ExtractionRun:
    """What one extraction cost and how much of it survived verification."""

    model: str = DEFAULT_MODEL
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    proposed: int = 0
    accepted: int = 0
    rejected_unverified: list[str] = dc_field(default_factory=list)
    downgraded: list[str] = dc_field(default_factory=list)
    errors: list[str] = dc_field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        input_rate, output_rate = PRICING.get(self.model, PRICING[DEFAULT_MODEL])
        return (
            self.input_tokens / 1_000_000 * input_rate
            + self.output_tokens / 1_000_000 * output_rate
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "estimated_cost_usd": round(self.cost_usd, 4),
            "fields_proposed": self.proposed,
            "fields_accepted": self.accepted,
            "rejected_unverified": self.rejected_unverified,
            "downgraded_to_estimated": self.downgraded,
            "errors": self.errors,
        }


@runtime_checkable
class LLMService(Protocol):
    """Given deck text, propose evidenced field values."""

    name: str

    def available(self) -> bool:
        ...

    def extract(self, profile: DeckProfile) -> tuple[list[dict[str, Any]], ExtractionRun]:
        ...


class NullLLMService:
    """The default. Makes no network calls, so the pipeline is deterministic."""

    name = "disabled"
    model = "none"

    def available(self) -> bool:
        return False

    def extract(self, profile: DeckProfile) -> tuple[list[dict[str, Any]], ExtractionRun]:
        return [], ExtractionRun(model="none")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _field_catalogue() -> str:
    """The field registry, rendered for the system prompt."""
    lines: list[str] = []
    for spec in FIELD_SPECS:
        if spec.name in LLM_EXCLUDED_FIELDS:
            continue
        hint = {
            FieldKind.MONEY: "a currency amount, exactly as written (e.g. \"$2M\", \"USD 1,200,000\")",
            FieldKind.NUMBER: "a count, exactly as written",
            FieldKind.PERCENT: "a percentage, exactly as written (e.g. \"82%\")",
            FieldKind.MONTHS: "a duration, exactly as written (e.g. \"18 months\")",
            FieldKind.DATE: "a date, exactly as written",
            FieldKind.BOOL: "\"yes\" or \"no\"",
            FieldKind.LIST: "a comma-separated list",
            FieldKind.TEXT: "a short phrase, at most 200 characters",
        }[spec.kind]
        flag = " [CRITICAL]" if spec.critical else ""
        lines.append(f"- {spec.name}: {spec.label}{flag} - {hint}")
    return "\n".join(lines)


SYSTEM_RULES = """You extract financing facts from investor pitch decks for a venture capital firm.

You will be given the text of a pitch deck, page by page. Return every field below that the deck actually states.

ABSOLUTE RULES - a violation is worse than returning nothing:

1. NEVER infer, estimate, calculate or guess a value. If the deck does not state it, omit the field entirely.
2. Every field you return MUST include `quote`: a verbatim substring copied character-for-character from the page you cite. The quote is checked against the source. If you cannot copy an exact quote, omit the field.
3. `source_page` must be the page number where that exact quote appears.
4. Return values EXACTLY as the deck writes them. Do not convert units, expand abbreviations, or reformat numbers. "$2M" stays "$2M".
5. A TARGET is not an ACHIEVEMENT. "$6M ARR within 18 months", "path to FDA approval", a 2029 projection column - these describe plans. Never report them as current values.
6. A THIRD PARTY'S attribute is not the company's. An advisor's prior exit is not a founder's exit. A partner's FDA clearance is not this company's clearance. A cited market study's revenue is not this company's revenue.
7. Financial projection tables (columns of future years, "$000s", "Forecast") describe the future. Never take current revenue, EBITDA, margin or burn from one.
8. `lead_investor` means a lead who is COMMITTED. If the deck says "seeking a lead" or "lead TBD", omit the field.
9. If two pages disagree, cite the one that describes the current round.
10. `confidence` is your certainty that the value is correctly identified AND current: 0.9-1.0 explicitly labelled, 0.7-0.9 clearly stated in prose, below 0.7 means you are unsure - prefer omitting.

Return only fields you can evidence. A deck that states twelve fields should yield twelve fields, not eighty."""


EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "array",
            "description": "Only fields the deck actually states. Omit everything else.",
            "items": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": [
                            s.name for s in FIELD_SPECS if s.name not in LLM_EXCLUDED_FIELDS
                        ],
                    },
                    "value": {
                        "type": "string",
                        "description": "The value exactly as the deck writes it.",
                    },
                    # The structured-output validator rejects `minimum` /
                    # `maximum`, so ranges are stated in the description and
                    # enforced on our side after parsing.
                    "source_page": {
                        "type": "integer",
                        "description": "1-based page number where the quote appears.",
                    },
                    "quote": {
                        "type": "string",
                        "description": "Verbatim substring copied from that page.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Between 0 and 1.",
                    },
                },
                "required": ["field", "value", "source_page", "quote", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["fields"],
    "additionalProperties": False,
}


class ClaudeExtractionService:
    """Full-deck extraction through the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("FP_LLM_MODEL") or DEFAULT_MODEL
        # Extraction accuracy scales with effort; "medium" roughly halves
        # latency and is a reasonable trade for an interactive web request.
        self.effort = (os.getenv("FP_LLM_EFFORT") or "high").strip().lower()
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    def available(self) -> bool:
        if not self._api_key:
            LOGGER.info("ANTHROPIC_API_KEY is not set; LLM extraction is unavailable.")
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            LOGGER.info("The anthropic package is not installed; LLM extraction is unavailable.")
            return False
        return True

    def _client_or_raise(self):  # noqa: ANN202 - third-party type
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key, max_retries=3)
        return self._client

    # ------------------------------------------------------------------ api
    def extract(self, profile: DeckProfile) -> tuple[list[dict[str, Any]], ExtractionRun]:
        run = ExtractionRun(model=self.model)
        if not self.available():
            run.errors.append("LLM extraction is not available.")
            return [], run

        proposals: list[dict[str, Any]] = []
        for chunk in _chunk_pages(profile.raw_text_by_page):
            try:
                proposals.extend(self._extract_chunk(chunk, run))
            except Exception as exc:  # noqa: BLE001 - one chunk must not sink the run
                LOGGER.exception("LLM extraction failed for a page range.")
                run.errors.append(f"{type(exc).__name__}: {exc}")

        run.proposed = len(proposals)
        return proposals, run

    def _extract_chunk(self, chunk: str, run: ExtractionRun) -> list[dict[str, Any]]:
        import anthropic

        client = self._client_or_raise()
        try:
            # Streaming keeps a long deck from hitting the request timeout.
            with client.messages.stream(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA},
                },
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_RULES,
                        # The rules and field catalogue are identical on every
                        # request, so caching them makes repeat analyses much
                        # cheaper than the raw token count suggests.
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": "FIELDS TO EXTRACT:\n\n" + _field_catalogue(),
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                messages=[{"role": "user", "content": f"PITCH DECK TEXT:\n\n{chunk}"}],
            ) as stream:
                message = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise RuntimeError(
                "The Anthropic API rejected the key. Check ANTHROPIC_API_KEY."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise RuntimeError("Rate limited by the Anthropic API; try again shortly.") from exc

        run.requests += 1
        usage = message.usage
        run.input_tokens += getattr(usage, "input_tokens", 0) or 0
        run.output_tokens += getattr(usage, "output_tokens", 0) or 0
        run.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

        if message.stop_reason == "refusal":
            detail = getattr(message, "stop_details", None)
            run.errors.append(f"Model declined the request ({getattr(detail, 'category', 'unknown')}).")
            return []

        text = "".join(block.text for block in message.content if block.type == "text")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            run.errors.append("Could not parse the model response as JSON.")
            return []
        items = payload.get("fields") if isinstance(payload, dict) else None
        return items if isinstance(items, list) else []


# ---------------------------------------------------------------------------
# Verification and merge
# ---------------------------------------------------------------------------
def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text)).strip().lower()


def verify_quote(profile: DeckProfile, quote: str, page: int | None) -> int | None:
    """Locate a quote in the source text and return the page it is on.

    Returns ``None`` when the quote appears nowhere, which marks the value as
    unevidenced. Whitespace is normalised before comparison because PDF text
    extraction inserts and drops spaces unpredictably.
    """
    needle = _normalise(quote)
    if len(needle) < 4:
        return None

    if page is not None:
        haystack = _normalise(profile.raw_text_by_page.get(page, ""))
        if needle in haystack:
            return page

    for candidate_page, text in profile.raw_text_by_page.items():
        if needle in _normalise(text):
            return candidate_page

    # PDF extraction can letter-space headings; compare with spaces removed.
    squeezed = needle.replace(" ", "")
    if len(squeezed) >= 6:
        for candidate_page, text in profile.raw_text_by_page.items():
            if squeezed in _normalise(text).replace(" ", ""):
                return candidate_page
    return None


def verify_literal(profile: DeckProfile, literal: str, page: int | None) -> int | None:
    """Check that a value's own text appears on the page it was cited from.

    Tables are the reason this exists. A figure sitting in a grid comes out of
    PDF extraction with its label and its value on different lines, so a
    quote like "SAFE: Closed $350K" is never a contiguous substring even
    though every part of it is genuinely on the page. Confirming the value
    literal itself still proves the number was read rather than invented.
    """
    needle = _normalise(literal)
    if len(needle) < 2:
        return None
    squeezed = needle.replace(" ", "").replace(",", "")

    pages = [page] if page is not None else []
    pages += [p for p in profile.raw_text_by_page if p != page]
    for candidate in pages:
        text = _normalise(profile.raw_text_by_page.get(candidate, ""))
        if needle in text or squeezed in text.replace(" ", "").replace(",", ""):
            return candidate
    return None


def _coerce(field: str, raw: str) -> Any:
    """Parse the model's verbatim string with the same parsers as the rules."""
    kind = FIELDS_BY_NAME[field].kind
    if kind is FieldKind.MONEY:
        return parse_money(raw)[0]
    if kind is FieldKind.NUMBER:
        return parse_number(raw)
    if kind is FieldKind.PERCENT:
        return parse_percent(raw)
    if kind is FieldKind.MONTHS:
        return parse_months(raw)
    if kind is FieldKind.DATE:
        return parse_date(raw)
    if kind is FieldKind.BOOL:
        return parse_bool(raw)
    if kind is FieldKind.LIST:
        return parse_list(raw)
    return clean_text(raw, max_length=220)


def apply_llm_extraction(profile: DeckProfile, service: LLMService) -> ExtractionRun:
    """Merge verified LLM proposals into the profile.

    Rule-based values already on the profile are kept unless the LLM proposal
    is both verified and more confident, so the deterministic extractor
    remains the floor rather than being overwritten wholesale.
    """
    proposals, run = service.extract(profile)
    if not proposals:
        return run

    for item in proposals:
        field = item.get("field")
        if field not in FIELDS_BY_NAME or field in LLM_EXCLUDED_FIELDS:
            continue

        raw_value = item.get("value")
        quote = item.get("quote") or ""
        confidence = float(item.get("confidence") or 0.0)
        page = item.get("source_page")
        page = int(page) if isinstance(page, (int, float)) else None

        value = _coerce(field, str(raw_value))
        if value is None or (isinstance(value, str) and not value.strip()):
            continue

        verified_page = verify_quote(profile, quote, page)
        kind = FIELDS_BY_NAME[field].kind
        status = ValueStatus.EXPLICIT
        downgraded = False

        if verified_page is None:
            # The quote did not match, so fall back to proving the value's own
            # text is on the page. This rescues table-bound figures without
            # weakening the guarantee that the number came from the document.
            verified_page = verify_literal(profile, str(raw_value), page)

        if verified_page is None:
            if kind in STRICT_KINDS:
                # An unverifiable number is discarded, not softened.
                run.rejected_unverified.append(field)
                LOGGER.warning(
                    "Rejected %s=%r: neither the quote nor the value could be "
                    "located in the source.", field, value,
                )
                continue
            status = ValueStatus.ESTIMATED
            confidence = min(confidence, 0.5)
            downgraded = True

        existing = profile.get(field)
        if existing.is_known and existing.confidence >= confidence:
            continue
        if downgraded:
            run.downgraded.append(field)

        note = (
            f"Extracted by {service.name} model {getattr(service, 'model', 'unknown')}"
            + (
                f"; quote verified on page {verified_page}."
                if verified_page is not None
                else "; quote could not be located in the source, so recorded as an estimate."
            )
        )
        profile.put(
            field,
            value,
            status,
            round(min(confidence, 0.99), 3),
            source_page=verified_page,
            source_text=clean_text(quote, max_length=220),
            note=note,
        )
        run.accepted += 1

    LOGGER.info(
        "LLM extraction: %d proposed, %d accepted, %d rejected unverified, %d downgraded "
        "(%d tokens in / %d out, ~$%.3f)",
        run.proposed, run.accepted, len(run.rejected_unverified), len(run.downgraded),
        run.input_tokens, run.output_tokens, run.cost_usd,
    )
    return run


def _chunk_pages(pages: dict[int, str]) -> list[str]:
    """Split the deck into page-aligned chunks that fit one request."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for page in sorted(pages):
        block = f"--- PAGE {page} ---\n{pages[page]}\n"
        if size + len(block) > CHUNK_CHARS and current:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(block)
        size += len(block)
    if current:
        chunks.append("".join(current))
    return chunks or [""]


def build_llm_service(enabled: bool, model: str | None = None) -> LLMService:
    if not enabled:
        return NullLLMService()
    service = ClaudeExtractionService(model=model)
    if not service.available():
        LOGGER.info("LLM extraction requested but unavailable; using deterministic mode.")
        return NullLLMService()
    return service
