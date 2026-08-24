"""Turn the strings found in a deck into typed Python values.

These parsers are deliberately conservative: anything ambiguous returns
``None`` so the caller records the field as MISSING rather than inventing a
number. Currency is *recorded*, never converted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------
CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₩": "KRW",
    "₽": "RUB",
    "₪": "ILS",
}
CURRENCY_CODES = (
    "USD", "EUR", "GBP", "CAD", "AUD", "CHF", "JPY", "CNY", "INR",
    "SGD", "HKD", "NZD", "SEK", "NOK", "DKK", "ILS", "AED", "ZAR",
)

#: Scale words and suffixes. Order matters for the regex alternation below.
_SCALES: list[tuple[str, float]] = [
    ("trillion", 1e12),
    ("billion", 1e9),
    ("million", 1e6),
    ("thousand", 1e3),
    ("crore", 1e7),
    ("lakh", 1e5),
    ("bn", 1e9),
    ("mm", 1e6),
    ("mn", 1e6),
    ("k", 1e3),
    ("m", 1e6),
    ("b", 1e9),
    ("t", 1e12),
]
_SCALE_ALT = "|".join(re.escape(s) for s, _ in _SCALES)
_SCALE_LOOKUP = dict(_SCALES)

_CURRENCY_ALT = "|".join(
    [re.escape(s) for s in CURRENCY_SYMBOLS] + list(CURRENCY_CODES)
)

#: A currency amount, e.g. "$2.5M", "USD 1,200,000", "3 million euros".
MONEY_RE = re.compile(
    rf"(?P<cur>{_CURRENCY_ALT})?\s*"
    rf"(?P<num>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    rf"(?P<scale>{_SCALE_ALT})?\b",
    re.IGNORECASE,
)

PERCENT_RE = re.compile(
    r"(?P<num>-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:%|percent|pct\b)", re.IGNORECASE
)

NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")

_MONTHS_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?:\+\s*)?(?P<unit>months?|mos?\b|years?|yrs?\b|quarters?)",
    re.IGNORECASE,
)

_MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_TRUE_WORDS = ("yes", "true", "full time", "full-time", "fulltime", "100%", "committed")
_FALSE_WORDS = ("no", "false", "part time", "part-time", "parttime", "fractional")

#: Values that mean "the deck left this blank".
_NULL_TOKENS = {
    "", "-", "--", "n/a", "na", "none", "tbd", "tba", "unknown",
    "not provided", "not applicable", "?", "???", "xx", "xxx", "t.b.d.",
}


def clean_text(value: Any, max_length: int | None = None) -> str | None:
    """Collapse whitespace and reject placeholder strings."""
    if value is None:
        return None
    text = str(value).replace(" ", " ")
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n•-:;,")
    if text.lower() in _NULL_TOKENS or not text:
        return None
    if max_length and len(text) > max_length:
        text = text[: max_length - 1].rstrip() + "…"
    return text


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _apply_scale(number: float, scale: str | None, raw_number: str) -> float:
    """Apply a k/M/B style multiplier.

    A bare "m"/"k" after a number that already contains thousands separators
    (e.g. "1,200,000 m") is nonsense, but that case is harmless: we only ever
    apply the multiplier the text actually carries.
    """
    if not scale:
        return number
    factor = _SCALE_LOOKUP.get(scale.lower())
    if factor is None:
        return number
    # Guard against double-counting: "$1,500,000K" is far more likely to be a
    # mis-parse than a real 1.5bn figure, so ignore the suffix when the number
    # is already large and comma-formatted.
    if "," in raw_number and number >= 10_000:
        return number
    return number * factor


def parse_money(text: str | None) -> tuple[float | None, str | None]:
    """Parse the first currency amount in ``text``.

    Returns ``(amount, currency_code_or_None)``. Currency is reported, never
    converted, so a EUR figure stays a EUR figure.
    """
    if not text:
        return None, None
    match = MONEY_RE.search(str(text))
    if not match:
        return None, None
    number = _to_float(match.group("num"))
    if number is None:
        return None, None
    value = _apply_scale(number, match.group("scale"), match.group("num"))
    cur_raw = match.group("cur")
    currency: str | None = None
    if cur_raw:
        currency = CURRENCY_SYMBOLS.get(cur_raw, cur_raw.upper())
    else:
        low = str(text).lower()
        for word, code in (("euro", "EUR"), ("pound", "GBP"), ("dollar", "USD")):
            if word in low:
                currency = code
                break
    return value, currency


def parse_percent(text: str | None) -> float | None:
    """Parse a percentage, returned on a 0-100 scale."""
    if not text:
        return None
    match = PERCENT_RE.search(str(text))
    if match:
        return _to_float(match.group("num"))
    # Bare decimal fractions such as "0.35" are only treated as percentages
    # when they are unambiguous (strictly between 0 and 1).
    stripped = str(text).strip()
    if re.fullmatch(r"0?\.\d+", stripped):
        val = _to_float(stripped)
        return val * 100.0 if val is not None else None
    return None


def parse_number(text: str | None) -> float | None:
    """Parse a plain count, honouring k/M suffixes ("12k customers")."""
    if not text:
        return None
    s = str(text)
    match = re.search(
        rf"(?P<num>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<scale>{_SCALE_ALT})?\b",
        s,
        re.IGNORECASE,
    )
    if not match:
        return None
    number = _to_float(match.group("num"))
    if number is None:
        return None
    scale = match.group("scale")
    # Only trust a scale suffix that is glued to the number, so "5 million
    # users" scales but "5 companies in market" does not.
    if scale and scale.lower() in ("k", "m", "b", "bn", "mm", "mn", "million", "billion", "thousand"):
        return _apply_scale(number, scale, match.group("num"))
    return number


def parse_months(text: str | None) -> float | None:
    """Parse a duration into months ("18 months", "2 years", "3 quarters")."""
    if not text:
        return None
    match = _MONTHS_RE.search(str(text))
    if not match:
        return None
    number = _to_float(match.group("num"))
    if number is None:
        return None
    unit = match.group("unit").lower()
    if unit.startswith(("year", "yr")):
        return number * 12.0
    if unit.startswith("quarter"):
        return number * 3.0
    return number


def parse_bool(text: str | None) -> bool | None:
    if text is None:
        return None
    low = str(text).strip().lower()
    if not low:
        return None
    for word in _TRUE_WORDS:
        if word in low:
            return True
    for word in _FALSE_WORDS:
        if word in low:
            return False
    return None


def parse_date(text: str | None) -> str | None:
    """Parse a date into an ISO string. Month-year inputs resolve to day 1."""
    if not text:
        return None
    s = str(text).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    match = re.search(
        r"\b(?P<mon>jan|feb|mar|apr|may|jun|jul|aug|sept|sep|oct|nov|dec)[a-z]*\.?"
        r"[\s,]+(?P<year>20\d{2}|19\d{2})",
        s,
        re.IGNORECASE,
    )
    if match:
        month = _MONTH_NAMES[match.group("mon").lower()]
        return date(int(match.group("year")), month, 1).isoformat()
    match = re.search(r"\b(?P<q>[qQ][1-4])\s*(?P<year>20\d{2})", s)
    if match:
        month = {"q1": 1, "q2": 4, "q3": 7, "q4": 10}[match.group("q").lower()]
        return date(int(match.group("year")), month, 1).isoformat()
    match = re.fullmatch(r"\s*(20\d{2}|19\d{2})\s*", s)
    if match:
        return date(int(match.group(1)), 1, 1).isoformat()
    return None


def parse_list(text: str | None, max_items: int = 12) -> list[str] | None:
    """Split a delimited string into a clean list of names."""
    if not text:
        return None
    raw = re.split(r"[;,\n•\|]|\band\b|&", str(text))
    items: list[str] = []
    for part in raw:
        cleaned = clean_text(part, max_length=80)
        if cleaned and len(cleaned) > 1 and cleaned not in items:
            items.append(cleaned)
    return items[:max_items] or None


def months_until(iso_date: str | None, reference: date | None = None) -> float | None:
    """Months from ``reference`` (default today) until an ISO date.

    Negative when the date has already passed, which is exactly what the
    note-maturity check needs to see.
    """
    if not iso_date:
        return None
    try:
        target = date.fromisoformat(iso_date)
    except ValueError:
        return None
    ref = reference or date.today()
    return (target.year - ref.year) * 12 + (target.month - ref.month) + (target.day - ref.day) / 30.0


# ---------------------------------------------------------------------------
# Positional finders.
#
# A cue such as "pre-money" needs the number *nearest to it*, not the first
# number on the line: in "raising $2M at $8M pre-money" the first money value
# belongs to a different field. These finders return every candidate with its
# position so the caller can choose by proximity.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValueSpan:
    """A parsed value together with where it was found.

    ``strong`` marks a value that carries an explicit unit marker - a currency
    symbol/code for money, a % sign for percentages. A bare "100 million" in
    "TAM = 100 million ventilators" is a count, not an amount of money, so
    money fields prefer strong spans and only fall back to weak ones.
    """

    value: float
    start: int
    end: int
    strong: bool = True


def find_money(text: str) -> list[ValueSpan]:
    out: list[ValueSpan] = []
    for match in MONEY_RE.finditer(text or ""):
        number = _to_float(match.group("num"))
        if number is None:
            continue
        has_currency = bool(match.group("cur"))
        has_scale = bool(match.group("scale"))
        if not has_currency and not has_scale and "," not in match.group("num"):
            # A bare small integer is never an amount of money on its own.
            continue
        out.append(
            ValueSpan(
                _apply_scale(number, match.group("scale"), match.group("num")),
                match.start(),
                match.end(),
                strong=has_currency,
            )
        )
    return out


def find_percent(text: str) -> list[ValueSpan]:
    return [
        ValueSpan(v, m.start(), m.end(), True)
        for m in PERCENT_RE.finditer(text or "")
        if (v := _to_float(m.group("num"))) is not None
    ]


def find_number(text: str) -> list[ValueSpan]:
    out: list[ValueSpan] = []
    pattern = re.compile(
        rf"(?P<num>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<scale>{_SCALE_ALT})?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text or ""):
        number = _to_float(match.group("num"))
        if number is None:
            continue
        scale = match.group("scale")
        if scale and scale.lower() in ("k", "m", "b", "bn", "mm", "mn", "million", "billion", "thousand"):
            number = _apply_scale(number, scale, match.group("num"))
        out.append(ValueSpan(number, match.start(), match.end(), True))
    return out


def find_months(text: str) -> list[ValueSpan]:
    out: list[ValueSpan] = []
    for match in _MONTHS_RE.finditer(text or ""):
        number = _to_float(match.group("num"))
        if number is None:
            continue
        unit = match.group("unit").lower()
        if unit.startswith(("year", "yr")):
            number *= 12.0
        elif unit.startswith("quarter"):
            number *= 3.0
        out.append(ValueSpan(number, match.start(), match.end(), True))
    return out


def nearest_span(
    spans: list[ValueSpan], cue_start: int, cue_end: int, prefer_strong: bool = False
) -> ValueSpan | None:
    """Pick the span closest to a cue, preferring values that follow it."""
    if not spans:
        return None
    if prefer_strong:
        strong = [s for s in spans if s.strong]
        if strong:
            spans = strong

    def sort_key(span: ValueSpan) -> tuple[int, int]:
        if span.start >= cue_end:  # value follows the cue - the common layout
            return (0, span.start - cue_end)
        return (1, cue_start - span.end)

    return min(spans, key=sort_key)


# ---------------------------------------------------------------------------
# Display helpers (used by the report writers)
# ---------------------------------------------------------------------------
def format_money(value: float | None, currency: str | None = "USD", missing: str = "Not Provided") -> str:
    if value is None:
        return missing
    symbol = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency or "USD", "")
    prefix = symbol if symbol else f"{currency} " if currency else ""
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{prefix}{value / 1e9:.1f}B".replace(".0B", "B")
    if magnitude >= 1e6:
        return f"{prefix}{value / 1e6:.1f}M".replace(".0M", "M")
    if magnitude >= 1e3:
        return f"{prefix}{value / 1e3:.0f}K"
    return f"{prefix}{value:,.0f}"


def format_percent(value: float | None, missing: str = "Not Provided", decimals: int = 0) -> str:
    if value is None:
        return missing
    return f"{value:.{decimals}f}%"


def format_months(value: float | None, missing: str = "Not Provided") -> str:
    if value is None:
        return missing
    if value >= 24 and value % 12 == 0:
        return f"{value / 12:.0f} yrs"
    return f"{value:.0f} mo"
