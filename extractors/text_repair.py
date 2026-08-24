"""Tolerance for letter-spaced PDF text.

Design decks routinely set headings with wide tracking, and PDF text
extraction turns that into strings like ``"RAIS ING $2M"`` or
``"M ARKE T S IZ E"``. A keyword pattern will never match those, and the
lines involved are often the ones carrying the raise amount.

The fix is to match cues against a whitespace-free rendering of the line and
map the match position back to the original string, so values are still read
from the untouched text. Because collapsing whitespace can also create
spurious matches across word boundaries, this path is only used on lines that
actually look letter-spaced.
"""

from __future__ import annotations

import re
from functools import lru_cache

#: A line qualifies as letter-spaced when it has at least this many very short
#: alphabetic tokens and they make up a large share of the line.
_MIN_SHORT_TOKENS = 3
_SHORT_TOKEN_RATIO = 0.30


def looks_letter_spaced(line: str) -> bool:
    """Heuristic: does this line look like tracked-out heading text?"""
    tokens = [t for t in re.split(r"\s+", line.strip()) if t]
    if len(tokens) < 4:
        return False
    alpha = [t for t in tokens if t.isalpha()]
    if len(alpha) < 3:
        return False
    short = [t for t in alpha if len(t) <= 2]
    if len(short) < _MIN_SHORT_TOKENS:
        return False
    return len(short) / len(tokens) >= _SHORT_TOKEN_RATIO


def flatten_with_map(line: str) -> tuple[str, list[int]]:
    """Return the line with whitespace removed, plus index -> original index."""
    flat_chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(line):
        if not char.isspace():
            flat_chars.append(char)
            positions.append(index)
    return "".join(flat_chars), positions


#: Whitespace constructs that must be dropped to match against flattened text.
_SPACE_CONSTRUCTS = (
    (r"\[\\s-\]", "-?"),
    (r"\[\\s\\-\]", "-?"),
    (r"\\s\+", ""),
    (r"\\s\*", ""),
    (r"\\s\?", ""),
    (r"\\s", ""),
    (r"\\b", ""),
)


@lru_cache(maxsize=512)
def flatten_pattern(pattern: str) -> str:
    """Rewrite a cue regex so it can match whitespace-free text.

    Whitespace requirements become optional and word boundaries are dropped,
    since neither survives flattening. Literal spaces inside the pattern are
    removed too.
    """
    out = pattern
    for construct, replacement in _SPACE_CONSTRUCTS:
        out = re.sub(construct, replacement, out)
    # Literal spaces that were not part of an escape sequence.
    out = out.replace(" ", "")
    return out


def map_span(positions: list[int], start: int, end: int, fallback_len: int) -> tuple[int, int]:
    """Translate a span in flattened coordinates back to the original line."""
    if not positions:
        return 0, fallback_len
    start = max(0, min(start, len(positions) - 1))
    end = max(start + 1, min(end, len(positions)))
    original_start = positions[start]
    original_end = positions[end - 1] + 1
    return original_start, original_end


_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_PUNCT_SPACE = re.compile(r"\s*([–—/:|•])\s*")


def repair_spacing(line: str) -> str:
    """Best-effort readable form of a letter-spaced line.

    Flattening "CO M PE TITIO N" gives "COMPETITION"; flattening a mixed-case
    heading gives a run like "BUSINESSMODEL–Razor/RazorBlade", which is
    re-split on case transitions and punctuation. Used only for display of
    text fields - numeric values are always read from the original text.
    """
    flat, _ = flatten_with_map(line)
    if not flat:
        return line.strip()
    # A long all-caps line flattens to one unreadable run because there are no
    # case transitions to split on, so the original spacing is the better of
    # two bad options. A short one ("CO M PE TITIO N") flattens cleanly.
    if not any(ch.islower() for ch in flat) and len(flat) > 24:
        return re.sub(r"\s+", " ", line).strip()
    spaced = _PUNCT_SPACE.sub(r" \1 ", flat)
    spaced = _CAMEL_SPLIT.sub(" ", spaced)
    return re.sub(r"\s+", " ", spaced).strip()
