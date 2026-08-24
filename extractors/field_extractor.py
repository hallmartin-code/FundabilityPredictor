"""Rule-based field extraction: deck text in, evidenced field values out.

The extractor collects *candidates* from every page, then resolves each field
to a single best value. Resolution prefers the highest-confidence candidate
and, on ties, the value the most pages agree on. Nothing is invented: a field
with no candidate stays MISSING.

Three real-world hazards are handled explicitly, because each one silently
produces a *wrong* number rather than a missing one:

* Letter-spaced headings ("RAIS ING $2M") - matched via a flattened variant.
* Forecast tables ("Revenue - 2,430 10,620 ...") - suppressed, so a 2029
  projection is never reported as current revenue.
* Values printed above their label - the window scope looks up as well as down.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from extractors.base import ExtractedDocument, Page
from extractors.field_patterns import CUES, STANDALONE_PATTERNS, Cue
from extractors.normalization import (
    ValueSpan,
    clean_text,
    find_money,
    find_months,
    find_number,
    find_percent,
    nearest_span,
    parse_bool,
    parse_date,
    parse_list,
    parse_money,
    parse_months,
    parse_number,
    parse_percent,
)
from extractors.text_repair import (
    flatten_pattern,
    flatten_with_map,
    looks_letter_spaced,
    map_span,
    repair_spacing,
)
from models.evidence import ValueStatus
from models.fields import FIELDS_BY_NAME, FieldKind
from models.profile import DeckProfile

LOGGER = logging.getLogger(__name__)

#: Plausibility bounds per field kind. A mis-parsed number is worse than a
#: missing one, so anything outside these ranges is discarded.
MONEY_BOUNDS = (1_000.0, 5e11)
COUNT_BOUNDS = (0.0, 5e7)
MONTH_BOUNDS = (0.0, 240.0)
PERCENT_BOUNDS = (0.0, 100.0)

#: Fields where a percentage above 100 is legitimate (growth can be 400%).
UNBOUNDED_PERCENT_FIELDS = frozenset({"revenue_growth", "market_growth_rate"})

#: Shortest de-spaced match worth trusting. Below this, collapsing whitespace
#: manufactures coincidental hits across word boundaries.
MIN_FLAT_MATCH_CHARS = 6

#: Confidence multiplier when the value sits before its cue rather than after
#: it. Decks label metrics as "Gross margin: 82%"; a number appearing before
#: the same words is usually part of a sentence about something else.
PRECEDING_VALUE_PENALTY = 0.88

#: Current-state financial fields, which must never be filled from a forecast.
FORECAST_SENSITIVE_FIELDS = frozenset(
    {
        "annual_revenue", "arr", "mrr", "ebitda", "gross_margin",
        "burn_rate", "cash_balance", "revenue_growth", "customers",
        "paying_customers", "average_contract_value",
    }
)

FORECAST_MARKERS = re.compile(
    r"\$\s?000s|\bprojection|\bforecast|\bpro\s*forma|\bP&L\b|\bprojected\b|"
    r"\bfinancial\s+model\b|\bplan\s+vs\b",
    re.IGNORECASE,
)

EMAIL_RE = re.compile(r"\b([\w.+-]+)@([\w-]+\.[\w.-]+)\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b")

#: Words that must never be taken as a company name.
_NAME_STOPWORDS = frozenset(
    """the and for our with from that this what who why how are was were will
    company deck pitch confidential overview summary investor investors round
    seed series problem solution market team traction product technology
    financials appendix contact thank thanks you your all inc llc ltd corp
    version final draft copy update updated new presentation slides slide
    january february march april may june july august september october
    november december""".split()
)

#: Capitalised words that look like a person's name but are job titles or
#: business nouns. Without this, "Clinical Advisor" becomes a founder.
_NON_NAME_WORDS = frozenset(
    """advisor advisory adviser director officer chief executive president vice
    engineering medical clinical technical regulatory business commercial
    investor investment manager management operations research development
    successful exit patent patents portfolio market markets size revenue team
    board committee founder founders co-founder partner partners science
    scientific sales marketing finance financial legal quality strategy
    strategic global national international university hospital institute
    center centre group holdings ventures capital fund forum network solutions
    systems technologies technology therapeutics diagnostics medical devices
    inc llc ltd corp gmbh limited company due diligence completed contact
    information problem solution product roadmap traction summary overview""".split()
)

#: Filename noise stripped when guessing a company name from the file name.
_FILENAME_NOISE = re.compile(
    r"\b(deck|pitch|pitchdeck|presentation|investor|investors|final|draft|copy|"
    r"v\d+(?:\.\d+)*|version|confidential|update[d]?|teaser|overview|summary|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|20\d\d)\b",
    re.IGNORECASE,
)

_PERSON_NAME_RE = re.compile(
    r"\b([A-Z][a-z]{1,15})\s+([A-Z][a-z'’-]{1,20})\b"
    r"(?:,\s*(?:MS|MSc|PhD|Ph\.D\.|MD|M\.D\.|MBA|BME|BSc|BS|BA|JD|DVM|RN|PharmD|DPhil))?"
)
_FOUNDER_TITLE_RE = re.compile(r"co-?founder|founding\s+(?:team|partner)|\bfounder\b", re.IGNORECASE)
_CEO_TITLE_RE = re.compile(r"\bCEO\b|chief\s+executive", re.IGNORECASE)


@dataclass
class Candidate:
    field: str
    value: Any
    confidence: float
    page: int
    text: str
    status: ValueStatus = ValueStatus.EXPLICIT


class FieldExtractor:
    """Extract every registry field it can evidence from a document."""

    def __init__(self, max_source_text: int = 220) -> None:
        self.max_source_text = max_source_text
        self._cues: list[tuple[Cue, re.Pattern[str], re.Pattern[str] | None, re.Pattern[str]]] = []
        for cue in CUES:
            self._cues.append(
                (
                    cue,
                    cue.compiled(),
                    cue.compiled_exclude(),
                    re.compile(flatten_pattern(cue.pattern), re.IGNORECASE),
                )
            )
        self._standalone = [
            (
                field,
                re.compile(pattern, re.IGNORECASE),
                conf,
                re.compile(exclude, re.IGNORECASE) if exclude else None,
            )
            for field, pattern, conf, exclude in STANDALONE_PATTERNS
        ]

    # ------------------------------------------------------------------- api
    def extract(self, document: ExtractedDocument) -> DeckProfile:
        profile = DeckProfile(
            source=document.source,
            extraction_quality=document.quality(),
            raw_text_by_page=document.text_by_page(),
        )

        candidates: list[Candidate] = []
        for page in document.pages:
            candidates.extend(self._candidates_from_page(page))
        candidates.extend(self._people_candidates(document))
        candidates.extend(self._company_name_candidates(document))

        for field, best in self._resolve(candidates).items():
            profile.put(
                field,
                best.value,
                best.status,
                round(min(best.confidence, 0.99), 3),
                source_page=best.page,
                source_text=clean_text(best.text, max_length=self.max_source_text),
            )

        self._derive_website(profile, document)

        LOGGER.info(
            "Extracted %d of %d fields from %s",
            len(profile.known_fields()),
            len(FIELDS_BY_NAME),
            document.source.filename,
        )
        return profile

    # ------------------------------------------------------------- candidates
    def _candidates_from_page(self, page: Page) -> list[Candidate]:
        out: list[Candidate] = []
        lines = [ln.strip() for ln in page.searchable_text().splitlines()]
        lines = [ln for ln in lines if ln]
        is_forecast = _is_forecast_page(page)

        for index, line in enumerate(lines):
            previous = lines[index - 1] if index > 0 else ""
            following = lines[index + 1] if index + 1 < len(lines) else ""
            out.extend(self._cue_candidates(line, previous, following, page.number, is_forecast))
            out.extend(self._standalone_candidates(line, page.number, is_forecast))
        return out

    def _cue_candidates(
        self, line: str, previous: str, following: str, page_no: int, is_forecast: bool
    ) -> list[Candidate]:
        out: list[Candidate] = []
        spaced = looks_letter_spaced(line)
        flat, positions = flatten_with_map(line) if spaced else ("", [])

        for cue, pattern, exclude, flat_pattern in self._cues:
            if is_forecast and cue.field in FORECAST_SENSITIVE_FIELDS:
                continue
            if exclude is not None:
                guard_text = f"{line} {following}" if cue.exclude_scope == "window" else line
                if exclude.search(guard_text):
                    continue

            match_span: tuple[int, int] | None = None
            penalty = 1.0
            match = pattern.search(line)
            if match:
                match_span = (match.start(), match.end())
            elif spaced:
                flat_match = flat_pattern.search(flat)
                # Collapsing whitespace can manufacture short matches across
                # word boundaries ("...as a fee" -> "safe"), so only trust a
                # de-spaced hit that is substantial.
                if flat_match and (flat_match.end() - flat_match.start()) >= MIN_FLAT_MATCH_CHARS:
                    match_span = map_span(positions, flat_match.start(), flat_match.end(), len(line))
                    penalty = 0.95
            if match_span is None:
                continue

            spec = FIELDS_BY_NAME[cue.field]
            if cue.capture_text or spec.kind in (FieldKind.TEXT, FieldKind.LIST):
                value = self._capture_text(cue, spec.kind, line, following, match_span)
                if value:
                    out.append(
                        Candidate(cue.field, value, cue.confidence * penalty, page_no, line)
                    )
                continue

            value, scope_penalty, source = self._capture_value(
                cue, spec.kind, line, previous, following, match_span
            )
            if value is not None and self._is_plausible(cue.field, spec.kind, value):
                out.append(
                    Candidate(
                        cue.field,
                        value,
                        cue.confidence * penalty * scope_penalty,
                        page_no,
                        source,
                    )
                )
        return out

    def _standalone_candidates(self, line: str, page_no: int, is_forecast: bool) -> list[Candidate]:
        out: list[Candidate] = []
        for field, pattern, confidence, exclude in self._standalone:
            if is_forecast and field in FORECAST_SENSITIVE_FIELDS:
                continue
            if exclude is not None and exclude.search(line):
                continue
            match = pattern.search(line)
            if not match:
                continue
            spec = FIELDS_BY_NAME[field]
            value = _PARSERS[spec.kind](match.group(0))
            if isinstance(value, tuple):  # money returns (amount, currency)
                value = value[0]
            if value is not None and self._is_plausible(field, spec.kind, value):
                out.append(Candidate(field, value, confidence, page_no, line))
        return out

    # ----------------------------------------------------------- value pickup
    def _capture_value(
        self,
        cue: Cue,
        kind: FieldKind,
        line: str,
        previous: str,
        following: str,
        span: tuple[int, int],
    ) -> tuple[Any, float, str]:
        """Find the value nearest the cue, widening to neighbouring lines."""
        finder = _FINDERS.get(kind)
        prefer_strong = kind is FieldKind.MONEY
        cue_start, cue_end = span

        if finder is None:
            # DATE and BOOL have no positional form; parse the whole line.
            parser = _PARSERS[kind]
            value = parser(line)
            if value is None and cue.scope == "window" and following:
                return parser(following), 0.85, following
            return value, 1.0, line

        if cue.scope == "after":
            spans = finder(line[cue_end:])
            chosen = _preferred(spans, prefer_strong)
            if chosen is not None:
                return chosen.value, 1.0, line
        else:
            chosen = nearest_span(finder(line), cue_start, cue_end, prefer_strong=prefer_strong)
            if chosen is not None and (chosen.strong or not prefer_strong):
                # "Gross margin: 82%" states the company's metric;
                # "distributors lose 12% of gross margin" describes the
                # problem. Both match the same cue, so the label form - value
                # after the cue - is trusted more.
                after_cue = chosen.start >= cue_end
                return chosen.value, 1.0 if after_cue else PRECEDING_VALUE_PENALTY, line

        if cue.scope == "window":
            # Decks very often print the number above its label
            # ("$3.5B" on one line, "TAM = ..." on the next), so look up
            # before looking down.
            for neighbour, note in ((previous, "above"), (following, "below")):
                if not neighbour:
                    continue
                chosen = _preferred(finder(neighbour), prefer_strong)
                if chosen is not None:
                    return chosen.value, 0.85, f"{neighbour} [{note}] {line}"

        # Last resort on the cue line: accept a weak (unit-less) money value.
        if prefer_strong and cue.scope != "after":
            chosen = nearest_span(finder(line), cue_start, cue_end)
            if chosen is not None:
                return chosen.value, 0.75, line
        return None, 1.0, line

    def _capture_text(
        self, cue: Cue, kind: FieldKind, line: str, following: str, span: tuple[int, int]
    ) -> Any:
        cue_start, cue_end = span
        if cue.field == "website":
            # The cue pattern *is* the URL, so take the match itself rather
            # than the whole line it sits on.
            return clean_text(line[cue_start:cue_end].strip(" .,;:"))
        if cue.scope == "after":
            raw = line[cue_end:]
        elif cue.scope == "window" and len(line) < cue_end + 4 and following:
            # The cue is a bare label ("Use of Funds"); the value is below it.
            raw = following
        else:
            raw = line

        if kind is FieldKind.LIST:
            return parse_list(raw)

        # A tracked-out heading reads as "CO M PE TITIO N"; prefer a clean
        # neighbouring line, and otherwise repair the spacing for display.
        if looks_letter_spaced(raw):
            if following and not looks_letter_spaced(following):
                raw = following
            else:
                raw = repair_spacing(raw)

        cleaned = clean_text(raw, max_length=self.max_source_text)
        if not cleaned or len(cleaned) < 3:
            return None
        # Contact details are never the answer to a business question.
        if cue.field != "website" and EMAIL_RE.search(cleaned):
            return None
        if PHONE_RE.fullmatch(cleaned.strip()):
            return None
        return cleaned

    # ------------------------------------------------------------ plausibility
    @staticmethod
    def _is_plausible(field: str, kind: FieldKind, value: Any) -> bool:
        if not isinstance(value, (int, float)):
            return True
        if kind is FieldKind.MONEY:
            return MONEY_BOUNDS[0] <= value <= MONEY_BOUNDS[1]
        if kind is FieldKind.NUMBER:
            return COUNT_BOUNDS[0] <= value <= COUNT_BOUNDS[1]
        if kind is FieldKind.MONTHS:
            return MONTH_BOUNDS[0] < value <= MONTH_BOUNDS[1]
        if kind is FieldKind.PERCENT:
            if field in UNBOUNDED_PERCENT_FIELDS:
                return 0.0 <= value <= 100_000.0
            return PERCENT_BOUNDS[0] <= value <= PERCENT_BOUNDS[1]
        return True

    # --------------------------------------------------------------- resolve
    @staticmethod
    def _resolve(candidates: list[Candidate]) -> dict[str, Candidate]:
        """Pick one candidate per field: highest confidence, then consensus."""
        grouped: dict[str, list[Candidate]] = {}
        for cand in candidates:
            grouped.setdefault(cand.field, []).append(cand)

        resolved: dict[str, Candidate] = {}
        for field, group in grouped.items():
            top = max(c.confidence for c in group)
            shortlist = [c for c in group if c.confidence >= top - 0.02]
            if len(shortlist) > 1:
                counts = Counter(_hashable(c.value) for c in shortlist)
                winner_value, winner_count = counts.most_common(1)[0]
                if winner_count > 1:
                    shortlist = [c for c in shortlist if _hashable(c.value) == winner_value]
            best = min(shortlist, key=lambda c: (-c.confidence, c.page))
            # Corroboration means the deck says it in more than one *place*.
            # Counting candidates instead would reward a value that merely
            # matched two different rules on the same line.
            agreement = len(
                {c.page for c in group if _hashable(c.value) == _hashable(best.value)}
            )
            if agreement > 1:
                best = Candidate(
                    best.field,
                    best.value,
                    min(best.confidence + 0.04 * (agreement - 1), 0.99),
                    best.page,
                    best.text,
                    best.status,
                )
            resolved[field] = best
        return resolved

    # ----------------------------------------------------------------- people
    def _people_candidates(self, document: ExtractedDocument) -> list[Candidate]:
        """Read founder and CEO names from name/title pairs.

        Team slides put names on one line and titles on the next, so a name is
        only accepted when a founder or CEO title sits on the same or an
        adjacent line.
        """
        out: list[Candidate] = []
        for page in document.pages:
            lines = [ln.strip() for ln in page.searchable_text().splitlines() if ln.strip()]
            for index, line in enumerate(lines):
                names = _person_names(line)
                if not names:
                    continue
                context = " ".join(
                    lines[max(0, index - 1): min(len(lines), index + 2)]
                )
                if _FOUNDER_TITLE_RE.search(context):
                    confidence = min(0.80 + 0.03 * len(names), 0.95)
                    out.append(Candidate("founders", names, confidence, page.number, line))
                if _CEO_TITLE_RE.search(context):
                    # With several names on the line, the CEO is the one whose
                    # own line or immediate neighbour carries the CEO title.
                    ceo = _closest_name_to_title(lines, index, _CEO_TITLE_RE) or names[0]
                    out.append(Candidate("ceo", ceo, 0.86, page.number, line))
        return out

    # ---------------------------------------------------------- company name
    def _company_name_candidates(self, document: ExtractedDocument) -> list[Candidate]:
        """Guess the company name from the cover slide, then the filename.

        Both signals are weak alone, so agreement between them is what
        produces a high-confidence value.
        """
        out: list[Candidate] = []
        from_file = _name_from_filename(document.source.filename)
        cover_names = _name_candidates_from_cover(document.pages[:2])
        frequency = _token_frequency(document)

        if from_file:
            confidence = 0.62
            if any(from_file.lower() == c.lower() for c in cover_names):
                confidence = 0.94
            elif frequency.get(from_file.lower(), 0) >= 3:
                confidence = 0.88
            out.append(
                Candidate(
                    "company_name", from_file, confidence, 1,
                    f"Filename: {document.source.filename}",
                )
            )

        for name in cover_names:
            repeats = frequency.get(name.lower(), 0)
            out.append(
                Candidate("company_name", name, 0.70 + min(repeats, 6) * 0.03, 1, name)
            )
        return out

    # --------------------------------------------------------------- derived
    @staticmethod
    def _derive_website(profile: DeckProfile, document: ExtractedDocument) -> None:
        """Reconcile the website against the company name.

        Decks cite other organisations' URLs (an investor, a patent office, a
        cited study), and any of them can win a naive first-match. A domain
        that echoes the company name is far more likely to be the company's
        own, and a contact email domain beats an unrelated URL entirely.
        """
        company = profile.text("company_name")
        current = profile.text("website")
        if current and _domain_matches_company(current, company):
            return

        for page in document.pages:
            for match in EMAIL_RE.finditer(page.searchable_text()):
                domain = match.group(2).lower()
                if domain.split(".")[0] in _PUBLIC_EMAIL_HOSTS:
                    continue
                if company and not _domain_matches_company(domain, company):
                    continue
                profile.put(
                    "website",
                    domain,
                    ValueStatus.ESTIMATED,
                    0.60,
                    source_page=page.number,
                    source_text=match.group(0),
                    note="Inferred from the contact email domain; not stated as a website.",
                )
                return

        if current and company:
            # Keep the value, but say plainly that it may not be this company.
            existing = profile.get("website")
            existing.note = (
                "Domain does not match the company name; it may belong to a "
                "third party cited in the deck."
            )
            existing.confidence = min(existing.confidence, 0.40)
            profile.set(existing)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PUBLIC_EMAIL_HOSTS = frozenset(
    {"gmail", "outlook", "hotmail", "yahoo", "icloud", "proton", "protonmail", "aol", "me"}
)


def _domain_matches_company(domain: str, company: str) -> bool:
    """Does this URL/domain plausibly belong to the named company?"""
    if not company:
        return True  # nothing to contradict
    host = re.sub(r"^https?://", "", domain.lower()).split("/")[0]
    host_core = re.sub(r"[^a-z0-9]", "", host.split(".")[0].removeprefix("www"))
    company_core = re.sub(r"[^a-z0-9]", "", company.lower())
    if not host_core or not company_core:
        return False
    return host_core in company_core or company_core in host_core


def _preferred(spans: list[ValueSpan], prefer_strong: bool) -> ValueSpan | None:
    if not spans:
        return None
    if prefer_strong:
        strong = [s for s in spans if s.strong]
        return strong[0] if strong else None
    return spans[0]


def _is_forecast_page(page: Page) -> bool:
    """Detect a projections page, whose numbers are future, not current."""
    text = page.searchable_text()
    years = {int(y) for y in re.findall(r"\b(20[2-5]\d)\b", text)}
    future_years = {y for y in years if y >= date.today().year}
    if len(future_years) >= 3:
        return True
    return bool(FORECAST_MARKERS.search(text)) and len(years) >= 2


def _person_names(line: str) -> list[str]:
    names: list[str] = []
    for match in _PERSON_NAME_RE.finditer(line):
        first, last = match.group(1), match.group(2)
        if first.lower() in _NON_NAME_WORDS or last.lower() in _NON_NAME_WORDS:
            continue
        if first.lower() in _NAME_STOPWORDS or last.lower() in _NAME_STOPWORDS:
            continue
        full = f"{first} {last}"
        if full not in names:
            names.append(full)
    return names[:8]


def _closest_name_to_title(lines: list[str], index: int, title: re.Pattern[str]) -> str | None:
    """The name on whichever nearby line the title actually sits next to."""
    for offset in (0, 1, -1):
        pos = index + offset
        if 0 <= pos < len(lines) and title.search(lines[pos]):
            for probe in (pos, pos - 1, pos + 1):
                if 0 <= probe < len(lines):
                    found = _person_names(lines[probe])
                    if found:
                        return found[0]
    return None


def _money_only(text: str) -> float | None:
    return parse_money(text)[0]


_PARSERS: dict[FieldKind, Callable[[str], Any]] = {
    FieldKind.MONEY: _money_only,
    FieldKind.NUMBER: parse_number,
    FieldKind.PERCENT: parse_percent,
    FieldKind.MONTHS: parse_months,
    FieldKind.DATE: parse_date,
    FieldKind.BOOL: parse_bool,
    FieldKind.TEXT: lambda t: clean_text(t),
    FieldKind.LIST: parse_list,
}

_FINDERS: dict[FieldKind, Callable[[str], list[ValueSpan]]] = {
    FieldKind.MONEY: find_money,
    FieldKind.NUMBER: find_number,
    FieldKind.PERCENT: find_percent,
    FieldKind.MONTHS: find_months,
}


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, float):
        return round(value, 4)
    return value


def _name_from_filename(filename: str) -> str | None:
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename)
    stem = re.sub(r"\([^)]*\)", " ", stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = _FILENAME_NOISE.sub(" ", stem)
    stem = re.sub(r"\b\d+(\.\d+)*\b", " ", stem)
    cleaned = clean_text(stem, max_length=60)
    if not cleaned:
        return None
    words = [w for w in cleaned.split() if w.lower() not in _NAME_STOPWORDS]
    if not words:
        return None
    return " ".join(words[:4])


def _name_candidates_from_cover(pages: list[Page]) -> list[str]:
    """Short, title-cased lines on the cover are usually the company name."""
    out: list[str] = []
    for page in pages:
        candidate_lines: list[str] = []
        if page.title:
            candidate_lines.append(page.title)
        candidate_lines.extend(page.text.splitlines()[:6])
        for raw in candidate_lines:
            # Cover titles are the most likely place for wide letter spacing,
            # and "H e l i x  B i o w o r k s" must still yield the name.
            if looks_letter_spaced(raw):
                raw = repair_spacing(raw)
            line = clean_text(raw, max_length=60)
            if not line or len(line) > 45:
                continue
            words = line.split()
            if not 1 <= len(words) <= 4:
                continue
            if all(w.lower() in _NAME_STOPWORDS for w in words):
                continue
            if not any(w[:1].isupper() for w in words if w):
                continue
            if re.search(r"[.!?]$|^\d|https?://|@", line):
                continue
            if line not in out:
                out.append(line)
    return out[:4]


def _token_frequency(document: ExtractedDocument) -> dict[str, int]:
    """How often each capitalised token or bigram appears across the deck."""
    counts: Counter[str] = Counter()
    for page in document.pages:
        seen_on_page: set[str] = set()
        tokens = re.findall(r"\b[A-Z][A-Za-z0-9&'.]{2,}\b", page.searchable_text())
        for i, token in enumerate(tokens):
            if token.lower() in _NAME_STOPWORDS:
                continue
            seen_on_page.add(token.lower())
            if i + 1 < len(tokens):
                seen_on_page.add(f"{token} {tokens[i + 1]}".lower())
        counts.update(seen_on_page)
    return dict(counts)
