"""Cue patterns that locate each registry field inside deck text.

A :class:`Cue` is a keyword regex plus instructions for where the value sits
relative to it. Cues are intentionally narrow and carry an ``exclude`` guard,
because a wrong confident number is far more damaging to a financing
prediction than a missing one.

``scope`` semantics:
    line   - parse the whole line the cue matched
    after  - parse only the text following the cue match
    window - parse the cue line plus the following line, which covers the
             very common deck layout of a label above its value
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Cue:
    field: str
    pattern: str
    confidence: float
    scope: str = "line"
    exclude: str | None = None
    #: For TEXT fields, capture the matched line itself rather than a number.
    capture_text: bool = False
    #: "line" checks the exclude guard against the cue line only; "window"
    #: also checks the following line, which is what catches layouts like
    #: "Series A:" on one line and "TBD" on the next.
    exclude_scope: str = "line"

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern, re.IGNORECASE)

    def compiled_exclude(self) -> re.Pattern[str] | None:
        return re.compile(self.exclude, re.IGNORECASE) if self.exclude else None


# Guards reused across the financing cues.
_RAISED_TO_DATE = r"raised\s+to\s+date|to\s+date|previously|so\s+far|already\s+raised|since\s+inception"

#: Marks a figure as a plan rather than a fact. Applied to every current-state
#: financial cue so a target can never be reported as an achieved number.
_FORWARD_LOOKING = (
    r"\bmilestones?\b|\btargets?\b|\bgoals?\b|\bprojected\b|\bprojections?\b|\bforecast\b|"
    r"\bplan(?:ned)?\b|\bwithin\s+\d+\s+(?:months?|years?|quarters?)\b|\bby\s+20\d\d\b|"
    r"\bexit\s+year\b|\bwe\s+will\b|\bexpects?\b|\bpath\s+to\b|\brun[\s-]?rate\s+by\b"
)
_VALUATION_WORDS = r"valuation|pre-?money|post-?money|cap\b"

CUES: tuple[Cue, ...] = (
    # ------------------------------------------------------------- financing
    Cue("target_raise", r"\b(?:currently\s+)?(?:raising|seeking|target(?:ing)?\s+raise|round\s+size|the\s+ask|our\s+ask|capital\s+sought)\b", 0.92, "line", exclude=_RAISED_TO_DATE),
    Cue("target_raise", r"\b(?:seeking|raising)\s+(?:to\s+raise\s+)?(?=.*[\$€£])", 0.90, "after", exclude=_RAISED_TO_DATE),
    Cue("target_raise", r"\b(?:current\s+round|this\s+round|round\s+one|series\s+[a-c]\s+round)\b", 0.72, "line", exclude=_RAISED_TO_DATE),
    Cue("target_raise", r"\b(?:total\s+)?(?:offering|raise)\s*(?:amount|size)?\s*[:\-]", 0.85, "window", exclude=_RAISED_TO_DATE),
    Cue("target_raise", r"\bask\s*[:\-]", 0.80, "window", exclude=_RAISED_TO_DATE),

    # Window scope because tranche tables print "Closed / Tranche 1" on one
    # line and the amounts on the next.
    Cue("amount_committed", r"\b(?:committed|subscribed|soft[\s-]?circled|closed|secured|in\s+hand|hard\s+circled)\b", 0.88, "window", exclude=r"commitment\s+to\s+(?:our|the)\s+mission|committed\s+team|committed\s+to\s+(?:building|improving|delivering)|closing\s+the\s+gap"),
    Cue("amount_committed", r"\b(?:already|to\s+date)\s+(?:committed|raised\s+in\s+this\s+round)\b", 0.88, "line"),

    Cue("percent_committed", r"\b(?:\d+\s*%\s*(?:of\s+(?:the\s+)?round\s+)?(?:committed|subscribed|closed|secured|filled))", 0.93, "line"),
    Cue("percent_committed", r"\bround\s+(?:is\s+)?\d+\s*%\s*(?:committed|subscribed|closed)", 0.93, "line"),

    # "SAFE" must be the instrument, never the English word: a line reading
    # "make safe ventilation possible" is not a financing term.
    Cue("security_type", r"\bSAFE\s*(?:note|agreement|[:\-])|\bsimple\s+agreement\s+for\s+future\s+equity\b|\bconvertible\s+(?:note|debt|security)\b|\bpreferred\s+(?:stock|shares|equity)\b|\bseries\s+[a-c]\s+preferred\b|\bcommon\s+(?:stock|equity)\b|\bpriced\s+round\b|\brevenue[\s-]share\b", 0.85, "line", capture_text=True),

    Cue("pre_money_valuation", r"\bpre-?\s?money\b|\bpre\s+valuation\b", 0.94, "line"),
    Cue("post_money_valuation", r"\bpost-?\s?money\b", 0.94, "line"),
    Cue("safe_cap", r"\bsafe\b.{0,30}\bcap\b|\bcap\b.{0,20}\bsafe\b", 0.88, "line"),
    Cue("note_cap", r"\b(?:valuation\s+cap|note\s+cap|conversion\s+cap)\b", 0.90, "line"),
    Cue("note_cap", r"\bcap\s*[:\-]", 0.72, "window", exclude=r"market\s+cap|cap\s+table|capacity"),
    Cue("discount", r"\bdiscount\b", 0.88, "line", exclude=r"discount\s+rate\s+of\s+\d+%\s+(?:in\s+)?(?:dcf|npv)|discounted\s+cash"),
    Cue("interest_rate", r"\b(?:interest(?:\s+rate)?|coupon|accrue[sd]?\s+at)\b", 0.85, "line"),
    Cue("maturity_date", r"\bmaturity\b|\bmatures?\s+(?:on|in)\b", 0.88, "line"),
    Cue("minimum_investment", r"\bminimum\s+(?:investment|check|subscription|ticket)\b|\bmin\.?\s+check\b", 0.90, "line"),

    Cue("lead_investor", r"\blead\s+investor\b|\bround\s+lead\b|\blead\s+by\b|\bled\s+by\b|\bseeking\s+a?\s*lead\b", 0.88, "line", capture_text=True),
    Cue("existing_investors", r"\b(?:existing|current|prior)\s+investors?\b|\bbacked\s+by\b|\binvestors?\s+include\b|\bour\s+investors\b", 0.85, "window", capture_text=True),

    Cue("prior_funding", r"\b(?:raised\s+to\s+date|to\s+date\s+raised|previously\s+raised|already\s+raised|capital\s+raised|funding\s+raised|raised\s+since\s+inception)\b", 0.90, "line"),
    Cue("prior_funding", r"\bnon-?dilutive\s+(?:funding|capital|grants?)\b", 0.72, "line"),
    Cue("total_funding_to_date", r"\btotal\s+(?:funding|capital|raised)\b", 0.88, "line"),
    Cue("previous_round", r"\b(?:previous|last|prior)\s+round\b", 0.85, "line", capture_text=True),
    Cue("previous_round_size", r"\b(?:previous|last|prior)\s+round\b", 0.80, "line"),
    Cue("previous_round_date", r"\b(?:previous|last|prior)\s+round\b|\bclosed\s+(?:our|the)\s+\w+\s+round\b", 0.78, "line"),

    Cue("use_of_funds", r"\buse\s+of\s+(?:funds|proceeds)\b|\bproceeds\s+will\s+be\s+used\b|\bfunds?\s+will\s+be\s+used\b|\bcapital\s+allocation\b|\bwhat\s+we.{0,3}ll\s+do\s+with\s+(?:the|this)\b", 0.90, "window", capture_text=True),

    # -------------------------------------------------------------- traction
    Cue("annual_revenue", r"\b(?:annual\s+revenue|revenue\s+\(annual\)|yearly\s+revenue|revenue\s+run[\s-]?rate|trailing\s+twelve|ttm\s+revenue)\b", 0.92, "line"),
    Cue("annual_revenue", r"\brevenue\s*[:\-]", 0.78, "window", exclude=r"revenue\s+model|revenue\s+share|revenue\s+stream|projected|forecast|20\d\d[eE]"),
    # "$6M ARR within 18 months" is a milestone, not current revenue. Reading
    # a target as an achievement is the most damaging error in the whole
    # traction category, so the guard is deliberately broad.
    Cue("arr", r"\barr\b|\bannual\s+recurring\s+revenue\b", 0.94, "line", exclude=_FORWARD_LOOKING + r"|arrival|arrange|array"),
    Cue("mrr", r"\bmrr\b|\bmonthly\s+recurring\s+revenue\b", 0.94, "line", exclude=_FORWARD_LOOKING),
    Cue("revenue_growth", r"\brevenue\s+growth\b|\bgrowth\s+rate\b|\bgrew\s+\d+\s*%|\byoy\b|\byear[\s-]over[\s-]year\b", 0.85, "line", exclude=r"market\s+growth|cagr\s+of\s+the\s+market"),
    Cue("gross_margin", r"\bgross\s+margin\b|\bgm\s*[:\-]", 0.93, "line"),
    Cue("ebitda", r"\bebitda\b", 0.92, "line"),
    Cue("burn_rate", r"\b(?:burn\s+rate|monthly\s+burn|net\s+burn|cash\s+burn)\b", 0.92, "line"),
    Cue("cash_balance", r"\b(?:cash\s+(?:balance|on\s+hand|position)|cash\s+in\s+bank)\b", 0.90, "line"),
    Cue("runway_months", r"\brunway\b", 0.93, "line"),

    Cue("customers", r"\b(?:customers|clients|accounts)\b", 0.72, "line", exclude=r"customer\s+(?:acquisition|success|support|discovery|persona|segment|journey|validation|concentration)|potential\s+customers|target\s+customers"),
    Cue("paying_customers", r"\bpaying\s+(?:customers|clients|users|accounts)\b", 0.92, "line"),
    Cue("enterprise_customers", r"\benterprise\s+(?:customers|clients|accounts|logos)\b", 0.90, "line"),
    Cue("pilots", r"\bpilots?\b|\bpilot\s+programs?\b|\bpilot\s+sites?\b", 0.85, "line", exclude=r"pilot\s+(?:in\s+command|licen[cs]e)|autopilot"),
    Cue("lois", r"\blois?\b|\bletters?\s+of\s+intent\b", 0.90, "line"),
    Cue("contracts", r"\b(?:signed\s+contracts?|contracts?\s+signed|purchase\s+orders?|POs\b)\b", 0.85, "line"),
    Cue("pipeline", r"\b(?:sales\s+)?pipeline\b", 0.85, "line"),
    Cue("backlog", r"\bbacklog\b", 0.88, "line"),
    Cue("renewal_rate", r"\b(?:renewal\s+rate|net\s+revenue\s+retention|nrr\b|gross\s+retention|retention\s+rate)\b", 0.90, "line"),
    Cue("churn", r"\bchurn\b", 0.90, "line"),
    Cue("average_contract_value", r"\b(?:acv\b|average\s+contract\s+value|average\s+(?:deal|order)\s+(?:size|value)|arpu\b)", 0.90, "line"),
    Cue("sales_cycle", r"\bsales\s+cycle\b", 0.88, "line", capture_text=True),
    Cue("customer_concentration", r"\bcustomer\s+concentration\b|\blargest\s+customer\s+(?:is|represents|accounts)\b|\btop\s+customer\s+\d+\s*%", 0.88, "line"),

    # ---------------------------------------------------------------- market
    Cue("tam", r"\btam\b|\btotal\s+addressable\s+market\b|\bmarket\s+size\b|\baddressable\s+market\b", 0.92, "window", exclude=r"\bsam\b|\bsom\b"),
    Cue("sam", r"\bsam\b|\bserviceable\s+(?:available|addressable)\s+market\b", 0.92, "window", exclude=r"\bsamples?\b|\bsame\b"),
    Cue("som", r"\bsom\b|\bserviceable\s+obtainable\s+market\b|\bobtainable\s+market\b", 0.90, "window", exclude=r"\bsome\b"),
    Cue("market_growth_rate", r"\bcagr\b|\bmarket\s+(?:is\s+)?growing\b|\bmarket\s+growth\b", 0.88, "line"),
    Cue("competitive_environment", r"\bcompetiti(?:on|ve|ors)\b|\bversus\s+competitors\b|\blandscape\b", 0.72, "line", capture_text=True),
    Cue("primary_geography", r"\b(?:primary\s+market|initial\s+market|go[\s-]to[\s-]market\s+geography|launch\s+market)\b", 0.80, "line", capture_text=True),
    Cue("market_timing", r"\bwhy\s+now\b|\bmarket\s+timing\b|\btailwinds?\b|\binflection\s+point\b", 0.82, "window", capture_text=True),

    # ------------------------------------------------------------------ team
    Cue("employees", r"\b(?:employees|headcount|team\s+size|full[\s-]time\s+(?:employees|staff)|ftes?\b)", 0.88, "line"),
    Cue("founder_experience", r"\b(?:previously|prior(?:ly)?\s+at|formerly|\d+\+?\s+years?\s+(?:of\s+)?experience|veteran\s+of)\b", 0.75, "line", capture_text=True),
    # An advisor with a prior exit is not a founder with a prior exit, and the
    # two carry very different financing weight.
    Cue("prior_exits", r"\b(?:exited|prior\s+exit|successful\s+exit|acquired\s+by|sold\s+(?:his|her|their)\s+(?:company|business)|ipo(?:ed)?)\b", 0.85, "line", exclude=r"\badvisor(?:y)?\b|\badviser\b|\bconsultant\b|\bboard\s+member\b", capture_text=True),
    Cue("domain_expertise", r"\b(?:domain\s+expert(?:ise)?|deep\s+expertise|subject\s+matter\s+expert|\bmd\b|\bphd\b)", 0.75, "line", capture_text=True),
    Cue("technical_expertise", r"\b(?:cto\b|chief\s+technology|head\s+of\s+engineering|vp\s+engineering|chief\s+scientific|cso\b)", 0.82, "line", capture_text=True),
    Cue("commercial_expertise", r"\b(?:cro\b|chief\s+revenue|chief\s+commercial|vp\s+(?:sales|commercial)|head\s+of\s+sales|ccо?\b)", 0.82, "line", capture_text=True),
    Cue("founder_full_time", r"\bfull[\s-]?time\b|\bpart[\s-]?time\b|\bfractional\b", 0.85, "line", capture_text=True),
    Cue("key_hires", r"\b(?:key\s+hires?|recent\s+hires?|planned\s+hires?|hiring\s+plan)\b", 0.85, "window", capture_text=True),
    Cue("missing_roles", r"\b(?:open\s+roles?|gaps?\s+in\s+the\s+team|to\s+be\s+hired|tbh\b|recruiting\s+for)\b", 0.82, "window", capture_text=True),

    # --------------------------------------------------------------- product
    Cue("product_stage", r"\b(?:product\s+stage|prototype|beta|pilot\s+stage|commercial(?:ly)?\s+available|in\s+market|pre[\s-]revenue|mvp\b|proof\s+of\s+concept|poc\b|trl\s*\d)", 0.82, "line", capture_text=True),
    Cue("mvp_status", r"\bmvp\b|\bminimum\s+viable\s+product\b", 0.88, "line", capture_text=True),
    Cue("commercial_product_status", r"\b(?:commercial(?:ly)?\s+(?:available|launch|ready|stage)|first\s+revenue|generally\s+available|shipping)\b", 0.85, "line", capture_text=True),
    Cue("regulatory_stage", r"\b(?:fda|510\(?k\)?|de\s+novo|pma\b|ce\s+mark|ukca|regulatory\s+(?:pathway|stage|strategy)|ind\b|nda\b|breakthrough\s+device)\b", 0.88, "line", capture_text=True),
    # "FDA approvals" in an advisor's biography is experience, not a company
    # approval, and reading it as one would materially inflate the score.
    # Requires an achieved status, not the bare word "approval": a bullet
    # reading "FDA approvals" in a biography, or a goal such as "to reach FDA
    # approval", must not register as an approval the company holds.
    Cue("regulatory_approvals", r"\bfda\s+(?:cleared|approved)\b|\bce\s+marked\b|\b(?:received|granted|obtained|secured)\s+(?:fda\s+)?(?:clearance|approval)\b|\b510\(?k\)?\s+clear(?:ed|ance\s+(?:granted|received))\b", 0.90, "line", exclude=r"\badvisor(?:y)?\b|\badviser\b|\bexperience\b|\bexpertise\b|\bbackground\b|\byears?\s+of\b|\bto\s+reach\b|\bpath(?:way)?\s+to\b|\bseeking\b|\bonce\b|\bupon\b|\bafter\b|\bwhen\b|\binterested\b|\bplans?\b|\bintends?\b|\bwill\b|\bwould\b|\bcould\b|\bexpects?\b|\btargets?\b|\bgoal\b|\bmilestone\b", capture_text=True),
    Cue("clinical_stage", r"\b(?:preclinical|pre-?clinical|phase\s+(?:i{1,3}|[1-3])\b|clinical\s+trial|first[\s-]in[\s-]human|\bIDE\s+(?:study|trial|submission|approval))", 0.88, "line", capture_text=True),
    Cue("intellectual_property", r"\b(?:intellectual\s+property|\bip\s+(?:portfolio|strategy|position)|trade\s+secret|exclusive\s+licen[cs]e)\b", 0.85, "window", capture_text=True),
    Cue("patents", r"\bpatents?\b|\bpatent\s+(?:pending|granted|filed|application)\b|\bpct\b|\buspto\b", 0.88, "line", capture_text=True),
    Cue("partnerships", r"\b(?:partnerships?|strategic\s+partner|distribution\s+(?:partner|agreement)|collaboration\s+with|mou\b|joint\s+development)\b", 0.85, "window", capture_text=True),
    Cue("customer_validation", r"\b(?:customer\s+validation|testimonial|kol\b|key\s+opinion\s+leader|letters?\s+of\s+support|case\s+stud(?:y|ies))\b", 0.85, "window", capture_text=True),
    Cue("technology_readiness", r"\btrl\s*\d|\btechnology\s+readiness\b", 0.90, "line", capture_text=True),

    # --------------------------------------------------------------- company
    # The lookbehind stops the domain half of "spencer@acme.com" being read as
    # the company website.
    Cue("website", r"(?<![@\w.])(?:https?://|www\.)[\w.-]+\.[a-z]{2,}\b|(?<![@\w.])[\w-]+\.(?:com|io|ai|co|net|org|health|bio|tech)\b", 0.90, "line", capture_text=True),
    Cue("headquarters", r"\b(?:headquarter(?:s|ed)|based\s+in|hq\b|located\s+in|offices?\s+in)\b", 0.85, "line", capture_text=True),
    Cue("business_model", r"\b(?:business\s+model|revenue\s+model|how\s+we\s+(?:make\s+money|earn)|monetis?z?ation|pricing\s+model|subscription\s+model|razor[\s-]and[\s-]blade)\b", 0.88, "window", capture_text=True),
    # A roadmap entry such as "Series A: TBD" describes a future round, not the
    # round currently being raised.
    Cue("stage_declared", r"\b(?:pre-?seed|seed\s+(?:round|stage|financing)|series\s+[a-f]\b|bridge\s+round|growth\s+round|friends\s+and\s+family)\b", 0.85, "line", exclude=r"\btbd\b|\bto\s+be\s+determined\b|\bnext\s+round\b|\bfuture\s+round\b", capture_text=True, exclude_scope="window"),
    Cue("industry", r"\b(?:industry|vertical|sector)\s*[:\-]", 0.82, "after", capture_text=True),
    # "available in" alone is far too loose - it matches "available in the data
    # room" - so a place name must follow it. The trailing alternative is
    # wrapped in (?-i:...) because these cues compile with IGNORECASE, which
    # would otherwise let [A-Z] match any lower-case word.
    Cue("geographic_market", r"\b(?:target\s+(?:geograph|market\s+region)|markets?\s+served|operating\s+in|(?:available|launched|selling)\s+in)\s+(?:the\s+)?(?:US|USA|UK|EU|EMEA|APAC|LATAM|MENA|North\s+America|South\s+America|Europe|Asia|Africa|Australia|Canada|Germany|France|India|Japan|China|(?-i:[A-Z][a-z]{3,}))\b", 0.78, "line", capture_text=True),
)


#: Grouped for fast lookup during extraction.
CUES_BY_FIELD: dict[str, list[Cue]] = {}
for _cue in CUES:
    CUES_BY_FIELD.setdefault(_cue.field, []).append(_cue)


# ---------------------------------------------------------------------------
# Standalone high-precision patterns applied to the whole document. These do
# not need a cue word nearby because their shape is unambiguous on its own.
# ---------------------------------------------------------------------------
#: ``(field, pattern, confidence, exclude_or_None)``. The exclude guard is
#: checked against the whole line, exactly as it is for cues.
STANDALONE_PATTERNS: tuple[tuple[str, str, float, str | None], ...] = (
    ("target_raise", r"(?:raising|seeking)\s+(?:a\s+)?[\$€£]\s?[\d.,]+\s*(?:k|m|mm|bn|b|million|billion|thousand)?", 0.90, _RAISED_TO_DATE),
    ("target_raise", r"[\$€£]\s?[\d.,]+\s*(?:k|m|mm|million)?\s+(?:seed|series\s+[a-c]|pre-?seed|bridge)\s+round", 0.88, _RAISED_TO_DATE),
    ("arr", r"[\$€£]\s?[\d.,]+\s*(?:k|m|mm|million)?\s+(?:in\s+)?arr\b", 0.94, _FORWARD_LOOKING),
    ("mrr", r"[\$€£]\s?[\d.,]+\s*(?:k|m|mm|million)?\s+(?:in\s+)?mrr\b", 0.94, _FORWARD_LOOKING),
    ("runway_months", r"\b\d+(?:\.\d+)?\s*(?:months?|mos?)\s+(?:of\s+)?runway\b", 0.93, None),
    ("runway_months", r"\brunway\s+(?:of\s+|to\s+|into\s+)?\d+(?:\.\d+)?\s*(?:months?|mos?|years?)\b", 0.93, None),
    ("percent_committed", r"\b\d{1,3}\s*%\s+(?:of\s+(?:the\s+)?round\s+)?(?:is\s+)?(?:committed|subscribed|closed|secured|spoken\s+for)\b", 0.93, None),
    ("gross_margin", r"\b\d{1,3}\s*%\s+gross\s+margins?\b", 0.93, _FORWARD_LOOKING),
    ("employees", r"\b\d{1,4}\s+(?:full[\s-]time\s+)?(?:employees|ftes?|staff)\b", 0.90, _FORWARD_LOOKING),
    ("customers", r"\b\d{1,6}\s+(?:paying\s+)?(?:customers|clients|accounts)\b", 0.85, _FORWARD_LOOKING),
    ("pilots", r"\b\d{1,4}\s+(?:active\s+)?pilots?\b", 0.88, _FORWARD_LOOKING),
    ("lois", r"\b\d{1,4}\s+(?:signed\s+)?(?:lois?|letters?\s+of\s+intent)\b", 0.90, _FORWARD_LOOKING),
)
