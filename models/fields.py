"""The canonical field registry.

Extraction, normalisation, overrides, scoring and JSON output all key off this
single list, so adding a new field is a one-line change here plus a pattern in
``extractors/field_patterns.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum


class FieldKind(str, Enum):
    """How a raw string should be normalised into a Python value."""

    MONEY = "money"
    NUMBER = "number"
    PERCENT = "percent"
    MONTHS = "months"
    TEXT = "text"
    LIST = "list"
    BOOL = "bool"
    DATE = "date"


class FieldCategory(str, Enum):
    COMPANY = "company"
    FINANCING = "financing"
    TRACTION = "traction"
    MARKET = "market"
    TEAM = "team"
    PRODUCT = "product"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    category: FieldCategory
    kind: FieldKind
    #: Fields a founder-facing financing prediction genuinely depends on.
    critical: bool = False
    aliases: tuple[str, ...] = dc_field(default_factory=tuple)


def _f(
    name: str,
    label: str,
    category: FieldCategory,
    kind: FieldKind,
    critical: bool = False,
    aliases: tuple[str, ...] = (),
) -> FieldSpec:
    return FieldSpec(name, label, category, kind, critical, aliases)


C, F, T, M, TM, P = (
    FieldCategory.COMPANY,
    FieldCategory.FINANCING,
    FieldCategory.TRACTION,
    FieldCategory.MARKET,
    FieldCategory.TEAM,
    FieldCategory.PRODUCT,
)
MONEY, NUM, PCT, MON, TXT, LST, BOOL, DATE = (
    FieldKind.MONEY,
    FieldKind.NUMBER,
    FieldKind.PERCENT,
    FieldKind.MONTHS,
    FieldKind.TEXT,
    FieldKind.LIST,
    FieldKind.BOOL,
    FieldKind.DATE,
)


FIELD_SPECS: tuple[FieldSpec, ...] = (
    # ---------------------------------------------------------------- company
    _f("company_name", "Company Name", C, TXT, critical=True),
    _f("website", "Website", C, TXT),
    _f("headquarters", "Headquarters", C, TXT),
    _f("country", "Country", C, TXT),
    _f("geographic_market", "Geographic Market", C, TXT),
    _f("industry", "Industry", C, TXT),
    _f("sector", "Sector", C, TXT, critical=True),
    _f("subsector", "Subsector", C, TXT),
    _f("business_model", "Business Model", C, TXT),
    _f("stage_declared", "Declared Stage", C, TXT, critical=True),
    # -------------------------------------------------------------- financing
    _f("target_raise", "Target Raise", F, MONEY, critical=True,
       aliases=("current raise", "raising", "seeking")),
    _f("amount_committed", "Amount Committed", F, MONEY, critical=True,
       aliases=("committed", "subscribed", "soft circled")),
    _f("percent_committed", "Percent of Round Committed", F, PCT, critical=True),
    _f("security_type", "Security Type", F, TXT),
    _f("pre_money_valuation", "Pre-Money Valuation", F, MONEY, critical=True),
    _f("post_money_valuation", "Post-Money Valuation", F, MONEY),
    _f("safe_cap", "SAFE Cap", F, MONEY),
    _f("note_cap", "Convertible Note Cap", F, MONEY),
    _f("discount", "Discount", F, PCT),
    _f("interest_rate", "Interest Rate", F, PCT),
    _f("maturity_date", "Maturity Date", F, DATE),
    _f("minimum_investment", "Minimum Investment", F, MONEY),
    _f("lead_investor", "Lead Investor", F, TXT, critical=True),
    _f("existing_investors", "Existing Investors", F, LST),
    _f("prior_funding", "Prior Funding Raised", F, MONEY, critical=True),
    _f("previous_round", "Previous Round", F, TXT),
    _f("previous_round_date", "Previous Round Date", F, DATE),
    _f("previous_round_size", "Previous Round Size", F, MONEY),
    _f("total_funding_to_date", "Total Funding To Date", F, MONEY),
    _f("use_of_funds", "Use of Funds", F, TXT, critical=True),
    # --------------------------------------------------------------- traction
    _f("annual_revenue", "Annual Revenue", T, MONEY, critical=True),
    _f("arr", "ARR", T, MONEY, critical=True),
    _f("mrr", "MRR", T, MONEY),
    _f("revenue_growth", "Revenue Growth Rate", T, PCT),
    _f("gross_margin", "Gross Margin", T, PCT, critical=True),
    _f("ebitda", "EBITDA", T, MONEY),
    _f("burn_rate", "Monthly Burn Rate", T, MONEY, critical=True),
    _f("cash_balance", "Cash Balance", T, MONEY),
    _f("runway_months", "Runway (months)", T, MON, critical=True),
    _f("customers", "Number of Customers", T, NUM, critical=True),
    _f("paying_customers", "Paying Customers", T, NUM),
    _f("enterprise_customers", "Enterprise Customers", T, NUM),
    _f("pilots", "Pilots", T, NUM),
    _f("lois", "Letters of Intent", T, NUM),
    _f("contracts", "Contracts", T, NUM),
    _f("pipeline", "Pipeline", T, MONEY),
    _f("backlog", "Backlog", T, MONEY),
    _f("renewal_rate", "Renewal Rate", T, PCT),
    _f("churn", "Churn", T, PCT),
    _f("average_contract_value", "Average Contract Value", T, MONEY),
    _f("sales_cycle", "Sales Cycle", T, TXT),
    _f("customer_concentration", "Customer Concentration", T, PCT),
    # ----------------------------------------------------------------- market
    _f("tam", "TAM", M, MONEY, critical=True),
    _f("sam", "SAM", M, MONEY),
    _f("som", "SOM", M, MONEY),
    _f("market_growth_rate", "Market Growth Rate", M, PCT),
    _f("primary_geography", "Primary Geography", M, TXT),
    _f("competitive_environment", "Competitive Environment", M, TXT),
    _f("market_timing", "Market Timing", M, TXT),
    # ------------------------------------------------------------------- team
    _f("founders", "Founders", TM, LST, critical=True),
    _f("ceo", "CEO", TM, TXT),
    _f("employees", "Employees", TM, NUM),
    _f("founder_experience", "Founder Experience", TM, TXT),
    _f("prior_exits", "Prior Exits", TM, TXT),
    _f("domain_expertise", "Domain Expertise", TM, TXT),
    _f("technical_expertise", "Technical Expertise", TM, TXT),
    _f("commercial_expertise", "Commercial Expertise", TM, TXT),
    _f("founder_full_time", "Founder Full-Time", TM, BOOL),
    _f("key_hires", "Key Hires", TM, TXT),
    _f("missing_roles", "Missing Critical Roles", TM, TXT),
    # ---------------------------------------------------------------- product
    _f("product_stage", "Product Stage", P, TXT, critical=True),
    _f("mvp_status", "MVP Status", P, TXT),
    _f("commercial_product_status", "Commercial Product Status", P, TXT),
    _f("regulatory_stage", "Regulatory Stage", P, TXT),
    _f("intellectual_property", "Intellectual Property", P, TXT),
    _f("patents", "Patents", P, TXT),
    _f("partnerships", "Partnerships", P, TXT),
    _f("customer_validation", "Customer Validation", P, TXT),
    _f("technology_readiness", "Technology Readiness", P, TXT),
    _f("clinical_stage", "Clinical Stage", P, TXT),
    _f("regulatory_approvals", "Regulatory Approvals", P, TXT),
)


FIELDS_BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in FIELD_SPECS}
CRITICAL_FIELDS: tuple[str, ...] = tuple(s.name for s in FIELD_SPECS if s.critical)


def fields_in(category: FieldCategory) -> tuple[FieldSpec, ...]:
    return tuple(s for s in FIELD_SPECS if s.category is category)


def spec_for(name: str) -> FieldSpec:
    try:
        return FIELDS_BY_NAME[name]
    except KeyError as exc:  # pragma: no cover - guards programmer error
        raise KeyError(f"Unknown field {name!r}. Add it to FIELD_SPECS.") from exc
