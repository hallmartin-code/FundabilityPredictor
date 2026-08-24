"""Fundability Predictor - command line interface.

    python app.py analyse deck.pdf
    python app.py review  deck.pdf --save-overrides overrides.json
    python app.py analyse deck.pdf --overrides overrides.json
    python app.py explain deck.pdf

``review`` exists because extraction is never perfect: it prints what was
found, with the page each value came from, and can emit an editable overrides
file. ``analyse --overrides`` then re-runs the model with the corrections,
which are recorded separately from the extracted values throughout the output.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from analysis.pipeline import analyse_profile, apply_overrides, extract_profile
from config.settings import ConfigError, RuntimeSettings, load_config
from extractors.base import ExtractionError
from extractors.ingest import supported_extensions
from extractors.normalization import (
    parse_bool,
    parse_date,
    parse_money,
    parse_months,
    parse_number,
    parse_percent,
)
from models.fields import FIELD_SPECS, FieldCategory, FieldKind, spec_for
from models.prediction import AnalysisResult
from models.profile import DeckProfile
from reports.html_report import render_html
from reports.json_report import (
    ConsentRecord,
    build_evidence_map,
    build_lead_record,
    write_json,
)
from reports.pdf_report import render_pdf
from services.llm import build_llm_service
from services.notify import notify_analysis
from services.email import build_email_service, recipients_from_env
from reports.email_report import build_analysis_email

LOGGER = logging.getLogger("financing_predictor")

EXIT_OK, EXIT_USAGE, EXIT_FAILURE = 0, 2, 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _configure_logging(level: str, quiet: bool) -> None:
    logging.basicConfig(
        level=getattr(logging, "ERROR" if quiet else level, logging.INFO),
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def safe_name(name: str | None, fallback: str = "Company") -> str:
    """A filesystem-safe token for output filenames."""
    cleaned = re.sub(r"[^\w\-. ]+", "", (name or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._-")
    return cleaned[:60] or fallback


def _coerce(field: str, raw: Any) -> Any:
    """Convert an override string into the field's declared type."""
    if raw is None or isinstance(raw, (int, float, bool, list)):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    kind = spec_for(field).kind
    if kind is FieldKind.MONEY:
        value, _ = parse_money(text)
        return value if value is not None else _number_or_none(text)
    if kind is FieldKind.NUMBER:
        return parse_number(text)
    if kind is FieldKind.PERCENT:
        return parse_percent(text) if "%" in text else _number_or_none(text)
    if kind is FieldKind.MONTHS:
        return parse_months(text) if not text.replace(".", "").isdigit() else float(text)
    if kind is FieldKind.BOOL:
        return parse_bool(text)
    if kind is FieldKind.DATE:
        return parse_date(text)
    if kind is FieldKind.LIST:
        return [part.strip() for part in text.split(",") if part.strip()]
    return text


def _number_or_none(text: str) -> float | None:
    try:
        return float(text.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Merge --overrides file and repeated --set field=value flags."""
    overrides: dict[str, Any] = {}

    if getattr(args, "overrides", None):
        path = Path(args.overrides)
        if not path.exists():
            raise ExtractionError(f"Overrides file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ExtractionError("The overrides file must contain a JSON object.")
        # Accept either {"field": value} or the annotated template form.
        for field, value in data.items():
            if isinstance(value, dict) and "value" in value:
                value = value["value"]
            overrides[field] = value

    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise ExtractionError(f"--set expects field=value, got {item!r}")
        field, _, raw = item.partition("=")
        overrides[field.strip()] = raw.strip()

    resolved: dict[str, Any] = {}
    for field, value in overrides.items():
        try:
            spec_for(field)
        except KeyError:
            LOGGER.warning("Ignoring override for unknown field %r.", field)
            continue
        coerced = _coerce(field, value)
        if coerced is not None:
            resolved[field] = coerced
        elif value not in (None, ""):
            LOGGER.warning("Could not interpret override %s=%r; ignoring.", field, value)
    return resolved


def _consent(args: argparse.Namespace) -> ConsentRecord:
    """Consent is only ever recorded from an explicit flag."""
    return ConsentRecord(
        analysis_consent=True if args.consent_analysis else None,
        ten_capital_contact_consent=True if args.consent_contact else None,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_review(args: argparse.Namespace) -> int:
    settings = RuntimeSettings.from_env()
    llm = build_llm_service(args.llm or settings.llm_enabled, settings.llm_model)
    profile = extract_profile(args.deck, llm=llm, overrides=collect_overrides(args))

    _print_profile(profile)

    if args.save_overrides:
        path = Path(args.save_overrides)
        template = {
            spec.name: {
                "label": spec.label,
                "value": profile.get(spec.name).value,
                "status": profile.get(spec.name).status.value,
                "source_page": profile.get(spec.name).source_page,
            }
            for spec in FIELD_SPECS
        }
        write_json(template, path)
        print(f"\nEditable overrides template written to {path}", file=sys.stderr)
    return EXIT_OK


def cmd_analyse(args: argparse.Namespace) -> int:
    settings = RuntimeSettings.from_env()
    config = load_config()
    llm = build_llm_service(args.llm or settings.llm_enabled, settings.llm_model)

    profile = extract_profile(args.deck, llm=llm, overrides=collect_overrides(args))
    result = analyse_profile(
        profile, config, market_condition=args.market or settings.market_condition
    )

    out_dir = Path(args.out) if args.out else settings.output_dir
    stem = safe_name(result.company_name, fallback=Path(args.deck).stem)
    page = (args.page or settings.page_size).upper()

    pdf_path = render_pdf(result, out_dir / f"{stem}_Financing_Probability_Report.pdf", page=page)
    data_path = write_json(
        build_lead_record(result, _consent(args)),
        out_dir / f"{stem}_Financing_Probability_Data.json",
    )
    evidence_path = write_json(
        build_evidence_map(profile), out_dir / f"{stem}_Extraction_Evidence.json"
    )
    written = [pdf_path, data_path, evidence_path]

    artifacts = {"pdf": pdf_path, "data": data_path, "evidence": evidence_path}
    if args.html:
        html_path = render_html(result, out_dir / f"{stem}_Financing_Probability_Report.html")
        written.append(html_path)
        artifacts["html"] = html_path

    # Email is best-effort: the files above already exist either way.
    email = notify_analysis(
        result, artifacts, settings, enabled=True if args.email else None
    )

    _print_summary(result)
    if email.sent or email.error:
        print(f"\n{email.summary}", file=sys.stderr)
    print("\nFiles written:", file=sys.stderr)
    for path in written:
        print(f"  {path}", file=sys.stderr)
    return EXIT_OK


def cmd_test_email(args: argparse.Namespace) -> int:
    """Verify the mail configuration without running an analysis.

    Deployment problems here are configuration problems - an unverified
    sending domain, a typo in the recipient - and they are much easier to
    diagnose before a real report depends on them.
    """
    settings = RuntimeSettings.from_env()
    service = build_email_service(True)

    print("\nEmail configuration")
    print("=" * 60)
    print(f"  RESEND_API_KEY   {'set' if settings.resend_api_key else 'NOT SET'}")
    print(f"  Recipients       {', '.join(recipients_from_env())}")
    print(f"  Sender           {getattr(service, 'sender', 'n/a')}")
    print(f"  Attachments      {', '.join(settings.email_attachments)}")
    print(f"  Enabled          {settings.email_enabled}")
    print(f"  On failure       {settings.email_on_failure}")

    if not service.available():
        print(
            "\nCannot send: Resend is not configured. Set RESEND_API_KEY.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    if getattr(service, "uses_sandbox_sender", False):
        print(
            "\nWARNING: sending from Resend's sandbox address. It only delivers to\n"
            "         the Resend account owner's own email. Verify a domain and\n"
            "         set FP_EMAIL_FROM to reach a team mailbox.",
            file=sys.stderr,
        )

    from services.email import EmailMessage

    result = service.send(
        EmailMessage(
            subject="[TEST] Fundability Predictor - email configuration check",
            html=(
                "<p>This is a configuration test from the Startup Financing Probability "
                "Predictor.</p><p>If you received it, analysis result emails will arrive "
                "at this address.</p>"
            ),
            text=(
                "This is a configuration test from the Startup Financing Probability "
                "Predictor.\n\nIf you received it, analysis result emails will arrive "
                "at this address.\n"
            ),
            to=recipients_from_env(),
            tags=[("type", "test")],
        )
    )
    print(f"\n{result.summary}", file=sys.stderr)
    return EXIT_OK if result.sent else EXIT_FAILURE


def cmd_explain(args: argparse.Namespace) -> int:
    settings = RuntimeSettings.from_env()
    llm = build_llm_service(args.llm or settings.llm_enabled, settings.llm_model)
    profile = extract_profile(args.deck, llm=llm, overrides=collect_overrides(args))
    result = analyse_profile(
        profile, load_config(), market_condition=args.market or settings.market_condition
    )

    print(f"\n{result.company_name} - why this financing probability?\n")
    for horizon in ("p3", "p6", "p12"):
        print(f"--- {horizon.removeprefix('p')}-month horizon ---")
        print(result.prediction.bridge_text(horizon))
        print()

    print("Readiness score breakdown:")
    for category in result.readiness.categories:
        print(f"  {category.label:26s} {category.points:5.1f} / {category.weight:<4.0f}")
        for signal in category.signals:
            mark = "*" if signal.evidenced else "-"
            print(f"     {mark} {signal.label:34s} {signal.awarded:5.2f} / {signal.weight:<5.2f} {signal.detail}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
def _print_profile(profile: DeckProfile) -> None:
    print(f"\nExtracted profile - {profile.source.filename}")
    print(f"{'=' * 78}")
    for category in FieldCategory:
        rows = [
            value
            for value in profile.by_category(category)
            if value.is_known
        ]
        if not rows:
            continue
        print(f"\n{category.value.upper()}")
        for value in rows:
            page = f"p{value.source_page}" if value.source_page else "  -"
            flag = " (override)" if value.overridden else ""
            print(
                f"  {spec_for(value.field).label:28s} {str(value.value)[:44]:46s} "
                f"{value.status.value:10s} {value.confidence:.2f} {page}{flag}"
            )

    missing = [f for f in profile.missing_fields() if spec_for(f).critical]
    if missing:
        print(f"\nMISSING CRITICAL FIELDS ({len(missing)})")
        print("  " + ", ".join(spec_for(f).label for f in missing))
    for warning in profile.extraction_quality.warnings():
        print(f"\n! {warning}")
    for inconsistency in profile.inconsistencies:
        print(f"\n! {inconsistency.message}")
    print(f"\nData completeness: {profile.completeness() * 100:.0f}%")


def _print_summary(result: AnalysisResult) -> None:
    final = result.prediction.final
    print(f"\n{result.company_name}")
    print(f"{'=' * 78}")
    print(
        f"{result.sector.label} | declared {result.stage.declared.value} -> "
        f"inferred {result.stage.inferred.value} (stage confidence {result.stage.confidence})"
    )
    print(f"\nProbability of completing the round")
    print(f"  3 months   {final.p3:5.0f}%")
    print(f"  6 months   {final.p6:5.0f}%")
    print(f" 12 months   {final.p12:5.0f}%")
    print(f"\nFundability score    {result.readiness.total:.0f}/100  ({result.readiness.label})")
    print(f"Prediction confidence {result.confidence.score:.0f}/100 ({result.confidence.level})")
    print(f"Comparable profile   {result.comparable.label}")
    print(f"{result.prediction.market_condition.disclosure}")

    print("\nBlockers")
    for blocker in result.blockers:
        print(f"  [{blocker.severity:8s}] {blocker.title}")
    print("\nActions")
    for index, action in enumerate(result.actions, start=1):
        print(f"  {index}. [{action.impact:6s} {action.timeframe:9s}] {action.action}")
    print(
        f"\nTEN Capital fit {result.ten_capital.fit_score:.0f}/100 - "
        f"{result.ten_capital.recommended_next_step}"
    )

    estimated = result.profile.estimated_fields()
    if estimated:
        labels = ", ".join(spec_for(f).label for f in sorted(estimated))
        print(f"\nEstimated (not stated in the deck): {labels}")
    overridden = result.profile.overridden_fields()
    if overridden:
        labels = ", ".join(spec_for(f).label for f in overridden)
        print(f"User overrides applied: {labels}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    formats = ", ".join(supported_extensions())
    parser = argparse.ArgumentParser(
        prog="app.py",
        description=(
            "Estimate the probability that a startup completes its financing round "
            f"within 3, 6 and 12 months. Supported deck formats: {formats}."
        ),
    )
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING or ERROR.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Errors only.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("deck", help=f"Path to the pitch deck ({formats}).")
        sub.add_argument(
            "--set", action="append", metavar="FIELD=VALUE",
            help="Override an extracted field. Repeatable.",
        )
        sub.add_argument("--overrides", help="JSON file of field overrides.")
        sub.add_argument("--llm", action="store_true", help="Enable optional LLM extraction assist.")
        sub.add_argument(
            "--market",
            choices=["highly_favorable", "favorable", "neutral", "difficult", "highly_difficult"],
            help="Capital market environment assumption (default: neutral).",
        )

    review = subparsers.add_parser("review", help="Extract and print the profile for correction.")
    add_common(review)
    review.add_argument("--save-overrides", help="Write an editable overrides template here.")
    review.set_defaults(func=cmd_review)

    analyse_cmd = subparsers.add_parser("analyse", aliases=["analyze"], help="Run the full analysis and write reports.")
    add_common(analyse_cmd)
    analyse_cmd.add_argument("--out", help="Output directory (default: ./output).")
    analyse_cmd.add_argument("--page", choices=["LETTER", "A4"], help="PDF page size.")
    analyse_cmd.add_argument("--html", action="store_true", help="Also write an HTML version.")
    analyse_cmd.add_argument(
        "--email", action="store_true",
        help="Email the results (requires RESEND_API_KEY; see FP_EMAIL_TO).",
    )
    analyse_cmd.add_argument(
        "--consent-analysis", action="store_true",
        help="Record that the company consented to this analysis.",
    )
    analyse_cmd.add_argument(
        "--consent-contact", action="store_true",
        help="Record that the company consented to TEN Capital contact.",
    )
    analyse_cmd.set_defaults(func=cmd_analyse)

    test_email = subparsers.add_parser(
        "test-email", help="Send a test email to verify the Resend configuration."
    )
    test_email.set_defaults(func=cmd_test_email)

    explain = subparsers.add_parser("explain", help="Print the probability bridge and score breakdown.")
    add_common(explain)
    explain.set_defaults(func=cmd_explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level.upper(), args.quiet)

    # Consent flags only exist on `analyse`; default them for the other commands.
    for flag in ("consent_analysis", "consent_contact", "email", "llm"):
        if not hasattr(args, flag):
            setattr(args, flag, False)

    try:
        return int(args.func(args))
    except ExtractionError as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE
    except ConfigError as exc:
        LOGGER.error("Configuration problem: %s", exc)
        return EXIT_FAILURE
    except KeyboardInterrupt:  # pragma: no cover
        LOGGER.error("Interrupted.")
        return EXIT_FAILURE
    except Exception:  # noqa: BLE001 - the CLI is the last line of defence
        LOGGER.exception("Unexpected failure.")
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
