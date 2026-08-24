"""One place that decides what gets emailed, so the CLI and the web app agree.

Both entry points call :func:`notify_analysis`; neither knows anything about
Resend. Every call is wrapped so a mail failure can only ever produce a logged
warning, never a lost report.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import RuntimeSettings
from models.prediction import AnalysisResult
from reports.email_report import build_analysis_email, build_failure_email
from services.email import EmailResult, build_email_service

LOGGER = logging.getLogger(__name__)

#: Artifact keys that may be attached, in the order they should appear.
ATTACHMENT_ORDER = ("pdf", "data", "evidence", "html")


def select_attachments(
    artifacts: dict[str, Path], wanted: tuple[str, ...]
) -> list[Path]:
    """Resolve the configured attachment list against what was produced."""
    chosen: list[Path] = []
    for key in ATTACHMENT_ORDER:
        if key in wanted and key in artifacts:
            path = artifacts[key]
            if path.exists():
                chosen.append(path)
    unknown = set(wanted) - set(ATTACHMENT_ORDER)
    if unknown:
        LOGGER.warning("Ignoring unknown attachment key(s): %s", ", ".join(sorted(unknown)))
    return chosen


def notify_analysis(
    result: AnalysisResult,
    artifacts: dict[str, Path],
    settings: RuntimeSettings | None = None,
    enabled: bool | None = None,
) -> EmailResult:
    """Email the results of a completed analysis.

    ``enabled`` lets a caller force sending on (a CLI ``--email`` flag) without
    changing the environment; ``None`` defers to the configured default.
    """
    settings = settings or RuntimeSettings.from_env()
    send = settings.email_enabled if enabled is None else enabled

    service = build_email_service(send)
    if not service.available():
        reason = (
            "Email notifications are disabled."
            if not send
            else "Resend is not configured (RESEND_API_KEY missing)."
        )
        return EmailResult(sent=False, skipped_reason=reason, recipients=settings.email_to)

    try:
        message = build_analysis_email(
            result,
            recipients=settings.email_to,
            attachment_paths=select_attachments(artifacts, settings.email_attachments),
        )
        return service.send(message)
    except Exception as exc:  # noqa: BLE001 - the report must survive a mail bug
        LOGGER.exception("Could not compose or send the analysis email.")
        return EmailResult(
            sent=False,
            error=f"{type(exc).__name__}: {exc}",
            recipients=settings.email_to,
        )


def notify_failure(
    filename: str,
    error: str,
    settings: RuntimeSettings | None = None,
    enabled: bool | None = None,
) -> EmailResult:
    """Email that an analysis failed, so a silent drop is never the outcome."""
    settings = settings or RuntimeSettings.from_env()
    send = settings.email_enabled if enabled is None else enabled
    if not send or not settings.email_on_failure:
        return EmailResult(
            sent=False,
            skipped_reason="Failure notifications are disabled.",
            recipients=settings.email_to,
        )

    service = build_email_service(True)
    if not service.available():
        return EmailResult(
            sent=False,
            skipped_reason="Resend is not configured (RESEND_API_KEY missing).",
            recipients=settings.email_to,
        )

    try:
        return service.send(build_failure_email(filename, error, settings.email_to))
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Could not send the failure email.")
        return EmailResult(
            sent=False,
            error=f"{type(exc).__name__}: {exc}",
            recipients=settings.email_to,
        )
