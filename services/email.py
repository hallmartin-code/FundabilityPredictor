"""Email delivery through Resend.

One rule governs this module: **a delivery failure must never fail an
analysis.** The report is the product; the email is a convenience. Every send
is wrapped, returns a result object rather than raising, and logs what went
wrong so a misconfigured mailbox shows up in the logs instead of losing a
report.

Resend is reached over its REST API with ``httpx`` rather than the ``resend``
SDK, which is a thin wrapper over the single endpoint used here and would add
a dependency for no benefit.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

LOGGER = logging.getLogger(__name__)

RESEND_ENDPOINT = "https://api.resend.com/emails"
REQUEST_TIMEOUT_SECONDS = 20.0

#: Resend caps a message at 40MB across all attachments. Stay well inside it:
#: a report that cannot be attached should still arrive as a summary.
MAX_ATTACHMENT_BYTES = 15 * 1024 * 1024

DEFAULT_RECIPIENT = "Info@tencapital.group"

#: Resend's shared sandbox sender. It works without domain verification but
#: will only deliver to the Resend account owner's own address, so it is a
#: testing default, not a production one.
SANDBOX_SENDER = "TEN Capital Reports <onboarding@resend.dev>"


@dataclass
class Attachment:
    filename: str
    content: bytes

    @property
    def size(self) -> int:
        return len(self.content)

    def as_payload(self) -> dict[str, str]:
        return {
            "filename": self.filename,
            "content": base64.b64encode(self.content).decode("ascii"),
        }

    @classmethod
    def from_path(cls, path: Path) -> Attachment | None:
        try:
            return cls(filename=path.name, content=path.read_bytes())
        except OSError as exc:
            LOGGER.warning("Could not read attachment %s: %s", path, exc)
            return None


@dataclass
class EmailMessage:
    subject: str
    html: str
    text: str
    to: list[str]
    reply_to: str | None = None
    attachments: list[Attachment] = field(default_factory=list)
    tags: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class EmailResult:
    """What happened. Never an exception - the caller carries on regardless."""

    sent: bool = False
    message_id: str | None = None
    error: str | None = None
    skipped_reason: str | None = None
    recipients: list[str] = field(default_factory=list)
    attachments_dropped: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sent": self.sent,
            "message_id": self.message_id,
            "recipients": self.recipients,
            "error": self.error,
            "skipped_reason": self.skipped_reason,
            "attachments_dropped": self.attachments_dropped,
        }

    @property
    def summary(self) -> str:
        if self.sent:
            return f"Emailed {', '.join(self.recipients)} (id {self.message_id})."
        if self.skipped_reason:
            return f"Email skipped: {self.skipped_reason}"
        return f"Email failed: {self.error}"


@runtime_checkable
class EmailService(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def send(self, message: EmailMessage) -> EmailResult:
        ...


class NullEmailService:
    """The default. Sends nothing, so no analysis depends on a mail provider."""

    name = "disabled"

    def __init__(self, reason: str = "Email notifications are disabled.") -> None:
        self.reason = reason

    def available(self) -> bool:
        return False

    def send(self, message: EmailMessage) -> EmailResult:
        LOGGER.debug("Email suppressed (%s): %s", self.reason, message.subject)
        return EmailResult(sent=False, skipped_reason=self.reason, recipients=list(message.to))


class ResendEmailService:
    """Send through Resend's REST API."""

    name = "resend"

    def __init__(
        self,
        api_key: str | None = None,
        sender: str | None = None,
        reply_to: str | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("RESEND_API_KEY")
        self.sender = sender or os.getenv("FP_EMAIL_FROM") or SANDBOX_SENDER
        self.reply_to = reply_to or os.getenv("FP_EMAIL_REPLY_TO") or None

    def available(self) -> bool:
        if not self._api_key:
            LOGGER.info("RESEND_API_KEY is not set; email notifications are unavailable.")
            return False
        try:
            import httpx  # noqa: F401
        except ImportError:  # pragma: no cover - declared dependency
            LOGGER.warning("httpx is not installed; email notifications are unavailable.")
            return False
        return True

    @property
    def uses_sandbox_sender(self) -> bool:
        """True when sending from Resend's shared domain.

        Worth surfacing: that sender only delivers to the Resend account
        owner's own address, so a deployment that leaves it in place will
        silently fail to reach a team mailbox.
        """
        return "onboarding@resend.dev" in self.sender

    def send(self, message: EmailMessage) -> EmailResult:
        if not self.available():
            return EmailResult(
                sent=False,
                skipped_reason="Resend is not configured (RESEND_API_KEY missing).",
                recipients=list(message.to),
            )

        import httpx

        attachments, dropped = _fit_attachments(message.attachments)
        payload: dict = {
            "from": self.sender,
            "to": message.to,
            "subject": message.subject,
            "html": message.html,
            "text": message.text,
        }
        if self.reply_to:
            payload["reply_to"] = self.reply_to
        if attachments:
            payload["attachments"] = [a.as_payload() for a in attachments]
        if message.tags:
            payload["tags"] = [{"name": n, "value": v} for n, v in message.tags]

        try:
            response = httpx.post(
                RESEND_ENDPOINT,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - network failure must not propagate
            LOGGER.exception("Resend request failed.")
            return EmailResult(
                sent=False,
                error=f"{type(exc).__name__}: {exc}",
                recipients=list(message.to),
                attachments_dropped=dropped,
            )

        if response.status_code >= 400:
            detail = _error_detail(response)
            LOGGER.error("Resend rejected the message (%s): %s", response.status_code, detail)
            return EmailResult(
                sent=False,
                error=f"HTTP {response.status_code}: {detail}",
                recipients=list(message.to),
                attachments_dropped=dropped,
            )

        try:
            message_id = response.json().get("id")
        except Exception:  # noqa: BLE001 - a 2xx without JSON still means sent
            message_id = None

        LOGGER.info("Emailed %s (id %s).", ", ".join(message.to), message_id)
        return EmailResult(
            sent=True,
            message_id=message_id,
            recipients=list(message.to),
            attachments_dropped=dropped,
        )


def _error_detail(response) -> str:  # noqa: ANN001 - httpx.Response
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        return (response.text or "")[:300]
    if isinstance(body, dict):
        return str(body.get("message") or body.get("error") or body)[:300]
    return str(body)[:300]


def _fit_attachments(attachments: list[Attachment]) -> tuple[list[Attachment], list[str]]:
    """Keep attachments within the size budget, dropping the largest first.

    Dropping is preferable to failing: the summary in the email body carries
    the headline findings even when the PDF cannot ride along.
    """
    kept: list[Attachment] = []
    dropped: list[str] = []
    total = 0
    for item in attachments:
        if total + item.size > MAX_ATTACHMENT_BYTES:
            dropped.append(item.filename)
            LOGGER.warning("Dropped attachment %s: message size budget exceeded.", item.filename)
            continue
        kept.append(item)
        total += item.size
    return kept, dropped


def recipients_from_env() -> list[str]:
    """Recipient list, comma-separated, defaulting to the TEN Capital inbox."""
    raw = os.getenv("FP_EMAIL_TO") or DEFAULT_RECIPIENT
    return [address.strip() for address in raw.split(",") if address.strip()]


def build_email_service(enabled: bool) -> EmailService:
    if not enabled:
        return NullEmailService()
    service = ResendEmailService()
    if not service.available():
        return NullEmailService("Resend is not configured (RESEND_API_KEY missing).")
    if service.uses_sandbox_sender:
        LOGGER.warning(
            "Sending from %s. Resend's sandbox sender only delivers to the account "
            "owner's own address - verify a domain and set FP_EMAIL_FROM to reach "
            "a team mailbox.",
            service.sender,
        )
    return service
