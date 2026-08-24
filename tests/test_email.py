"""Email notification: payload shape, isolation from the analysis, and config.

No test here makes a network call or needs an API key. The Resend transport is
stubbed at the ``httpx.post`` boundary.
"""

from __future__ import annotations

import base64
import json

import pytest

from reports.email_report import (
    build_analysis_email,
    build_failure_email,
    build_html,
    build_text,
    subject_for,
)
from services.email import (
    DEFAULT_RECIPIENT,
    MAX_ATTACHMENT_BYTES,
    SANDBOX_SENDER,
    Attachment,
    EmailMessage,
    NullEmailService,
    ResendEmailService,
    _fit_attachments,
    build_email_service,
    recipients_from_env,
)
from services.notify import notify_analysis, notify_failure, select_attachments


class StubResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture()
def captured(monkeypatch):
    """Capture the Resend request instead of sending it."""
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return StubResponse(200, {"id": "msg_test_123"})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


@pytest.fixture()
def service(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.delenv("FP_EMAIL_FROM", raising=False)
    monkeypatch.delenv("FP_EMAIL_REPLY_TO", raising=False)
    return ResendEmailService()


def _message(**kwargs) -> EmailMessage:
    defaults = dict(
        subject="Subject",
        html="<p>Body</p>",
        text="Body",
        to=["Info@tencapital.group"],
    )
    defaults.update(kwargs)
    return EmailMessage(**defaults)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
def test_successful_send_reports_the_message_id(service, captured):
    result = service.send(_message())
    assert result.sent is True
    assert result.message_id == "msg_test_123"
    assert result.recipients == ["Info@tencapital.group"]


def test_payload_matches_the_resend_contract(service, captured):
    service.send(_message(subject="Hello", html="<p>hi</p>", text="hi"))
    payload = captured[0]["json"]
    assert payload["from"] == SANDBOX_SENDER
    assert payload["to"] == ["Info@tencapital.group"]
    assert payload["subject"] == "Hello"
    assert payload["html"] == "<p>hi</p>"
    assert payload["text"] == "hi"
    assert captured[0]["headers"]["Authorization"] == "Bearer re_test_key"
    assert captured[0]["url"] == "https://api.resend.com/emails"


def test_attachments_are_base64_encoded(service, captured):
    service.send(_message(attachments=[Attachment("report.pdf", b"%PDF-1.4 fake")]))
    attachment = captured[0]["json"]["attachments"][0]
    assert attachment["filename"] == "report.pdf"
    assert base64.b64decode(attachment["content"]) == b"%PDF-1.4 fake"


def test_a_plain_text_alternative_is_always_present(service, captured):
    """Its absence is a deliverability problem, not a cosmetic one."""
    service.send(_message())
    assert captured[0]["json"]["text"]


def test_reply_to_is_included_when_configured(monkeypatch, captured):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("FP_EMAIL_REPLY_TO", "reports@tencapital.group")
    ResendEmailService().send(_message())
    assert captured[0]["json"]["reply_to"] == "reports@tencapital.group"


def test_http_error_is_returned_not_raised(service, monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: StubResponse(403, {"message": "verify a domain at resend.com/domains"}),
    )
    result = service.send(_message())
    assert result.sent is False
    assert "403" in result.error
    assert "verify a domain" in result.error


def test_network_failure_is_returned_not_raised(service, monkeypatch):
    import httpx

    def boom(*args, **kwargs):
        raise ConnectionError("dns failure")

    monkeypatch.setattr(httpx, "post", boom)
    result = service.send(_message())
    assert result.sent is False
    assert "dns failure" in result.error


def test_missing_key_skips_rather_than_failing(monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    result = ResendEmailService(api_key=None).send(_message())
    assert result.sent is False
    assert "RESEND_API_KEY" in result.skipped_reason


def test_sandbox_sender_is_detected(service):
    assert service.uses_sandbox_sender is True


def test_custom_sender_is_not_flagged_as_sandbox(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("FP_EMAIL_FROM", "TEN Capital <reports@tencapital.group>")
    assert ResendEmailService().uses_sandbox_sender is False


# ---------------------------------------------------------------------------
# Attachment budget
# ---------------------------------------------------------------------------
def test_oversized_attachments_are_dropped_not_fatal():
    small = Attachment("small.pdf", b"x" * 100)
    huge = Attachment("huge.pdf", b"x" * (MAX_ATTACHMENT_BYTES + 1))
    kept, dropped = _fit_attachments([small, huge])
    assert [a.filename for a in kept] == ["small.pdf"]
    assert dropped == ["huge.pdf"]


def test_attachments_within_budget_are_all_kept():
    items = [Attachment(f"f{n}.json", b"x" * 1000) for n in range(3)]
    kept, dropped = _fit_attachments(items)
    assert len(kept) == 3 and dropped == []


def test_unreadable_attachment_is_skipped(tmp_path):
    assert Attachment.from_path(tmp_path / "missing.pdf") is None


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
def test_subject_leads_with_the_decision(saas_result):
    subject = subject_for(saas_result)
    assert subject.startswith("[")
    assert saas_result.company_name in subject
    assert "6-month financing probability" in subject


def test_html_contains_the_headline_findings(saas_result):
    html = build_html(saas_result, ["report.pdf"])
    assert saas_result.company_name in html
    assert f"{saas_result.prediction.final.p6:.0f}%" in html
    assert "PROBABILITY OF COMPLETING THE ROUND".title() in html or "Probability of completing" in html
    assert "report.pdf" in html


def test_html_uses_table_layout_for_mail_clients(saas_result):
    """Flexbox and grid do not render in Outlook."""
    html = build_html(saas_result, [])
    assert "<table" in html
    assert "display:flex" not in html
    assert "display:grid" not in html
    assert "<link" not in html  # no external stylesheets


def test_html_escapes_company_supplied_text(saas_result):
    from models.evidence import ValueStatus

    injected = saas_result.model_copy(deep=True)
    injected.profile.put("company_name", "<script>alert(1)</script>", ValueStatus.EXPLICIT, 0.9)
    html = build_html(injected, [])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_text_alternative_carries_the_same_findings(saas_result):
    text = build_text(saas_result, ["report.pdf"])
    assert saas_result.company_name in text
    assert "PROBABILITY OF COMPLETING THE ROUND" in text
    assert "TEN CAPITAL READINESS FLAGS" in text
    assert "report.pdf" in text


def test_required_disclaimers_appear_in_both_parts(saas_result):
    html = build_html(saas_result, [])
    text = build_text(saas_result, [])
    for body in (html, text):
        assert "not guarantees of financing outcomes" in body
        assert "does not constitute investment" in body
        assert "CONFIDENTIAL" in body


def test_market_condition_is_disclosed_as_an_assumption(saas_result):
    assert "ASSUMPTION" in build_text(saas_result, [])


def test_analysis_email_defaults_to_the_ten_capital_inbox(saas_result, monkeypatch):
    monkeypatch.delenv("FP_EMAIL_TO", raising=False)
    assert recipients_from_env() == [DEFAULT_RECIPIENT]
    message = build_analysis_email(saas_result, recipients_from_env())
    assert message.to == ["Info@tencapital.group"]


def test_recipients_can_be_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv("FP_EMAIL_TO", "a@x.com, b@y.com")
    assert recipients_from_env() == ["a@x.com", "b@y.com"]


def test_tags_are_sanitised_for_resend(saas_result):
    message = build_analysis_email(saas_result, ["a@b.com"])
    for name, value in message.tags:
        assert value
        assert all(c.isascii() and (c.isalnum() or c in "_-") for c in value), value


def test_failure_email_names_the_file_and_error():
    message = build_failure_email("deck.pdf", "ExtractionError: broken", ["a@b.com"])
    assert "FAILED" in message.subject
    assert "deck.pdf" in message.subject
    assert "deck.pdf" in message.html and "deck.pdf" in message.text
    assert "broken" in message.html and "broken" in message.text


# ---------------------------------------------------------------------------
# Notifier wiring
# ---------------------------------------------------------------------------
def test_attachment_selection_honours_configuration(tmp_path):
    artifacts = {}
    for key, name in (("pdf", "r.pdf"), ("data", "d.json"), ("evidence", "e.json")):
        path = tmp_path / name
        path.write_text("x", encoding="utf-8")
        artifacts[key] = path

    chosen = select_attachments(artifacts, ("pdf", "data"))
    assert [p.name for p in chosen] == ["r.pdf", "d.json"]


def test_attachment_selection_skips_files_that_were_not_produced(tmp_path):
    path = tmp_path / "r.pdf"
    path.write_text("x", encoding="utf-8")
    assert select_attachments({"pdf": path}, ("pdf", "html")) == [path]


def test_unknown_attachment_key_is_ignored(tmp_path):
    assert select_attachments({}, ("pdf", "nonsense")) == []


def test_notify_is_skipped_when_disabled(saas_result, monkeypatch):
    monkeypatch.setenv("FP_EMAIL_ENABLED", "off")
    result = notify_analysis(saas_result, {}, enabled=False)
    assert result.sent is False
    assert result.skipped_reason


def test_notify_sends_when_enabled(saas_result, monkeypatch, captured):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    result = notify_analysis(saas_result, {}, enabled=True)
    assert result.sent is True
    assert captured[0]["json"]["subject"].startswith("[")


def test_notify_never_raises_when_composition_fails(saas_result, monkeypatch):
    """A bug in the email code must not lose a completed analysis."""
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setattr(
        "services.notify.build_analysis_email",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("template blew up")),
    )
    result = notify_analysis(saas_result, {}, enabled=True)
    assert result.sent is False
    assert "template blew up" in result.error


def test_failure_notification_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("FP_EMAIL_ON_FAILURE", "off")
    result = notify_failure("deck.pdf", "boom", enabled=True)
    assert result.sent is False
    assert "disabled" in result.skipped_reason


def test_failure_notification_sends_when_enabled(monkeypatch, captured):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("FP_EMAIL_ON_FAILURE", "on")
    result = notify_failure("deck.pdf", "boom", enabled=True)
    assert result.sent is True
    assert "FAILED" in captured[0]["json"]["subject"]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def test_email_is_off_by_default_even_with_a_key(monkeypatch):
    """A key in the environment must not start sending mail on its own."""
    from config.settings import RuntimeSettings

    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.delenv("FP_EMAIL_ENABLED", raising=False)
    assert RuntimeSettings.from_env().email_enabled is False


def test_email_auto_mode_follows_the_key(monkeypatch):
    from config.settings import RuntimeSettings

    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("FP_EMAIL_ENABLED", "auto")
    assert RuntimeSettings.from_env().email_enabled is True

    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    assert RuntimeSettings.from_env().email_enabled is False


def test_default_recipient_is_the_ten_capital_inbox(monkeypatch):
    from config.settings import RuntimeSettings

    monkeypatch.delenv("FP_EMAIL_TO", raising=False)
    assert RuntimeSettings.from_env().email_to == ["Info@tencapital.group"]


def test_build_service_returns_null_when_disabled():
    assert isinstance(build_email_service(False), NullEmailService)


def test_build_service_returns_null_without_a_key(monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    assert isinstance(build_email_service(True), NullEmailService)


def test_null_service_records_why_it_skipped():
    result = NullEmailService().send(_message())
    assert result.sent is False
    assert result.skipped_reason
    assert result.recipients == ["Info@tencapital.group"]
