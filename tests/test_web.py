"""The FastAPI web app.

Runs entirely against the deterministic pipeline - no API key, no network.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from web import main as web_main
from web.jobs import JobRegistry, JobState


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A client with an isolated job directory and no access password."""
    monkeypatch.delenv("FP_ACCESS_PASSWORD", raising=False)
    monkeypatch.setenv("FP_LLM_ENABLED", "off")
    monkeypatch.setattr(web_main, "registry", JobRegistry(tmp_path / "jobs"))
    with TestClient(web_main.app) as test_client:
        yield test_client


def _wait(client, job_id: str, timeout: float = 90.0) -> dict:
    """Poll a job to completion, as the browser does."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/job/{job_id}").json()
        if payload["state"] in ("done", "failed"):
            return payload
        time.sleep(0.25)
    raise AssertionError("job did not finish in time")


def _upload(client, path, **fields):
    with path.open("rb") as fh:
        return client.post(
            "/analyse",
            files={"deck": (path.name, fh, "application/octet-stream")},
            data=fields,
        )


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------
def test_healthcheck_needs_no_auth(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_renders_the_upload_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert '<form id="form"' in response.text
    assert 'type="file"' in response.text
    assert "/static/app.css" in response.text


def test_index_advertises_only_the_formats_the_app_accepts(client):
    """The UI must never offer a format ingestion would reject."""
    from extractors.ingest import supported_extensions

    text = client.get("/").text
    for extension in supported_extensions():
        assert extension in text
    assert ".docx" not in text


def test_index_discloses_when_llm_is_unavailable(client):
    assert "no API key configured" in client.get("/").text


def test_index_states_the_real_upload_limit(client, monkeypatch):
    text = client.get("/").text
    assert str(web_main.MAX_UPLOAD_BYTES) in text  # the client-side guard
    assert f"{web_main.MAX_UPLOAD_BYTES // (1024 * 1024)}" in text


def test_index_names_the_configured_recipient_when_email_is_on(tmp_path, monkeypatch):
    """The disclosure must reflect where mail actually goes."""
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("FP_EMAIL_ENABLED", "on")
    monkeypatch.setenv("FP_EMAIL_TO", "Info@tencapital.group")
    monkeypatch.delenv("FP_ACCESS_PASSWORD", raising=False)
    monkeypatch.setattr(web_main, "registry", JobRegistry(tmp_path / "jobs"))
    with TestClient(web_main.app) as client:
        assert "Info@tencapital.group" in client.get("/").text


def test_index_says_so_when_email_is_off(client):
    text = client.get("/").text
    assert "Email notifications are not enabled" in text


def test_stylesheet_is_served(client):
    response = client.get("/static/app.css")
    assert response.status_code == 200
    assert "--navy-950" in response.text


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
def test_unsupported_extension_is_rejected(client, unsupported_file):
    response = _upload(client, unsupported_file)
    assert response.status_code == 400
    assert ".pdf" in response.json()["detail"]


def test_empty_upload_is_rejected(client, tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    assert _upload(client, empty).status_code == 400


def test_oversized_upload_is_rejected(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_main, "MAX_UPLOAD_BYTES", 1024)
    big = tmp_path / "big.pdf"
    big.write_bytes(b"%PDF-1.4\n" + b"x" * 5000)
    response = _upload(client, big)
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_corrupt_deck_fails_the_job_with_a_message(client, tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\nnot a real pdf\n")
    job_id = _upload(client, broken).json()["job_id"]
    payload = _wait(client, job_id)
    assert payload["state"] == "failed"
    assert payload["error"]


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------
def test_analysis_produces_all_three_artifacts(client, saas_deck):
    response = _upload(client, saas_deck)
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    payload = _wait(client, job_id)
    assert payload["state"] == "done", payload.get("error")
    assert set(payload["downloads"]) == {"pdf", "data", "evidence"}

    summary = payload["summary"]
    assert summary["company"] == "Northwind Analytics"
    assert 1 <= summary["p3"] <= summary["p6"] <= summary["p12"] <= 98
    assert summary["fields_total"] == 81


def test_downloads_serve_the_right_content(client, saas_deck):
    job_id = _upload(client, saas_deck).json()["job_id"]
    _wait(client, job_id)

    pdf = client.get(f"/download/{job_id}/pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")

    data = client.get(f"/download/{job_id}/data")
    record = json.loads(data.content)
    assert record["company"]["name"] == "Northwind Analytics"

    evidence = json.loads(client.get(f"/download/{job_id}/evidence").content)
    assert len(evidence["fields"]) == 81


def test_job_page_renders(client, saas_deck):
    job_id = _upload(client, saas_deck).json()["job_id"]
    page = client.get(f"/job/{job_id}")
    assert page.status_code == 200
    assert job_id in page.text


def test_market_condition_flows_through(client, saas_deck):
    job_id = _upload(client, saas_deck, market="highly_difficult").json()["job_id"]
    payload = _wait(client, job_id)
    record = json.loads(client.get(f"/download/{job_id}/data").content)
    assert record["prediction"]["market_condition"] == "Highly Difficult"
    assert "ASSUMPTION" in payload["summary"]["market_condition"]


def test_a4_page_size_is_honoured(client, saas_deck):
    from pypdf import PdfReader
    import io

    job_id = _upload(client, saas_deck, page_size="A4").json()["job_id"]
    _wait(client, job_id)
    pdf = PdfReader(io.BytesIO(client.get(f"/download/{job_id}/pdf").content))
    assert len(pdf.pages) == 1
    assert float(pdf.pages[0].mediabox.width) == pytest.approx(595, abs=2)


def test_report_is_always_one_page(client, sparse_deck):
    from pypdf import PdfReader
    import io

    job_id = _upload(client, sparse_deck).json()["job_id"]
    _wait(client, job_id)
    pdf = PdfReader(io.BytesIO(client.get(f"/download/{job_id}/pdf").content))
    assert len(pdf.pages) == 1


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------
def test_unticked_consent_is_recorded_as_null(client, saas_deck):
    job_id = _upload(client, saas_deck).json()["job_id"]
    _wait(client, job_id)
    consent = json.loads(client.get(f"/download/{job_id}/data").content)["consent"]
    assert consent["analysis_consent"] is None
    assert consent["ten_capital_contact_consent"] is None


def test_ticked_consent_is_recorded(client, saas_deck):
    job_id = _upload(client, saas_deck, analysis_consent="1").json()["job_id"]
    _wait(client, job_id)
    consent = json.loads(client.get(f"/download/{job_id}/data").content)["consent"]
    assert consent["analysis_consent"] is True
    # Only what was ticked is recorded.
    assert consent["ten_capital_contact_consent"] is None


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------
def test_unknown_job_returns_404(client):
    assert client.get("/api/job/deadbeef").status_code == 404
    assert client.get("/job/deadbeef").status_code == 404
    assert client.get("/download/deadbeef/pdf").status_code == 404


def test_unknown_artifact_returns_404(client, saas_deck):
    job_id = _upload(client, saas_deck).json()["job_id"]
    _wait(client, job_id)
    assert client.get(f"/download/{job_id}/nonsense").status_code == 404


def test_expired_job_is_purged_with_its_upload(tmp_path, monkeypatch):
    """The uploaded deck is confidential and must not outlive the job."""
    monkeypatch.setattr("web.jobs.JOB_TTL_SECONDS", -1)
    registry = JobRegistry(tmp_path / "jobs")
    job = registry.create("deck.pdf")
    (job.workdir / "deck.pdf").write_bytes(b"confidential")
    workdir = job.workdir

    assert registry.get(job.id) is None
    assert not workdir.exists()


def test_registry_evicts_beyond_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr("web.jobs.MAX_JOBS", 3)
    registry = JobRegistry(tmp_path / "jobs")
    jobs = [registry.create(f"deck{n}.pdf") for n in range(5)]
    assert registry.get(jobs[0].id) is None
    assert registry.get(jobs[-1].id) is not None


def test_job_state_messages_cover_every_state():
    from web.jobs import STATE_MESSAGES

    assert set(STATE_MESSAGES) == set(JobState)


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------
def test_password_gate_blocks_and_then_admits(tmp_path, monkeypatch):
    monkeypatch.setenv("FP_ACCESS_PASSWORD", "let-me-in")
    monkeypatch.setenv("FP_LLM_ENABLED", "off")
    monkeypatch.setattr(web_main, "registry", JobRegistry(tmp_path / "jobs"))

    with TestClient(web_main.app) as client:
        # The index redirects to the login form.
        assert client.get("/", follow_redirects=False).status_code == 303
        # The API refuses without the cookie.
        assert client.get("/api/job/anything").status_code == 401

        assert client.post("/login", data={"password": "wrong"}).status_code == 401
        assert client.post(
            "/login", data={"password": "let-me-in"}, follow_redirects=False
        ).status_code == 303
        # The cookie now admits the client.
        assert client.get("/").status_code == 200


def test_healthcheck_is_reachable_behind_the_password_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("FP_ACCESS_PASSWORD", "let-me-in")
    monkeypatch.setattr(web_main, "registry", JobRegistry(tmp_path / "jobs"))
    with TestClient(web_main.app) as client:
        assert client.get("/healthz").status_code == 200
