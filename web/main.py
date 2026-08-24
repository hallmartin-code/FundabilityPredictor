"""FastAPI web front end.

    uvicorn web.main:app --reload

Upload a deck, Claude reads it, the deterministic model scores it, and the
one-page report plus both JSON files come back as downloads.

Two deployment facts shape the design:

* Analysis runs for tens of seconds, so uploads start a background job and the
  browser polls. Holding the request open would hit a proxy timeout.
* Uploaded decks are confidential. They live in a per-job directory that is
  deleted when the job expires, and nothing is written outside it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import mkdtemp

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.pipeline import analyse_profile, extract_profile  # noqa: E402
from app import safe_name  # noqa: E402
from config.settings import RuntimeSettings, load_config  # noqa: E402
from extractors.base import ExtractionError  # noqa: E402
from extractors.ingest import supported_extensions  # noqa: E402
from reports.json_report import (  # noqa: E402
    ConsentRecord,
    build_evidence_map,
    build_lead_record,
    write_json,
)
from reports.pdf_report import render_pdf  # noqa: E402
from services.llm import build_llm_service  # noqa: E402
from services.notify import notify_analysis, notify_failure  # noqa: E402
from services.email import recipients_from_env  # noqa: E402
from web.jobs import JOB_TTL_SECONDS, JobRegistry, JobState  # noqa: E402

LOGGER = logging.getLogger("financing_predictor.web")

#: Uploads above this are rejected before anything is written to disk.
MAX_UPLOAD_BYTES = int(os.getenv("FP_MAX_UPLOAD_MB", "40")) * 1024 * 1024

#: Concurrent analyses. Each one holds a worker thread and an API call, so the
#: cap protects both the container and the API rate limit.
MAX_CONCURRENT = int(os.getenv("FP_MAX_CONCURRENT", "2"))

ARTIFACT_LABELS = {
    "pdf": "Financing Probability Report (PDF)",
    "data": "TEN Capital Lead Record (JSON)",
    "evidence": "Extraction Evidence (JSON)",
    "html": "Report (HTML)",
}

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
registry = JobRegistry(Path(os.getenv("FP_JOB_DIR") or mkdtemp(prefix="fp_jobs_")))
_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    logging.basicConfig(
        level=os.getenv("FP_LOG_LEVEL", "INFO").upper(),
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    load_config()  # fail fast on a bad config rather than on first upload
    cleaner = asyncio.create_task(_purge_loop())
    LOGGER.info(
        "Ready. LLM extraction %s, job dir %s",
        "enabled" if _llm_enabled() else "disabled",
        registry.root,
    )
    try:
        yield
    finally:
        cleaner.cancel()
        registry.shutdown()


app = FastAPI(
    title="Startup Financing Probability Predictor",
    description="Upload a pitch deck; get a one-page financing probability report.",
    version="1.0",
    lifespan=lifespan,
)


app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)


def _template_context() -> dict:
    """Chrome shared by every page.

    The accepted formats, size limit and notification address are read from
    the live configuration rather than written into the markup, so the UI can
    never advertise a format the app does not accept or a mailbox it does not
    send to.
    """
    settings = RuntimeSettings.from_env()
    return {
        "extensions": list(supported_extensions()),
        "max_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "max_bytes": MAX_UPLOAD_BYTES,
        "llm_enabled": settings.llm_enabled,
        "email_enabled": settings.email_enabled and settings.email_configured,
        "email_to": recipients_from_env(),
        "retention_minutes": JOB_TTL_SECONDS // 60,
    }


async def _purge_loop() -> None:
    while True:
        await asyncio.sleep(300)
        try:
            registry.purge_expired()
        except Exception:  # noqa: BLE001 - housekeeping must never kill the app
            LOGGER.exception("Job purge failed.")


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------
def _access_password() -> str | None:
    value = os.getenv("FP_ACCESS_PASSWORD", "").strip()
    return value or None


def require_access(request: Request) -> None:
    """Optional shared-password gate.

    A deployed instance spends money on every analysis, so an unauthenticated
    public URL is an open billing endpoint. Setting FP_ACCESS_PASSWORD closes
    it; leaving it unset keeps local development frictionless.
    """
    expected = _access_password()
    if not expected:
        return
    supplied = request.cookies.get("fp_access", "")
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Access password required.")


def _llm_enabled() -> bool:
    return RuntimeSettings.from_env().llm_enabled


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/healthz")
async def healthz() -> JSONResponse:
    """Railway health check. Deliberately does no work and needs no auth."""
    return JSONResponse({"status": "ok", "llm": _llm_enabled()})


@app.get("/login", response_class=HTMLResponse, response_model=None)
async def login_form(request: Request):
    if not _access_password():
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {**_template_context(), "error": None})


# response_model=None because this returns either a redirect or a re-rendered
# form, and FastAPI cannot build a response model from that union.
@app.post("/login", response_model=None)
async def login(request: Request, password: str = Form("")) -> RedirectResponse | HTMLResponse:
    expected = _access_password()
    if not expected:
        return RedirectResponse("/", status_code=303)
    if not secrets.compare_digest(password, expected):
        return templates.TemplateResponse(
            request,
            "login.html",
            {**_template_context(), "error": "Incorrect password."},
            status_code=401,
        )
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        "fp_access", expected, httponly=True, samesite="lax",
        secure=request.url.scheme == "https", max_age=12 * 3600,
    )
    return response


@app.get("/", response_class=HTMLResponse, response_model=None)
async def index(request: Request):
    if _access_password() and request.cookies.get("fp_access", "") != _access_password():
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "index.html", _template_context())


@app.post("/analyse", dependencies=[Depends(require_access)])
async def start_analysis(
    request: Request,
    deck: UploadFile = File(...),
    market: str = Form("neutral"),
    page_size: str = Form("LETTER"),
    use_llm: str = Form(""),
    analysis_consent: str = Form(""),
    contact_consent: str = Form(""),
) -> JSONResponse:
    filename = Path(deck.filename or "deck").name
    suffix = Path(filename).suffix.lower()
    if suffix not in supported_extensions():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(supported_extensions())}.",
        )

    payload = await deck.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
        )

    job = registry.create(filename)
    deck_path = job.workdir / filename
    deck_path.write_bytes(payload)

    # Consent is recorded only from an explicit tick, never inferred.
    consent = ConsentRecord(
        analysis_consent=True if analysis_consent else None,
        ten_capital_contact_consent=True if contact_consent else None,
    )

    asyncio.create_task(
        _run_job(
            job_id=job.id,
            deck_path=deck_path,
            market=market,
            page_size=page_size,
            use_llm=bool(use_llm),
            consent=consent,
        )
    )
    return JSONResponse({"job_id": job.id, "poll": f"/api/job/{job.id}"}, status_code=202)


@app.get("/job/{job_id}", response_class=HTMLResponse, dependencies=[Depends(require_access)])
async def job_page(request: Request, job_id: str) -> HTMLResponse:
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    return templates.TemplateResponse(
        request,
        "job.html",
        {**_template_context(), "job": job.as_dict(), "labels": ARTIFACT_LABELS},
    )


@app.get("/api/job/{job_id}", dependencies=[Depends(require_access)])
async def job_status(job_id: str) -> JSONResponse:
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    return JSONResponse(job.as_dict())


@app.get("/download/{job_id}/{kind}", dependencies=[Depends(require_access)])
async def download(job_id: str, kind: str) -> FileResponse:
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    path = job.artifacts.get(kind)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not available.")
    media = "application/pdf" if path.suffix == ".pdf" else (
        "text/html" if path.suffix == ".html" else "application/json"
    )
    return FileResponse(path, media_type=media, filename=path.name)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
async def _run_job(
    job_id: str,
    deck_path: Path,
    market: str,
    page_size: str,
    use_llm: bool,
    consent: ConsentRecord,
) -> None:
    async with _semaphore:
        try:
            await asyncio.to_thread(
                _analyse_sync, job_id, deck_path, market, page_size, use_llm, consent
            )
        except ExtractionError as exc:
            LOGGER.warning("Job %s rejected the file: %s", job_id, exc)
            registry.update(job_id, state=JobState.FAILED, error=str(exc))
            await _notify_failure(deck_path.name, str(exc))
        except Exception as exc:  # noqa: BLE001 - surface a usable message
            LOGGER.exception("Job %s failed.", job_id)
            detail = f"{type(exc).__name__}: {exc}"
            registry.update(job_id, state=JobState.FAILED, error=detail)
            await _notify_failure(deck_path.name, detail)


async def _notify_failure(filename: str, error: str) -> None:
    """Best-effort failure notification; never raises into the job handler."""
    try:
        await asyncio.to_thread(notify_failure, filename, error)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failure notification could not be sent.")


def _analyse_sync(
    job_id: str,
    deck_path: Path,
    market: str,
    page_size: str,
    use_llm: bool,
    consent: ConsentRecord,
) -> None:
    """The blocking pipeline, run in a worker thread."""
    settings = RuntimeSettings.from_env()
    config = load_config()

    registry.update(job_id, state=JobState.EXTRACTING)
    llm = build_llm_service(use_llm and settings.llm_enabled, settings.llm_model)
    profile = extract_profile(deck_path, llm=llm)

    registry.update(job_id, state=JobState.ANALYSING)
    result = analyse_profile(profile, config, market_condition=market or None)

    registry.update(job_id, state=JobState.RENDERING)
    out = deck_path.parent
    stem = safe_name(result.company_name, fallback=deck_path.stem)
    artifacts = {
        "pdf": render_pdf(
            result, out / f"{stem}_Financing_Probability_Report.pdf",
            page="A4" if page_size.upper() == "A4" else "LETTER",
        ),
        "data": write_json(
            build_lead_record(result, consent),
            out / f"{stem}_Financing_Probability_Data.json",
        ),
        "evidence": write_json(
            build_evidence_map(profile), out / f"{stem}_Extraction_Evidence.json"
        ),
    }

    summary = _summarise(result)

    # Best-effort: the artifacts above are already written and downloadable,
    # so a mail failure only ever adds a note to the results page.
    email = notify_analysis(result, artifacts, settings)
    summary["email"] = email.as_dict()
    if email.error:
        LOGGER.warning("Job %s: %s", job_id, email.summary)

    registry.update(job_id, state=JobState.DONE, artifacts=artifacts, summary=summary)
    LOGGER.info(
        "Job %s complete: %s, readiness %.0f, 6m %.0f%%",
        job_id, result.company_name, result.readiness.total, result.prediction.final.p6,
    )


def _summarise(result) -> dict:  # noqa: ANN001 - AnalysisResult
    """The headline numbers shown on the results page."""
    profile = result.profile
    return {
        "company": result.company_name,
        "sector": result.sector.label,
        "stage_declared": result.stage.declared.value,
        "stage_inferred": result.stage.inferred.value,
        "p3": result.prediction.final.p3,
        "p6": result.prediction.final.p6,
        "p12": result.prediction.final.p12,
        "readiness": result.readiness.total,
        "readiness_label": result.readiness.label,
        "confidence": result.confidence.score,
        "confidence_level": result.confidence.level,
        "comparable": result.comparable.label,
        "fit_score": result.ten_capital.fit_score,
        "next_step": result.ten_capital.recommended_next_step,
        "market_condition": result.prediction.market_condition.disclosure,
        "drivers": result.prediction.positive_signals[:5],
        "blockers": [
            {"severity": b.severity, "title": b.title, "detail": b.detail}
            for b in result.blockers
        ],
        "flags": [
            {"label": f.label, "value": f.value} for f in result.ten_capital.flags
        ],
        "actions": [
            {"action": a.action, "impact": a.impact, "timeframe": a.timeframe}
            for a in result.actions
        ],
        "fields_found": len(profile.known_fields()),
        "fields_total": 81,
        "llm": profile.llm_run,
    }
