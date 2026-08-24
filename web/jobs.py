"""In-process job registry for the web app.

Analysis takes tens of seconds once Claude is reading the deck, which is far
too long to hold an HTTP request open behind a proxy. Uploads therefore start
a background job and the browser polls for completion.

Jobs live in memory with their artifacts on local disk. That is the right fit
for a single Railway instance and a deliberately short retention window - the
uploaded deck is confidential, so it is deleted as soon as the job expires.
Running more than one replica would need shared storage instead.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

#: How long a finished job's files remain downloadable.
JOB_TTL_SECONDS = 30 * 60

#: Hard ceiling on concurrently retained jobs, oldest evicted first.
MAX_JOBS = 40

#: Presentation order for the download list.
DOWNLOAD_ORDER = ("pdf", "data", "evidence", "html")


class JobState(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    ANALYSING = "analysing"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"


#: User-facing progress text. The web UI shows these verbatim.
STATE_MESSAGES: dict[JobState, str] = {
    JobState.QUEUED: "Queued",
    JobState.EXTRACTING: "Reading the deck and extracting financing facts",
    JobState.ANALYSING: "Scoring readiness and computing probabilities",
    JobState.RENDERING: "Generating the report",
    JobState.DONE: "Complete",
    JobState.FAILED: "Failed",
}


@dataclass
class Job:
    id: str
    filename: str
    workdir: Path
    created_at: float = field(default_factory=time.monotonic)
    state: JobState = JobState.QUEUED
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.created_at > JOB_TTL_SECONDS

    @property
    def finished(self) -> bool:
        return self.state in (JobState.DONE, JobState.FAILED)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "state": self.state.value,
            "message": STATE_MESSAGES[self.state],
            "error": self.error,
            "summary": self.summary,
            # Ordered by what the reader wants first, not alphabetically -
            # the PDF is the deliverable, the JSON files are the audit trail.
            "downloads": sorted(
                self.artifacts, key=lambda key: DOWNLOAD_ORDER.index(key)
                if key in DOWNLOAD_ORDER else len(DOWNLOAD_ORDER)
            ),
        }


class JobRegistry:
    """Thread-safe job store with TTL eviction."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        job_id = uuid.uuid4().hex
        workdir = self.root / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, filename=filename, workdir=workdir)
        with self._lock:
            self._jobs[job_id] = job
            self._evict_locked()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.expired:
                self._discard_locked(job_id)
                return None
            return job

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in changes.items():
                setattr(job, key, value)

    def purge_expired(self) -> int:
        with self._lock:
            stale = [job_id for job_id, job in self._jobs.items() if job.expired]
            for job_id in stale:
                self._discard_locked(job_id)
        if stale:
            LOGGER.info("Purged %d expired job(s).", len(stale))
        return len(stale)

    def shutdown(self) -> None:
        with self._lock:
            for job_id in list(self._jobs):
                self._discard_locked(job_id)

    # -------------------------------------------------------------- internal
    def _evict_locked(self) -> None:
        if len(self._jobs) <= MAX_JOBS:
            return
        oldest = sorted(self._jobs.values(), key=lambda j: j.created_at)
        for job in oldest[: len(self._jobs) - MAX_JOBS]:
            self._discard_locked(job.id)

    def _discard_locked(self, job_id: str) -> None:
        job = self._jobs.pop(job_id, None)
        if job:
            # The uploaded deck is confidential; remove it with the job.
            shutil.rmtree(job.workdir, ignore_errors=True)
