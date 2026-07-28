from __future__ import annotations

import json
import os
import sys
import threading
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.experiments.run_benchmark_phase2 import ExpConfig, plot_all, plot_publication_pack, run_experiment, save_tables

app = FastAPI(title="5G Network Slicing Benchmark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PHASE2_DIR = BASE_DIR / "outputs_phase2"
PHASE1_DIR = BASE_DIR / "outputs"
PHASE2_DIR.mkdir(parents=True, exist_ok=True)
PHASE1_DIR.mkdir(parents=True, exist_ok=True)

RESEARCH_LOCK = threading.Lock()
RESEARCH_JOBS: dict[str, dict] = {}
LATEST_RESEARCH_JOB_ID: str | None = None

# Job state is mirrored to disk. The in-memory dict alone is not enough: a free-tier
# instance sleeps after idling and loses the process, after which the dashboard would poll
# a job_id that 404s forever with no way to recover. Persisting lets a woken instance
# report the last known outcome instead.
JOBS_FILE = PHASE2_DIR / "research_jobs.json"
MAX_PERSISTED_JOBS = 20


class ResearchRunRequest(BaseModel):
    """Bounds are deliberately tight -- see DEMO_* defaults for the reasoning."""

    num_slices: int = Field(default=3, ge=3, le=6)
    load_center: float = Field(default=1.0, ge=0.6, le=2.0)
    seeds: int = Field(default=2, ge=1, le=6)
    horizon: int = Field(default=120, ge=50, le=600)
    n_mc_urlcc: int = Field(default=24, ge=4, le=128)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jobs_from_disk() -> None:
    """Restore job history written by a previous process (best effort)."""
    global LATEST_RESEARCH_JOB_ID
    if not JOBS_FILE.exists():
        return
    try:
        with open(JOBS_FILE, encoding="utf-8") as fp:
            payload = json.load(fp)
        jobs = payload.get("jobs", {})
        if not isinstance(jobs, dict):
            return
        for job in jobs.values():
            # A job still marked "running" cannot be running: its thread died with the
            # previous process. Report it honestly instead of leaving the UI spinning.
            if job.get("status") == "running":
                job["status"] = "failed"
                job["message"] = "Server restarted while this run was in progress."
                job["finished_at"] = job.get("finished_at") or _utc_now_iso()
        RESEARCH_JOBS.update(jobs)
        LATEST_RESEARCH_JOB_ID = payload.get("latest_job_id") or LATEST_RESEARCH_JOB_ID
    except (OSError, json.JSONDecodeError, AttributeError):
        # Corrupt or unreadable history must never stop the API from booting.
        pass


def _persist_jobs_locked() -> None:
    """Write job history to disk. Caller must hold RESEARCH_LOCK."""
    try:
        recent = sorted(RESEARCH_JOBS.values(), key=lambda j: j.get("started_at") or "", reverse=True)
        trimmed = {job["job_id"]: job for job in recent[:MAX_PERSISTED_JOBS]}
        tmp = JOBS_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump({"jobs": trimmed, "latest_job_id": LATEST_RESEARCH_JOB_ID}, fp, indent=2)
        tmp.replace(JOBS_FILE)
    except OSError:
        # Persistence is a convenience; losing it must not fail the run.
        pass


def _centered_load_scales(load_center: float) -> tuple[float, ...]:
    """Load sweep for a demo run.

    Three points rather than five: the total work is
    ``seeds * len(load_scales) * 5 algorithms * horizon`` steps, so dropping two loads cuts
    the run by 40%. Three points are still enough to show a trend against load.
    """
    offsets = (-0.3, 0.0, 0.3)
    vals = sorted({round(min(2.0, max(0.6, load_center + o)), 2) for o in offsets})
    return tuple(vals)


def _pretty_plot_title(path: Path) -> str:
    name = path.stem.replace("_", " ").strip()
    return " ".join(w.capitalize() for w in name.split())


def _list_png_files(folder: Path, url_prefix: str) -> list[dict]:
    if not folder.exists():
        return []
    files = sorted(folder.glob("*.png"))
    return [{"name": f.name, "title": _pretty_plot_title(f), "url": f"{url_prefix}/{f.name}"} for f in files]


def _update_job(job_id: str, *, persist: bool = False, **updates) -> None:
    with RESEARCH_LOCK:
        if job_id in RESEARCH_JOBS:
            RESEARCH_JOBS[job_id].update(updates)
            # Progress ticks stay in memory; only terminal transitions hit the disk, so a
            # long run does not rewrite the file on every one of its steps.
            if persist:
                _persist_jobs_locked()


def _run_research_job(job_id: str, req_data: dict) -> None:
    try:
        load_scales = _centered_load_scales(float(req_data["load_center"]))
        cfg = ExpConfig(
            horizon=int(req_data["horizon"]),
            seeds=int(req_data["seeds"]),
            n_mc_urlcc=int(req_data["n_mc_urlcc"]),
            load_scales=load_scales,
            num_slices=int(req_data["num_slices"]),
            out_dir="outputs_phase2",
        )
        _update_job(job_id, message="Running multi-algorithm benchmark...", progress=0.02)

        def on_progress(done: int, total: int, info: dict) -> None:
            pct = 0.05 + 0.65 * (done / max(total, 1))
            msg = f"Completed {done}/{total}: seed={info['seed']} load={info['load_scale']} alg={info['algorithm']}"
            _update_job(job_id, progress=min(pct, 0.70), message=msg)

        result = run_experiment(cfg, progress_callback=on_progress)
        _update_job(job_id, message="Saving tables...", progress=0.75)
        result.to_csv(PHASE2_DIR / "benchmark_results_phase2.csv", index=False)
        _, sig_df = save_tables(result, PHASE2_DIR)

        _update_job(job_id, message="Rendering core plots...", progress=0.82)
        plot_all(result, PHASE2_DIR / "plots")

        _update_job(job_id, message="Rendering publication plot pack...", progress=0.90)
        plot_publication_pack(result, PHASE2_DIR, sig_df)

        with open(PHASE2_DIR / "config_used.json", "w", encoding="utf-8") as fp:
            json.dump(asdict(cfg), fp, indent=2)

        _update_job(
            job_id,
            persist=True,
            status="completed",
            progress=1.0,
            message="Full research run completed.",
            finished_at=_utc_now_iso(),
        )
    except Exception as exc:
        _update_job(
            job_id,
            persist=True,
            status="failed",
            message=f"Research run failed: {exc}",
            error=traceback.format_exc(limit=8),
            finished_at=_utc_now_iso(),
        )


@app.get("/api/health")
def health_check():
    """Liveness probe. The `status` field is the contract -- keep it exactly "ok"."""
    with RESEARCH_LOCK:
        running = any(j.get("status") == "running" for j in RESEARCH_JOBS.values())
    return {
        "status": "ok",
        "service": "5g-slicing-benchmark",
        "results_available": (PHASE2_DIR / "summary_with_ci95.csv").exists()
        or (PHASE2_DIR / "benchmark_results_phase2.csv").exists()
        or (PHASE1_DIR / "benchmark_results.csv").exists(),
        "research_running": running,
    }


@app.get("/api/results")
def get_benchmark_results():
    summary = PHASE2_DIR / "summary_with_ci95.csv"
    raw = PHASE2_DIR / "benchmark_results_phase2.csv"
    legacy = PHASE1_DIR / "benchmark_results.csv"

    if summary.exists():
        return pd.read_csv(summary).to_dict(orient="records")
    if raw.exists():
        return pd.read_csv(raw).to_dict(orient="records")
    if legacy.exists():
        return pd.read_csv(legacy).to_dict(orient="records")
    return {"error": "Benchmark results not found"}


@app.get("/api/plots")
def get_plot_manifest():
    core_phase2 = _list_png_files(PHASE2_DIR / "plots", "/artifacts_phase2/plots")
    publication = _list_png_files(PHASE2_DIR / "plots_publication", "/artifacts_phase2/plots_publication")
    legacy = _list_png_files(PHASE1_DIR / "plots", "/artifacts_phase1/plots")

    return {
        "core": core_phase2,
        "publication": publication,
        "legacy": legacy,
        "counts": {"core": len(core_phase2), "publication": len(publication), "legacy": len(legacy)},
    }


@app.post("/api/research/start")
def start_research_run(req: ResearchRunRequest):
    global LATEST_RESEARCH_JOB_ID
    req_data = req.model_dump()
    with RESEARCH_LOCK:
        running_job = next((j for j in RESEARCH_JOBS.values() if j.get("status") == "running"), None)
        if running_job is not None:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": "A full research run is already in progress.",
                    "running_job": running_job,
                },
            )

        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "status": "running",
            "progress": 0.0,
            "message": "Queued full research run.",
            "params": req_data,
            "started_at": _utc_now_iso(),
            "finished_at": None,
            "error": None,
            "out_dir": str(PHASE2_DIR),
        }
        RESEARCH_JOBS[job_id] = job
        LATEST_RESEARCH_JOB_ID = job_id
        _persist_jobs_locked()

    worker = threading.Thread(target=_run_research_job, args=(job_id, req_data), daemon=True)
    worker.start()
    return {"job": job}


@app.get("/api/research/status")
def get_latest_research_status():
    with RESEARCH_LOCK:
        if LATEST_RESEARCH_JOB_ID is None:
            return {"job": None, "status": "idle"}
        return {"job": RESEARCH_JOBS.get(LATEST_RESEARCH_JOB_ID), "status": "ok"}


@app.get("/api/research/status/{job_id}")
def get_research_status(job_id: str):
    with RESEARCH_LOCK:
        job = RESEARCH_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Research job not found")
    return {"job": job}


app.mount("/artifacts_phase2", StaticFiles(directory=PHASE2_DIR), name="artifacts_phase2")
app.mount("/artifacts_phase1", StaticFiles(directory=PHASE1_DIR), name="artifacts_phase1")

_load_jobs_from_disk()


if __name__ == "__main__":
    import uvicorn

    # Render (and most PaaS hosts) inject the port to bind; hardcoding 8000 makes the
    # service unreachable there. Reload is a local-development convenience only -- it
    # forks a watcher process, which is wasteful on a 512 MB instance.
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("DEV_RELOAD", "").lower() in {"1", "true", "yes"}
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
