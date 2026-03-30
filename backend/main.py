from __future__ import annotations

import json
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


class ResearchRunRequest(BaseModel):
    num_slices: int = Field(default=3, ge=3, le=12)
    load_center: float = Field(default=1.0, ge=0.6, le=2.0)
    seeds: int = Field(default=4, ge=1, le=10)
    horizon: int = Field(default=300, ge=50, le=1500)
    n_mc_urlcc: int = Field(default=32, ge=4, le=256)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _centered_load_scales(load_center: float) -> tuple[float, ...]:
    offsets = (-0.4, -0.2, 0.0, 0.2, 0.4)
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


def _update_job(job_id: str, **updates) -> None:
    with RESEARCH_LOCK:
        if job_id in RESEARCH_JOBS:
            RESEARCH_JOBS[job_id].update(updates)


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
            status="completed",
            progress=1.0,
            message="Full research run completed.",
            finished_at=_utc_now_iso(),
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            message=f"Research run failed: {exc}",
            error=traceback.format_exc(limit=8),
            finished_at=_utc_now_iso(),
        )


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
