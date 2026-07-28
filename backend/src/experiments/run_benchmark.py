"""Phase 1 -- quick benchmark.

This used to carry its own copy of the slice definitions, trace generation, environment
wiring and plotting. Keeping two copies meant the physics could (and did) drift apart, so
Phase 1 is now a thin quick-mode wrapper over the Phase 2 pipeline: same corrected M/M/1
delay model, same algorithms, same metrics -- just fewer seeds and a shorter horizon.

Use it for a fast sanity check; use ``run_benchmark_phase2`` for the full study.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.experiments.run_benchmark_phase2 import ExpConfig as Phase2Config
from src.experiments.run_benchmark_phase2 import plot_all, run_experiment as run_phase2, save_tables


@dataclass
class ExpConfig:
    horizon: int = 200
    seeds: int = 3
    load_scales: tuple[float, ...] = (0.8, 1.0, 1.2, 1.4, 1.6)
    num_slices: int = 3
    n_mc_urlcc: int = 16
    out_dir: str = "outputs"


def run_experiment(cfg: ExpConfig) -> pd.DataFrame:
    return run_phase2(
        Phase2Config(
            horizon=cfg.horizon,
            seeds=cfg.seeds,
            n_mc_urlcc=cfg.n_mc_urlcc,
            load_scales=cfg.load_scales,
            num_slices=cfg.num_slices,
            out_dir=cfg.out_dir,
        )
    )


if __name__ == "__main__":
    cfg = ExpConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_experiment(cfg)
    result.to_csv(out_dir / "benchmark_results.csv", index=False)
    save_tables(result, out_dir)
    plot_all(result, out_dir / "plots")
    with open(out_dir / "config_used.json", "w", encoding="utf-8") as fp:
        json.dump(asdict(cfg), fp, indent=2)
    print(f"Saved results to: {out_dir.resolve()}")
