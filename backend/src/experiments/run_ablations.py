from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.algorithms import CADMMAllocator
from src.environment import FiveGEnvironment
from src.experiments.run_benchmark_phase2 import build_slice_configs


def run_admm_ablation(seeds: int = 5, horizon: int = 200, load_scale: float = 1.2) -> pd.DataFrame:
    print(f"Running C_ADMM convergence rounds ablation at load={load_scale}...")
    rounds_list = [1, 2, 4, 8, 16]
    rows = []

    for seed in range(seeds):
        env = FiveGEnvironment(build_slice_configs(load_scale), seed=2000 + seed)
        for r in rounds_list:
            alg = CADMMAllocator(env.s, env.k, env.m, env.b_k, env.c_m, env.t_agg, rounds=r)
            state = env.reset()
            alg.reset()
            for t in range(horizon):
                out = alg.act(state)
                state, metrics = env.step(out.actions)
                rows.append({
                    "seed": seed,
                    "rounds": r,
                    "t": t,
                    "utility": float(np.mean(metrics["utilities"])),
                    "qos_success": float(np.mean(metrics["qos_ok"]))
                })

    return pd.DataFrame(rows)


def plot_ablation(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Average over time first to get per-seed performance
    summary = df.groupby(["rounds", "seed"], as_index=False).mean(numeric_only=True)
    # Average across seeds
    final = summary.groupby("rounds", as_index=False).mean(numeric_only=True)

    plt.figure(figsize=(7, 4))
    plt.plot(final["rounds"], final["utility"], marker="o", color="blue", linewidth=2, label="Mean Utility")
    plt.xlabel("C_ADMM Convergence Rounds (Communication Overhead)")
    plt.ylabel("Mean Utility")
    plt.title("Ablation: Utility vs. Communication Rounds")
    plt.grid(True, alpha=0.3)
    plt.xticks(final["rounds"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_admm_rounds_utility.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    out_dir = Path("outputs_phase2/ablations")
    df_admm = run_admm_ablation()
    df_admm.to_csv(out_dir / "admm_rounds_ablation.csv", index=False)
    plot_ablation(df_admm, out_dir)
    print(f"Ablation results saved to: {out_dir.resolve()}")