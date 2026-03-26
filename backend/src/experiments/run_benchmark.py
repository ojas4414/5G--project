from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.algorithms import (
    CADMMAllocator,
    IndependentMAPPOAllocator,
    MAANAllocator,
    MAANConfig,
    OMDBanditAllocator,
    StaticGreedyAllocator,
)
from src.environment import FiveGEnvironment, SliceConfig


@dataclass
class ExpConfig:
    horizon: int = 300
    seeds: int = 5
    load_scales: tuple[float, ...] = (0.8, 1.0, 1.2, 1.4, 1.6)
    out_dir: str = "outputs"


def build_slice_configs(load_scale: float) -> List[SliceConfig]:
    return [
        SliceConfig("eMBB", r_min=45e6, d_max=0.028, alpha=1.2, beta=0.8, gamma=0.6, omega=90.0 * load_scale),
        SliceConfig("URLLC", r_min=15e6, d_max=0.008, alpha=1.0, beta=1.4, gamma=1.6, omega=60.0 * load_scale),
        SliceConfig("mMTC", r_min=6e6, d_max=0.050, alpha=0.9, beta=0.7, gamma=0.4, omega=45.0 * load_scale),
    ]


def make_algorithms(env: FiveGEnvironment) -> Dict[str, object]:
    s, k, m = env.s, env.k, env.m
    return {
        "MAAN": MAANAllocator(s, k, m, env.b_k, env.c_m, env.t_agg, MAANConfig()),
        "Independent_MAPPO": IndependentMAPPOAllocator(s, k, m, env.b_k, env.c_m, env.t_agg),
        "C_ADMM": CADMMAllocator(s, k, m, env.b_k, env.c_m, env.t_agg),
        "Static_Greedy": StaticGreedyAllocator(np.array([0.45, 0.35, 0.2]), s, k, m, env.b_k, env.c_m, env.t_agg),
        "OMD_BF": OMDBanditAllocator(s, k, m, env.b_k, env.c_m, env.t_agg),
    }


def run_one(name: str, alg, env: FiveGEnvironment, horizon: int) -> pd.DataFrame:
    rows = []
    state = env.reset()
    alg.reset()
    for t in range(horizon):
        out = alg.act(state)
        state, metrics = env.step(out.actions)
        alg.observe(state, metrics)
        rows.append(
            {
                "t": t,
                "algorithm": name,
                "utility_mean": float(np.mean(metrics["utilities"])),
                "rate_mean": float(np.mean(metrics["rates"])),
                "delay_mean": float(np.mean(metrics["delays"])),
                "qos_success": float(np.mean(metrics["qos_ok"])),
                "urlcc_delay": float(metrics["delays"][1]),
                "embb_rate": float(metrics["rates"][0]),
                "fairness_jain": float((np.sum(metrics["utilities"]) ** 2) / (len(metrics["utilities"]) * np.sum(metrics["utilities"] ** 2) + 1e-9)),
                "radio_util": float(np.mean(metrics["rate_utilization"])),
                "compute_util": float(np.mean(metrics["compute_utilization"])),
                "transport_util": float(metrics["transport_utilization"]),
                "d_radio": float(np.mean(metrics["d_radio"])),
                "d_trans": float(np.mean(metrics["d_trans"])),
                "d_comp": float(np.mean(metrics["d_comp"])),
            }
        )
    return pd.DataFrame(rows)


def run_experiment(cfg: ExpConfig) -> pd.DataFrame:
    frames = []
    for seed in range(cfg.seeds):
        for load_scale in cfg.load_scales:
            env = FiveGEnvironment(build_slice_configs(load_scale), seed=100 + seed)
            algs = make_algorithms(env)
            for name, alg in algs.items():
                df = run_one(name, alg, env, cfg.horizon)
                df["seed"] = seed
                df["load_scale"] = load_scale
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_all(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = df.groupby(["algorithm", "load_scale"], as_index=False).mean(numeric_only=True)

    plots = [
        ("utility_mean", "Mean Utility vs Load"),
        ("qos_success", "QoS Success Ratio vs Load"),
        ("delay_mean", "Mean Delay vs Load"),
        ("rate_mean", "Mean Rate vs Load"),
        ("fairness_jain", "Jain Fairness vs Load"),
        ("radio_util", "Radio Utilization vs Load"),
        ("compute_util", "Compute Utilization vs Load"),
        ("transport_util", "Transport Utilization vs Load"),
        ("urlcc_delay", "URLLC Delay vs Load"),
        ("embb_rate", "eMBB Rate vs Load"),
        ("d_radio", "Radio Delay Component vs Load"),
        ("d_trans", "Transport Delay Component vs Load"),
        ("d_comp", "Compute Delay Component vs Load"),
    ]
    for metric, title in plots:
        plt.figure(figsize=(7, 4))
        for alg, grp in summary.groupby("algorithm"):
            plt.plot(grp["load_scale"], grp[metric], marker="o", label=alg)
        plt.xlabel("Load Scale")
        plt.ylabel(metric)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_vs_load.png", dpi=150)
        plt.close()

    # 14th plot: convergence trend over time under highest load.
    high = df[df["load_scale"] == df["load_scale"].max()]
    ts = high.groupby(["algorithm", "t"], as_index=False)["utility_mean"].mean()
    plt.figure(figsize=(8, 4))
    for alg, grp in ts.groupby("algorithm"):
        smooth = grp["utility_mean"].rolling(10, min_periods=1).mean()
        plt.plot(grp["t"], smooth, label=alg)
    plt.xlabel("Time Slot")
    plt.ylabel("Smoothed Utility")
    plt.title("Convergence Trend at Highest Load")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "convergence_utility_high_load.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    cfg = ExpConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_experiment(cfg)
    result.to_csv(out_dir / "benchmark_results.csv", index=False)
    plot_all(result, out_dir / "plots")
    with open(out_dir / "config_used.json", "w", encoding="utf-8") as fp:
        import json

        json.dump(asdict(cfg), fp, indent=2)
    print(f"Saved results to: {out_dir.resolve()}")
