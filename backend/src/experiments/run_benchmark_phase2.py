from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from src.algorithms import (
    CADMMAllocator,
    IndependentMAPPOPPOAllocator,
    MAANPPOAllocator,
    OMDBanditAllocator,
    StaticGreedyAllocator,
)
from src.environment import FiveGEnvironment, SliceConfig


@dataclass
class ExpConfig:
    horizon: int = 500
    seeds: int = 6
    n_mc_urlcc: int = 64
    load_scales: tuple[float, ...] = (0.8, 1.0, 1.2, 1.4, 1.6)
    out_dir: str = "outputs_phase2"


def build_slice_configs(load_scale: float) -> List[SliceConfig]:
    return [
        SliceConfig("eMBB", r_min=45e6, d_max=0.028, alpha=1.2, beta=0.8, gamma=0.6, omega=95.0 * load_scale),
        SliceConfig("URLLC", r_min=15e6, d_max=0.008, alpha=1.0, beta=1.5, gamma=1.8, omega=65.0 * load_scale, eps_urlcc=0.01),
        SliceConfig("mMTC", r_min=6e6, d_max=0.050, alpha=0.9, beta=0.7, gamma=0.4, omega=50.0 * load_scale),
    ]


def make_algorithms(env: FiveGEnvironment) -> Dict[str, object]:
    s, k, m = env.s, env.k, env.m
    return {
        "MAAN_PPO": MAANPPOAllocator(s, k, m, env.b_k, env.c_m, env.t_agg),
        "Independent_MAPPO_PPO": IndependentMAPPOPPOAllocator(s, k, m, env.b_k, env.c_m, env.t_agg),
        "C_ADMM": CADMMAllocator(s, k, m, env.b_k, env.c_m, env.t_agg),
        "Static_Greedy": StaticGreedyAllocator(np.array([0.45, 0.35, 0.2]), s, k, m, env.b_k, env.c_m, env.t_agg),
        "OMD_BF": OMDBanditAllocator(s, k, m, env.b_k, env.c_m, env.t_agg),
    }


def run_one(name: str, alg, env: FiveGEnvironment, horizon: int, n_mc_urlcc: int) -> pd.DataFrame:
    rows = []
    state = env.reset()
    alg.reset()
    last_actions = None
    for t in range(horizon):
        out = alg.act(state)
        last_actions = out.actions
        state, metrics = env.step(out.actions)
        alg.observe(state, metrics)
        urlcc_prob = env.saa_urlcc_violation_probability(last_actions, n_mc=n_mc_urlcc, urlcc_idx=1)
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
                "urlcc_violation_prob_saa": urlcc_prob,
            }
        )
    return pd.DataFrame(rows)


def ci95(x: np.ndarray) -> tuple[float, float]:
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(max(len(x), 1))) if len(x) > 1 else 0.0
    return m, 1.96 * se


def run_experiment(cfg: ExpConfig) -> pd.DataFrame:
    frames = []
    for seed in range(cfg.seeds):
        for load_scale in cfg.load_scales:
            env = FiveGEnvironment(build_slice_configs(load_scale), seed=1000 + seed)
            algs = make_algorithms(env)
            for name, alg in algs.items():
                df = run_one(name, alg, env, cfg.horizon, cfg.n_mc_urlcc)
                df["seed"] = seed
                df["load_scale"] = load_scale
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def save_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = (
        df.groupby(["algorithm", "seed", "load_scale"], as_index=False)[
            ["utility_mean", "qos_success", "delay_mean", "urlcc_violation_prob_saa", "fairness_jain"]
        ]
        .mean()
    )
    rows = []
    for (alg, load), grp in final.groupby(["algorithm", "load_scale"]):
        row = {"algorithm": alg, "load_scale": load}
        for met in ["utility_mean", "qos_success", "delay_mean", "urlcc_violation_prob_saa", "fairness_jain"]:
            m, ci = ci95(grp[met].to_numpy())
            row[f"{met}_mean"] = m
            row[f"{met}_ci95"] = ci
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "summary_with_ci95.csv", index=False)

    # --- Statistical Significance Tests (MAAN_PPO vs others) ---
    target_alg = "MAAN_PPO"
    if target_alg in final["algorithm"].values:
        sig_rows = []
        for load in final["load_scale"].unique():
            base_data = final[(final["algorithm"] == target_alg) & (final["load_scale"] == load)]
            for alg in final["algorithm"].unique():
                if alg == target_alg:
                    continue
                comp_data = final[(final["algorithm"] == alg) & (final["load_scale"] == load)]
                row = {"load_scale": load, "algorithm": alg}
                for met in ["utility_mean", "qos_success", "delay_mean"]:
                    stat, pval = stats.ttest_ind(base_data[met], comp_data[met], equal_var=False)
                    row[f"{met}_pval_vs_{target_alg}"] = pval
                sig_rows.append(row)
        if sig_rows:
            pd.DataFrame(sig_rows).to_csv(out_dir / "statistical_significance.csv", index=False)


def plot_all(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = df.groupby(["algorithm", "load_scale"], as_index=False).mean(numeric_only=True)
    metrics = [
        "utility_mean",
        "qos_success",
        "delay_mean",
        "rate_mean",
        "fairness_jain",
        "radio_util",
        "compute_util",
        "transport_util",
        "urlcc_delay",
        "embb_rate",
        "d_radio",
        "d_trans",
        "d_comp",
        "urlcc_violation_prob_saa",
    ]
    for metric in metrics:
        plt.figure(figsize=(7, 4))
        for alg, grp in summary.groupby("algorithm"):
            plt.plot(grp["load_scale"], grp[metric], marker="o", label=alg)
        plt.xlabel("Load Scale")
        plt.ylabel(metric)
        plt.title(f"{metric} vs load")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_vs_load.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    cfg = ExpConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_experiment(cfg)
    result.to_csv(out_dir / "benchmark_results_phase2.csv", index=False)
    save_tables(result, out_dir)
    plot_all(result, out_dir / "plots")
    with open(out_dir / "config_used.json", "w", encoding="utf-8") as fp:
        import json

        json.dump(asdict(cfg), fp, indent=2)
    print(f"Saved phase2 results to: {out_dir.resolve()}")
