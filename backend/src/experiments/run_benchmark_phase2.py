from __future__ import annotations

import gc
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, List

import matplotlib

# Force the non-interactive backend before pyplot is imported. Plots are rendered from a
# worker thread on a headless server; any GUI backend would either fail to import or try
# to touch a main-thread-only event loop.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as stats  # noqa: E402

from src.algorithms import (
    CADMMAllocator,
    IndependentMAPPOPPOAllocator,
    MAANPPOAllocator,
    OMDBanditAllocator,
    StaticGreedyAllocator,
)
from src.environment import FiveGEnvironment, SliceConfig


ALGORITHM_ORDER = (
    "MAAN_PPO",
    "Independent_MAPPO_PPO",
    "C_ADMM",
    "Static_Greedy",
    "OGD_Bandit",
)

ALGO_COLORS = {
    "MAAN_PPO": "#1f77b4",
    "Independent_MAPPO_PPO": "#ff7f0e",
    "C_ADMM": "#2ca02c",
    "Static_Greedy": "#d62728",
    "OGD_Bandit": "#9467bd",
}

ALGO_SHORT = {
    "MAAN_PPO": "MAAN",
    "Independent_MAPPO_PPO": "Ind-MAPPO",
    "C_ADMM": "C-ADMM",
    "Static_Greedy": "Static+Greedy",
    "OGD_Bandit": "OGD-Bandit",
}

# Render resolution for every saved figure. A rasterised figure is held in memory at
# roughly (w*dpi) x (h*dpi) x 4 bytes, and `bbox_inches="tight"` renders twice, so this
# is the main driver of peak RSS during the plotting phase. 140 dpi keeps the gallery
# crisp on screen at a fraction of the memory (and produces smaller PNGs to serve).
PLOT_DPI = int(os.environ.get("PLOT_DPI", "140"))

LOAD_METRICS = {
    "utility_mean": ("Mean Utility", "Higher is better"),
    "qos_success": ("QoS Success Ratio", "Higher is better"),
    "delay_mean": ("Mean Delay (s)", "Lower is better"),
    "rate_mean": ("Mean Rate (bps)", "Higher is better"),
    "fairness_jain": ("Jain Fairness", "Higher is better"),
    "radio_util": ("Radio Utilization", "Target near 1.0"),
    "compute_util": ("Compute Utilization", "Target near 1.0"),
    "transport_util": ("Transport Utilization", "Target near 1.0"),
    "urlcc_delay": ("URLLC Delay (s)", "Lower is better"),
    "embb_rate": ("eMBB Rate (bps)", "Higher is better"),
    "d_radio": ("Radio Delay Component (s)", "Lower is better"),
    "d_trans": ("Transport Delay Component (s)", "Lower is better"),
    "d_comp": ("Compute Delay Component (s)", "Lower is better"),
    "urlcc_violation_prob_saa": ("URLLC Violation Probability (SAA)", "Lower is better"),
}


@dataclass
class ExpConfig:
    horizon: int = 500
    seeds: int = 6
    n_mc_urlcc: int = 64
    load_scales: tuple[float, ...] = (0.8, 1.0, 1.2, 1.4, 1.6)
    num_slices: int = 3
    out_dir: str = "outputs_phase2"


def build_slice_configs(load_scale: float, num_slices: int = 3) -> List[SliceConfig]:
    """Slice SLAs. See ``fiveg_env`` for units.

    ``load_scale`` is applied to the traffic traces (see :func:`generate_common_traces`),
    not here: the SLA of a slice does not change when the network gets busier. ``omega``
    is a fixed processing density in cycles/bit -- URLLC costs the most compute per bit
    (tight scheduling, redundancy), mMTC the least.
    """
    base = [
        SliceConfig("eMBB", r_min=22e6, d_max=0.030, alpha=1.2, beta=0.8, gamma=0.6, omega=30.0),
        SliceConfig("URLLC", r_min=6e6, d_max=0.008, alpha=1.0, beta=1.5, gamma=1.8, omega=45.0, eps_urlcc=0.01),
        SliceConfig("mMTC", r_min=3e6, d_max=0.050, alpha=0.9, beta=0.7, gamma=0.4, omega=20.0),
    ]
    if num_slices <= 3:
        return base[:num_slices]

    extra = []
    for idx in range(num_slices - 3):
        extra.append(
            SliceConfig(
                name=f"mMTC_{idx + 2}",
                r_min=2.5e6,
                d_max=0.060,
                alpha=0.85,
                beta=0.65,
                gamma=0.35,
                omega=18.0,
            )
        )
    return base + extra


def static_slice_weights(num_slices: int) -> np.ndarray:
    if num_slices <= 0:
        return np.array([], dtype=float)
    if num_slices == 1:
        return np.array([1.0], dtype=float)
    if num_slices == 2:
        return np.array([0.56, 0.44], dtype=float)
    # Keep eMBB/URLLC emphasis and distribute the rest across mMTC-like slices.
    tail = np.full(num_slices - 2, 0.20 / max(num_slices - 2, 1), dtype=float)
    w = np.concatenate([np.array([0.45, 0.35], dtype=float), tail])
    return w / np.sum(w)


def generate_common_traces(
    s: int, k: int, horizon: int, seed: int, load_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Traffic and channel traces shared by every algorithm in a scenario.

    ``load_scale`` multiplies the offered traffic, which is what "load" means on every
    x-axis in this study. All algorithms in a scenario see byte-identical traces, so the
    per-seed comparison is a paired design (see :func:`save_tables`).
    """
    rng = np.random.default_rng(seed)
    lambda_trace = load_scale * rng.uniform(FiveGEnvironment.lam_low, FiveGEnvironment.lam_high, size=(horizon + 1, s))
    channel_trace = rng.exponential(scale=1.0, size=(horizon + 1, s, k))
    return lambda_trace, channel_trace


def make_algorithm(name: str, env: FiveGEnvironment):
    s, k, m = env.s, env.k, env.m
    r_min = env.r_min
    d_max = env.d_max
    omega = env.omega
    alpha = np.array([cfg.alpha for cfg in env.slice_configs], dtype=float)
    beta = np.array([cfg.beta for cfg in env.slice_configs], dtype=float)
    if name == "MAAN_PPO":
        return MAANPPOAllocator(s, k, m, env.b_k, env.c_m, env.t_agg, r_min=r_min, d_max=d_max)
    if name == "Independent_MAPPO_PPO":
        return IndependentMAPPOPPOAllocator(s, k, m, env.b_k, env.c_m, env.t_agg, r_min=r_min, d_max=d_max)
    if name == "C_ADMM":
        return CADMMAllocator(
            s, k, m, env.b_k, env.c_m, env.t_agg,
            r_min=r_min, d_max=d_max, omega=omega, alpha=alpha, beta=beta,
            w_prb=env.w_prb, n0=env.n0, l_pkt=env.l_pkt,
        )
    if name == "Static_Greedy":
        return StaticGreedyAllocator(
            static_slice_weights(s), s, k, m, env.b_k, env.c_m, env.t_agg,
            r_min=r_min, d_max=d_max, omega=omega, l_pkt=env.l_pkt,
        )
    if name == "OGD_Bandit":
        return OMDBanditAllocator(s, k, m, env.b_k, env.c_m, env.t_agg, d_max=d_max)
    raise ValueError(f"Unknown algorithm: {name}")


def _jain_index(x: np.ndarray) -> float:
    """Jain's fairness index, (sum x)^2 / (n * sum x^2), for non-negative x."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    denom = len(x) * float(np.sum(x**2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x) ** 2 / denom)


def ci95(x: np.ndarray) -> tuple[float, float]:
    """Mean and 95% CI half-width using Student's t.

    With the handful of seeds used here the normal quantile 1.96 understates the interval
    by roughly 30% (t(0.975, 5) = 2.571), so the t quantile is used instead.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(np.mean(x)) if n else float("nan")
    if n > 1:
        se = float(np.std(x, ddof=1) / np.sqrt(n))
        return m, float(stats.t.ppf(0.975, n - 1)) * se
    return m, 0.0


def _safe_series_mean(arr: np.ndarray | None) -> float:
    if arr is None or len(arr) == 0:
        return np.nan
    return float(np.mean(arr))


def _extract_price_info(alg, k: int, m: int) -> tuple[float, float, float, float]:
    if hasattr(alg, "use_prices") and not bool(getattr(alg, "use_prices")):
        # Independent MAPPO has no market/pricing mechanism; report as missing, not zero.
        return np.nan, np.nan, np.nan, np.nan
    if hasattr(alg, "prices"):
        p = np.asarray(getattr(alg, "prices"), dtype=float)
        if p.size >= k + m + 1:
            p_b = _safe_series_mean(p[:k])
            p_c = _safe_series_mean(p[k : k + m])
            p_t = float(p[k + m])
            p_all = _safe_series_mean(p)
            return p_b, p_c, p_t, p_all
    if hasattr(alg, "mu_b") and hasattr(alg, "mu_c") and hasattr(alg, "mu_t"):
        mu_b = np.asarray(getattr(alg, "mu_b"), dtype=float)
        mu_c = np.asarray(getattr(alg, "mu_c"), dtype=float)
        mu_t = float(getattr(alg, "mu_t"))
        all_p = np.concatenate([mu_b, mu_c, np.array([mu_t])])
        return _safe_series_mean(mu_b), _safe_series_mean(mu_c), mu_t, _safe_series_mean(all_p)
    return np.nan, np.nan, np.nan, np.nan


def run_one(name: str, alg, env: FiveGEnvironment, horizon: int, n_mc_urlcc: int) -> pd.DataFrame:
    rows = []
    state = env.reset()
    alg.reset()
    for t in range(horizon):
        t0 = time.perf_counter()
        out = alg.act(state)
        t1 = time.perf_counter()
        state, metrics = env.step(out.actions)
        t2 = time.perf_counter()
        alg.observe(state, metrics)
        t3 = time.perf_counter()
        urlcc_prob = env.saa_urlcc_violation_probability(out.actions, n_mc=n_mc_urlcc, urlcc_idx=1)
        t4 = time.perf_counter()

        p_b, p_c, p_t, p_all = _extract_price_info(alg, env.k, env.m)
        aux = out.aux if out.aux is not None else {}
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
                # Jain's index is only meaningful on a non-negative quantity, so it is
                # computed on achieved rates. Applying it to utilities (which are signed)
                # produces a number in [0, 1] that means nothing.
                "fairness_jain": _jain_index(metrics["rates"]),
                "radio_util": float(np.mean(metrics["rate_utilization"])),
                "compute_util": float(np.mean(metrics["compute_utilization"])),
                "transport_util": float(metrics["transport_utilization"]),
                "d_radio": float(np.mean(metrics["d_radio"])),
                "d_trans": float(np.mean(metrics["d_trans"])),
                "d_comp": float(np.mean(metrics["d_comp"])),
                "urlcc_violation_prob_saa": urlcc_prob,
                "act_runtime_ms": float((t1 - t0) * 1e3),
                "env_runtime_ms": float((t2 - t1) * 1e3),
                "observe_runtime_ms": float((t3 - t2) * 1e3),
                "saa_runtime_ms": float((t4 - t3) * 1e3),
                "algo_runtime_ms": float((t1 - t0 + t3 - t2) * 1e3),
                "total_step_runtime_ms": float((t4 - t0) * 1e3),
                "price_radio_mean": p_b,
                "price_compute_mean": p_c,
                "price_transport": p_t,
                "price_total_mean": p_all,
                "cadmm_r_pri": float(aux.get("r_pri", np.nan)),
                "cadmm_r_dual": float(aux.get("r_dual", np.nan)),
                "cadmm_rounds": float(aux.get("rounds", np.nan)),
                "cadmm_rho": float(aux.get("rho", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def run_experiment(
    cfg: ExpConfig,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    frames = []
    total_runs = cfg.seeds * len(cfg.load_scales) * len(ALGORITHM_ORDER)
    completed_runs = 0
    for seed in range(cfg.seeds):
        for load_scale in cfg.load_scales:
            slice_cfgs = build_slice_configs(load_scale, num_slices=cfg.num_slices)
            scenario_seed = 1000 + 97 * seed + int(round(100 * load_scale))
            lambda_trace, channel_trace = generate_common_traces(
                s=len(slice_cfgs), k=3, horizon=cfg.horizon, seed=scenario_seed, load_scale=load_scale
            )

            for alg_idx, alg_name in enumerate(ALGORITHM_ORDER):
                alg_seed = scenario_seed + 31 * (alg_idx + 1)
                np.random.seed(alg_seed)
                try:
                    import torch

                    torch.manual_seed(alg_seed)
                except Exception:
                    pass

                env = FiveGEnvironment(
                    slice_cfgs,
                    seed=scenario_seed,
                    lambda_trace=lambda_trace,
                    channel_trace=channel_trace,
                )
                alg = make_algorithm(alg_name, env)
                df = run_one(alg_name, alg, env, cfg.horizon, cfg.n_mc_urlcc)
                df["seed"] = seed
                df["load_scale"] = load_scale
                frames.append(df)
                completed_runs += 1
                if progress_callback is not None:
                    progress_callback(
                        completed_runs,
                        total_runs,
                        {"seed": seed, "load_scale": load_scale, "algorithm": alg_name},
                    )
    return pd.concat(frames, ignore_index=True)


def save_tables(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = (
        df.groupby(["algorithm", "seed", "load_scale"], as_index=False)[
            ["utility_mean", "qos_success", "delay_mean", "urlcc_violation_prob_saa", "fairness_jain", "algo_runtime_ms"]
        ]
        .mean()
    )
    rows = []
    for (alg, load), grp in final.groupby(["algorithm", "load_scale"]):
        row = {"algorithm": alg, "load_scale": load}
        for met in ["utility_mean", "qos_success", "delay_mean", "urlcc_violation_prob_saa", "fairness_jain", "algo_runtime_ms"]:
            m, ci = ci95(grp[met].to_numpy())
            row[f"{met}_mean"] = m
            row[f"{met}_ci95"] = ci
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "summary_with_ci95.csv", index=False)

    sig_df = _significance_table(final, metrics=["utility_mean", "qos_success", "delay_mean"])
    if not sig_df.empty:
        sig_df.to_csv(out_dir / "statistical_significance.csv", index=False)
    return summary_df, sig_df


def _holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (family-wise error rate control)."""
    p = np.asarray(pvals, dtype=float)
    finite = np.isfinite(p)
    adj = np.full_like(p, np.nan)
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        return adj
    order = idx[np.argsort(p[idx])]
    n = order.size
    running = 0.0
    for rank, pos in enumerate(order):
        running = max(running, (n - rank) * p[pos])
        adj[pos] = min(running, 1.0)
    return adj


def _significance_table(final: pd.DataFrame, metrics: List[str], target_alg: str = "MAAN_PPO") -> pd.DataFrame:
    """Paired t-tests of every algorithm against ``target_alg``, per load.

    All algorithms in a scenario are driven by identical traffic and channel traces
    (see :func:`generate_common_traces`), so seeds pair one-to-one across algorithms.
    That makes this a repeated-measures design: ``ttest_rel`` is the correct test and is
    strictly more powerful than the unpaired Welch test. Within each metric, p-values are
    then Holm-Bonferroni adjusted across all (algorithm, load) comparisons -- a 5-load
    sweep against 4 competitors is 20 simultaneous tests per metric, so uncorrected
    p-values would overstate significance badly.
    """
    if target_alg not in set(final["algorithm"]):
        return pd.DataFrame()

    rows = []
    for load in sorted(final["load_scale"].unique()):
        base = final[(final["algorithm"] == target_alg) & (final["load_scale"] == load)].sort_values("seed")
        for alg in ALGORITHM_ORDER:
            if alg == target_alg or alg not in set(final["algorithm"]):
                continue
            comp = final[(final["algorithm"] == alg) & (final["load_scale"] == load)].sort_values("seed")
            row = {"load_scale": load, "algorithm": alg, "n_seeds": int(len(base))}
            for met in metrics:
                a = base[met].to_numpy()
                b = comp[met].to_numpy()
                # ttest_rel needs >= 2 pairs and non-identical samples to be defined.
                if len(a) != len(b) or len(a) < 2 or np.allclose(a, b):
                    pval = float("nan")
                else:
                    pval = float(stats.ttest_rel(a, b).pvalue)
                row[f"{met}_pval_vs_{target_alg}"] = pval
                row[f"{met}_delta_vs_{target_alg}"] = float(np.mean(a) - np.mean(b))
            rows.append(row)

    sig_df = pd.DataFrame(rows)
    if sig_df.empty:
        return sig_df
    # Correct across the entire family of tests in this table.
    for met in metrics:
        col = f"{met}_pval_vs_{target_alg}"
        sig_df[f"{met}_padj_vs_{target_alg}"] = _holm_bonferroni(sig_df[col].to_numpy())
    return sig_df


def _add_caption(text: str) -> None:
    plt.figtext(0.5, -0.02, text, ha="center", fontsize=9)


def _plot_load_metric_with_ci(df: pd.DataFrame, metric: str, ylabel: str, caption: str, out_path: Path) -> None:
    per_seed = df.groupby(["algorithm", "seed", "load_scale"], as_index=False)[metric].mean()
    loads = sorted(per_seed["load_scale"].unique())
    plt.figure(figsize=(8, 4.8))
    for alg in ALGORITHM_ORDER:
        grp = per_seed[per_seed["algorithm"] == alg]
        means = []
        cis = []
        for ld in loads:
            vals = grp[grp["load_scale"] == ld][metric].to_numpy()
            m, ci = ci95(vals)
            means.append(m)
            cis.append(ci)
        means = np.array(means)
        cis = np.array(cis)
        color = ALGO_COLORS.get(alg, None)
        plt.plot(loads, means, marker="o", linewidth=2, label=ALGO_SHORT.get(alg, alg), color=color)
        plt.fill_between(loads, means - cis, means + cis, alpha=0.15, color=color)
    plt.xlabel("Load Scale")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Load (mean +/- 95% CI)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncols=2, fontsize=9)
    _add_caption(caption)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def plot_all(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, (ylabel, note) in LOAD_METRICS.items():
        _plot_load_metric_with_ci(
            df=df,
            metric=metric,
            ylabel=ylabel,
            caption=f"Takeaway: {note} while preserving stability under increasing offered load.",
            out_path=out_dir / f"{metric}_vs_load.png",
        )
    plt.close("all")
    gc.collect()


def plot_convergence_with_ci(df: pd.DataFrame, out_dir: Path, high_load: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("utility_mean", "Utility"),
        ("qos_success", "QoS Success Ratio"),
        ("urlcc_delay", "URLLC Delay (s)"),
    ]
    for metric, ylabel in metrics:
        sub = df[df["load_scale"] == high_load]
        per_seed_t = sub.groupby(["algorithm", "seed", "t"], as_index=False)[metric].mean()
        plt.figure(figsize=(8, 4.8))
        for alg in ALGORITHM_ORDER:
            grp = per_seed_t[per_seed_t["algorithm"] == alg]
            ts = sorted(grp["t"].unique())
            means = []
            cis = []
            for t in ts:
                vals = grp[grp["t"] == t][metric].to_numpy()
                m, ci = ci95(vals)
                means.append(m)
                cis.append(ci)
            means = pd.Series(means).rolling(10, min_periods=1).mean().to_numpy()
            cis = pd.Series(cis).rolling(10, min_periods=1).mean().to_numpy()
            color = ALGO_COLORS[alg]
            plt.plot(ts, means, color=color, linewidth=2, label=ALGO_SHORT[alg])
            plt.fill_between(ts, means - cis, means + cis, color=color, alpha=0.15)
        plt.xlabel("Time Slot")
        plt.ylabel(ylabel)
        plt.title(f"Convergence at High Load ({high_load})")
        plt.grid(True, alpha=0.3)
        plt.legend(ncols=2, fontsize=9)
        _add_caption("Takeaway: faster stabilization with tighter uncertainty indicates stronger online adaptation.")
        plt.tight_layout()
        plt.savefig(out_dir / f"convergence_{metric}_high_load.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()


def plot_urlcc_tail(df: pd.DataFrame, out_dir: Path, high_load: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df["load_scale"] == high_load]

    plt.figure(figsize=(8, 4.8))
    for alg in ALGORITHM_ORDER:
        vals = np.sort(sub[sub["algorithm"] == alg]["urlcc_delay"].to_numpy())
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        plt.plot(vals, y, linewidth=2, color=ALGO_COLORS[alg], label=ALGO_SHORT[alg])
    plt.xlabel("URLLC Delay (s)")
    plt.ylabel("CDF")
    plt.title(f"URLLC Delay CDF at High Load ({high_load})")
    plt.grid(True, alpha=0.3)
    plt.legend(ncols=2, fontsize=9)
    _add_caption("Takeaway: left-shifted CDF indicates more consistent low-latency behavior.")
    plt.tight_layout()
    plt.savefig(out_dir / "urlcc_delay_cdf_high_load.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4.8))
    for alg in ALGORITHM_ORDER:
        vals = np.sort(sub[sub["algorithm"] == alg]["urlcc_delay"].to_numpy())
        if len(vals) == 0:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ccdf = np.maximum(1.0 - cdf, 1e-6)
        plt.semilogy(vals, ccdf, linewidth=2, color=ALGO_COLORS[alg], label=ALGO_SHORT[alg])
    plt.xlabel("URLLC Delay (s)")
    plt.ylabel("CCDF (log scale)")
    plt.title(f"URLLC Delay CCDF at High Load ({high_load})")
    plt.grid(True, alpha=0.3)
    plt.legend(ncols=2, fontsize=9)
    _add_caption("Takeaway: lower tail probabilities at strict delay targets imply stronger URLLC reliability.")
    plt.tight_layout()
    plt.savefig(out_dir / "urlcc_delay_ccdf_high_load.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def _grouped_distribution_plot(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path, mode: str) -> None:
    per_seed = df.groupby(["algorithm", "seed", "load_scale"], as_index=False)[metric].mean()
    loads = sorted(per_seed["load_scale"].unique())
    fig, axes = plt.subplots(1, len(loads), figsize=(4.2 * len(loads), 4.6), sharey=True)
    if len(loads) == 1:
        axes = [axes]
    for idx, load in enumerate(loads):
        ax = axes[idx]
        vals = []
        labels = []
        for alg in ALGORITHM_ORDER:
            arr = per_seed[(per_seed["algorithm"] == alg) & (per_seed["load_scale"] == load)][metric].to_numpy()
            vals.append(arr)
            labels.append(ALGO_SHORT[alg])
        if mode == "box":
            # `labels=` was deprecated in matplotlib 3.9 and removed in 3.11.
            bp = ax.boxplot(vals, patch_artist=True, tick_labels=labels, showfliers=False)
            for patch, alg in zip(bp["boxes"], ALGORITHM_ORDER):
                patch.set_facecolor(ALGO_COLORS[alg])
                patch.set_alpha(0.45)
        else:
            vp = ax.violinplot(vals, showmeans=True, showmedians=False)
            for i, body in enumerate(vp["bodies"]):
                body.set_facecolor(ALGO_COLORS[ALGORITHM_ORDER[i]])
                body.set_alpha(0.35)
                body.set_edgecolor(ALGO_COLORS[ALGORITHM_ORDER[i]])
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(f"Load={load}")
        ax.grid(True, alpha=0.25)
        if mode == "box":
            ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    fig.supylabel(ylabel)
    title = "Boxplot" if mode == "box" else "Violin Plot"
    plt.suptitle(f"{title} Across Seeds by Load")
    _add_caption("Takeaway: tighter spread across seeds indicates robust algorithmic behavior.")
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("utility_mean", "Utility"),
        ("qos_success", "QoS Success Ratio"),
        ("delay_mean", "Delay (s)"),
    ]
    for metric, ylabel in metrics:
        _grouped_distribution_plot(df, metric, ylabel, out_dir / f"{metric}_boxplot_by_load.png", mode="box")
        _grouped_distribution_plot(df, metric, ylabel, out_dir / f"{metric}_violin_by_load.png", mode="violin")


def plot_pareto(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_df = df.groupby(["algorithm", "load_scale"], as_index=False).mean(numeric_only=True)
    load_markers = {ld: mk for ld, mk in zip(sorted(mean_df["load_scale"].unique()), ["o", "s", "^", "D", "v", "P", "X"])}

    plt.figure(figsize=(7, 5.2))
    for _, row in mean_df.iterrows():
        alg = row["algorithm"]
        load = row["load_scale"]
        plt.scatter(
            row["utility_mean"],
            row["qos_success"],
            s=90,
            color=ALGO_COLORS[alg],
            marker=load_markers[load],
            alpha=0.85,
        )
        plt.text(row["utility_mean"], row["qos_success"], f"{ALGO_SHORT[alg]}@{load}", fontsize=7)
    plt.xlabel("Utility")
    plt.ylabel("QoS Success Ratio")
    plt.title("Pareto: Utility vs QoS Success")
    plt.grid(True, alpha=0.3)
    _add_caption("Takeaway: points farther up-right dominate the utility-reliability trade-off.")
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_utility_vs_qos.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5.2))
    for _, row in mean_df.iterrows():
        alg = row["algorithm"]
        load = row["load_scale"]
        plt.scatter(
            row["utility_mean"],
            row["delay_mean"],
            s=90,
            color=ALGO_COLORS[alg],
            marker=load_markers[load],
            alpha=0.85,
        )
        plt.text(row["utility_mean"], row["delay_mean"], f"{ALGO_SHORT[alg]}@{load}", fontsize=7)
    plt.xlabel("Utility")
    plt.ylabel("Mean Delay (s)")
    plt.title("Pareto: Utility vs Delay")
    plt.grid(True, alpha=0.3)
    _add_caption("Takeaway: high utility with low delay marks stronger operating points.")
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_utility_vs_delay.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def plot_prices(df: pd.DataFrame, out_dir: Path, high_load: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df["load_scale"] == high_load]
    price_metrics = [
        ("price_radio_mean", "Radio Price"),
        ("price_compute_mean", "Compute Price"),
        ("price_transport", "Transport Price"),
        ("price_total_mean", "Mean Price"),
    ]
    for metric, ylabel in price_metrics:
        plt.figure(figsize=(8, 4.8))
        any_line = False
        for alg in ALGORITHM_ORDER:
            grp = sub[sub["algorithm"] == alg]
            if grp[metric].notna().sum() == 0:
                continue
            per_seed_t = grp.groupby(["seed", "t"], as_index=False)[metric].mean()
            ts = sorted(per_seed_t["t"].unique())
            means = []
            cis = []
            for t in ts:
                vals = per_seed_t[per_seed_t["t"] == t][metric].to_numpy()
                m, ci = ci95(vals)
                means.append(m)
                cis.append(ci)
            means = pd.Series(means).rolling(10, min_periods=1).mean().to_numpy()
            cis = pd.Series(cis).rolling(10, min_periods=1).mean().to_numpy()
            color = ALGO_COLORS[alg]
            plt.plot(ts, means, color=color, linewidth=2, label=ALGO_SHORT[alg])
            plt.fill_between(ts, means - cis, means + cis, color=color, alpha=0.15)
            any_line = True
        if not any_line:
            plt.close()
            continue
        plt.xlabel("Time Slot")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Trajectory at High Load ({high_load})")
        plt.grid(True, alpha=0.3)
        plt.legend(ncols=2, fontsize=9)
        _add_caption("Takeaway: smoother bounded prices indicate stable decentralized negotiation.")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_trajectory_high_load.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()


def plot_admm_diagnostics(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    admm = df[df["algorithm"] == "C_ADMM"].copy()
    if admm.empty:
        return

    for metric, title, filename in [
        ("cadmm_r_pri", "C-ADMM Primal Residual", "cadmm_primal_residual_vs_t.png"),
        ("cadmm_r_dual", "C-ADMM Dual Residual", "cadmm_dual_residual_vs_t.png"),
    ]:
        sub = admm[admm[metric].notna()]
        if sub.empty:
            continue
        per_seed_t = sub.groupby(["seed", "t"], as_index=False)[metric].mean()
        ts = sorted(per_seed_t["t"].unique())
        means = []
        cis = []
        for t in ts:
            vals = per_seed_t[per_seed_t["t"] == t][metric].to_numpy()
            m, ci = ci95(vals)
            means.append(m)
            cis.append(ci)
        means = np.maximum(np.array(means), 1e-9)
        cis = np.maximum(np.array(cis), 1e-9)
        plt.figure(figsize=(8, 4.8))
        plt.semilogy(ts, means, color=ALGO_COLORS["C_ADMM"], linewidth=2, label="Mean")
        plt.fill_between(ts, np.maximum(means - cis, 1e-9), means + cis, color=ALGO_COLORS["C_ADMM"], alpha=0.2)
        plt.xlabel("Time Slot")
        plt.ylabel(metric)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        _add_caption("Takeaway: downward residual trends indicate stronger ADMM consensus convergence.")
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

    rounds = admm[admm["cadmm_rounds"].notna()]
    if not rounds.empty:
        plt.figure(figsize=(7, 5.2))
        sc = plt.scatter(rounds["cadmm_rounds"], rounds["utility_mean"], c=rounds["load_scale"], cmap="viridis", alpha=0.35, s=18)
        plt.colorbar(sc, label="Load Scale")
        plt.xlabel("C-ADMM Rounds Used")
        plt.ylabel("Utility")
        plt.title("C-ADMM Rounds vs Utility")
        plt.grid(True, alpha=0.3)
        _add_caption("Takeaway: this view quantifies the communication-performance trade-off of extra ADMM rounds.")
        plt.tight_layout()
        plt.savefig(out_dir / "cadmm_rounds_vs_utility.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

        mean_rounds = rounds.groupby(["load_scale"], as_index=False)["cadmm_rounds"].mean()
        plt.figure(figsize=(7, 4.8))
        plt.plot(mean_rounds["load_scale"], mean_rounds["cadmm_rounds"], marker="o", linewidth=2, color=ALGO_COLORS["C_ADMM"])
        plt.xlabel("Load Scale")
        plt.ylabel("Average Rounds")
        plt.title("C-ADMM Average Rounds vs Load")
        plt.grid(True, alpha=0.3)
        _add_caption("Takeaway: higher rounds at heavier loads reflect increased consensus difficulty.")
        plt.tight_layout()
        plt.savefig(out_dir / "cadmm_avg_rounds_vs_load.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()


def plot_runtime(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_load_metric_with_ci(
        df=df,
        metric="algo_runtime_ms",
        ylabel="Algorithm Runtime per Slot (ms)",
        caption="Takeaway: lower runtime curves indicate better online deployability.",
        out_path=out_dir / "runtime_algo_ms_vs_load.png",
    )

    runtime_mean = df.groupby("algorithm", as_index=False)["algo_runtime_ms"].mean()
    plt.figure(figsize=(7.2, 4.8))
    xs = np.arange(len(runtime_mean))
    colors = [ALGO_COLORS[a] for a in runtime_mean["algorithm"]]
    plt.bar(xs, runtime_mean["algo_runtime_ms"], color=colors, alpha=0.85)
    plt.xticks(xs, [ALGO_SHORT[a] for a in runtime_mean["algorithm"]], rotation=20, ha="right")
    plt.ylabel("Mean Algorithm Runtime (ms)")
    plt.title("Overall Runtime Comparison")
    plt.grid(True, axis="y", alpha=0.3)
    _add_caption("Takeaway: this summarizes raw compute cost independent of reward quality.")
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_overall_bar.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def _plot_sig_heatmap(sig_df: pd.DataFrame, p_col: str, title: str, out_path: Path) -> None:
    if sig_df.empty or p_col not in sig_df.columns:
        return
    algs = [a for a in ALGORITHM_ORDER if a != "MAAN_PPO"]
    loads = sorted(sig_df["load_scale"].unique())
    mat = np.full((len(algs), len(loads)), np.nan, dtype=float)
    for i, alg in enumerate(algs):
        for j, ld in enumerate(loads):
            sub = sig_df[(sig_df["algorithm"] == alg) & (sig_df["load_scale"] == ld)]
            if not sub.empty:
                mat[i, j] = float(sub.iloc[0][p_col])
    val = np.log10(np.clip(mat, 1e-12, 1.0))
    plt.figure(figsize=(1.4 * len(loads) + 3.2, 0.9 * len(algs) + 2.4))
    im = plt.imshow(val, aspect="auto", cmap="YlOrRd_r", vmin=-6, vmax=0)
    plt.colorbar(im, label="log10(p-value)")
    plt.xticks(np.arange(len(loads)), [str(ld) for ld in loads])
    plt.yticks(np.arange(len(algs)), [ALGO_SHORT[a] for a in algs])
    plt.xlabel("Load Scale")
    plt.title(title)
    for i in range(len(algs)):
        for j in range(len(loads)):
            if np.isnan(mat[i, j]):
                txt = "NA"
            else:
                txt = f"{mat[i, j]:.2e}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=7)
    _add_caption("Takeaway: darker cells indicate stronger statistical separation from MAAN.")
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def plot_significance_heatmaps(sig_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if sig_df.empty:
        return
    # Plot the Holm-adjusted p-values -- the raw ones overstate significance.
    _plot_sig_heatmap(
        sig_df,
        "utility_mean_padj_vs_MAAN_PPO",
        "Significance Heatmap: Utility (vs MAAN, Holm-adjusted)",
        out_dir / "significance_heatmap_utility.png",
    )
    _plot_sig_heatmap(
        sig_df,
        "qos_success_padj_vs_MAAN_PPO",
        "Significance Heatmap: QoS Success (vs MAAN, Holm-adjusted)",
        out_dir / "significance_heatmap_qos_success.png",
    )
    _plot_sig_heatmap(
        sig_df,
        "delay_mean_padj_vs_MAAN_PPO",
        "Significance Heatmap: Delay (vs MAAN, Holm-adjusted)",
        out_dir / "significance_heatmap_delay.png",
    )


def plot_publication_pack(df: pd.DataFrame, out_dir: Path, sig_df: pd.DataFrame) -> None:
    pub_dir = out_dir / "plots_publication"
    pub_dir.mkdir(parents=True, exist_ok=True)
    high_load = float(np.max(df["load_scale"]))

    stages = (
        lambda: plot_convergence_with_ci(df, pub_dir, high_load=high_load),
        lambda: plot_urlcc_tail(df, pub_dir, high_load=high_load),
        lambda: plot_distributions(df, pub_dir),
        lambda: plot_pareto(df, pub_dir),
        lambda: plot_prices(df, pub_dir, high_load=high_load),
        lambda: plot_admm_diagnostics(df, pub_dir),
        lambda: plot_runtime(df, pub_dir),
        lambda: plot_significance_heatmaps(sig_df, pub_dir),
    )
    for stage in stages:
        stage()
        # Reclaim between stages rather than at the end. This whole pack renders inside a
        # 512 MB instance, and matplotlib holds font caches and figure buffers that only
        # a close-all plus a collection actually releases.
        plt.close("all")
        gc.collect()


if __name__ == "__main__":
    cfg = ExpConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_experiment(cfg)
    result.to_csv(out_dir / "benchmark_results_phase2.csv", index=False)
    _, sig_df = save_tables(result, out_dir)
    plot_all(result, out_dir / "plots")
    plot_publication_pack(result, out_dir, sig_df)
    with open(out_dir / "config_used.json", "w", encoding="utf-8") as fp:
        import json

        json.dump(asdict(cfg), fp, indent=2)
    print(f"Saved phase2 results to: {out_dir.resolve()}")
