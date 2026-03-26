# Agentic 5G Network Slicing - Implementation Report

## What was implemented

This project implements an end-to-end Python benchmark framework for your updated distributed 5G slicing problem and compares 5 algorithms:

1. `MAAN` (price-based multi-agent learning surrogate)
2. `Independent_MAPPO` (ablation without prices)
3. `C_ADMM` (distributed optimization baseline)
4. `Static_Greedy` (proportional + greedy repair-style heuristic baseline)
5. `OMD_BF` (online mirror descent with bandit feedback)

## Mapping to your problem formulation

- Multi-resource allocation per time-slot: radio `b_{s,k}`, compute `c_{s,m}`, transport `tau_s`, association `x_{s,m}`.
- Resource capacity enforcement over shared infrastructure.
- SINR-based rate model and decomposed delay model (radio + transport + compute).
- Utility with normalized rate-delay trade-off and smooth delay-violation penalty.
- Decentralized flavor through local actions and price/feedback-based adaptation.

## Project structure

- `src/environment/fiveg_env.py`: Updated environment + metrics engine.
- `src/algorithms/`: Five algorithm implementations with common API.
- `src/experiments/run_benchmark.py`: End-to-end benchmark driver and plotting.
- `outputs/`: Generated CSV and plot files after execution.

## Generated metrics and plots (14 total)

The benchmark auto-generates 14 comparison plots:

1. Mean utility vs load
2. QoS success ratio vs load
3. Mean delay vs load
4. Mean rate vs load
5. Jain fairness index vs load
6. Radio utilization vs load
7. Compute utilization vs load
8. Transport utilization vs load
9. URLLC delay vs load
10. eMBB rate vs load
11. Radio delay component vs load
12. Transport delay component vs load
13. Compute delay component vs load
14. Convergence trend (smoothed utility over time, highest load)

## How to run (local or Colab)

```bash
pip install -r requirements.txt
python -m src.experiments.run_benchmark
```

Outputs:

- `outputs/benchmark_results.csv`
- `outputs/config_used.json`
- `outputs/plots/*.png` (14 files)

## Notes on methodological accuracy

- This code is an implementation-oriented benchmark scaffold aligned with your pseudocode structure and distributed problem framing.
- The MAAN/Independent-MAPPO modules are lightweight algorithmic surrogates to keep runtime practical in Colab and allow large-scale comparative plotting.
- If you want publication-grade parity with exact PPO/CTDE and full chance-constrained SAA loops, we can now upgrade this scaffold in phase-2 by adding a full RL training pipeline (PyTorch) and repeated Monte Carlo evaluation blocks.

## Recommended next extension (phase-2)

1. Replace MAAN surrogate with full PPO actor-critic + CTDE critic.
2. Add explicit URLLC chance-constraint SAA estimator per slot.
3. Add confidence intervals and statistical significance tests across seeds.
4. Add ablations on communication cost and convergence rounds.

## Phase-2 upgrade started

The codebase now includes a higher-fidelity phase-2 pipeline:

- `src/algorithms/ppo_variants.py`
  - `MAAN_PPO`: PPO-style multi-agent actor + centralized critic with price signal.
  - `Independent_MAPPO_PPO`: same PPO structure, without prices.
- `src/environment/fiveg_env.py`
  - Added `saa_urlcc_violation_probability(...)` to estimate URLLC chance-constraint violations via Monte Carlo.
- `src/experiments/run_benchmark_phase2.py`
  - End-to-end evaluation with SAA metric and CI-ready summary table.
  - Outputs `summary_with_ci95.csv` for reporting mean +- 95% CI.

Run phase-2:

```bash
pip install -r requirements.txt
python -m src.experiments.run_benchmark_phase2
```
