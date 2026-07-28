# Agentic 5G Network Slicing - Implementation Report

## Current Alignment Status

This backend now implements a benchmark framework aligned with the two PDFs around:

1. Distributed multi-resource slicing (radio/compute/transport + MEC association)
2. Five algorithm tracks (`MAAN_PPO`, `Independent_MAPPO_PPO`, `C_ADMM`, `Static_Greedy`, `OGD_Bandit`)
3. Shared environment constraints and post-enforcement metric evaluation
4. URLLC chance-constraint evaluation via SAA
5. 12-14 cross-algorithm plots under load scaling

## Key Technical Updates

- Added `scipy` to dependencies for statistical significance tests.
- Added exogenous trace support in `FiveGEnvironment` so all algorithms can be evaluated on common random streams.
- Separated rollout RNG and SAA RNG in environment to avoid evaluation leakage into scenario dynamics.
- Updated PRB residual rounding to prefer higher marginal utility slices.
- Extended state with `prev_rates` and `prev_delays` for richer local observations.
- Refactored benchmark drivers to run each algorithm on identical traces per seed/load.
- Upgraded `Static_Greedy` to include delay-aware MEC association, stability clipping, and greedy QoS repair loops.
- Upgraded `OGD_Bandit` (formerly `OMD_BF`) to include perturbation-based bandit gradient estimation, clipped scalar reward, and projected price updates.
- Upgraded PPO variants with price-aware / price-free reward separation, local penalties, and damped dual-price updates.
- Upgraded `C_ADMM` with residual-based stopping and adaptive-`rho` balancing.

## Delay-Model Correction

The delay model was rebuilt because the previous one was not dimensionally consistent, and
the inconsistency was large enough to void every published comparison:

- `d_trans` was `lambda / tau` — a dimensionless utilisation ratio, not a time. Against a
  budget of 8–50 ms it evaluated to 10^4–10^9 s and dominated everything else.
- `d_radio` was `1 / (R - lambda)` with both terms in bit/s, giving s/bit, and its floor
  pinned it at exactly `t_tti` regardless of the PRB allocation — so radio decisions had
  no effect on delay at all.
- `d_comp` used an `omega` with no declared unit, and never fell below ~0.3 s.

The consequence was that `qos_success` was identically **0.000** and
`urlcc_violation_prob_saa` identically **1.000** for all five algorithms at every load,
while `radio_util`, `transport_util` and `d_radio` were constants. The significance tests
were computing p-values over quantities carrying no signal.

All three domains are now M/M/1 queues with a declared unit for every symbol (see the
module docstring in `src/environment/fiveg_env.py`), so the mean sojourn time
`W = L / (mu - lambda)` is genuinely a time. Related corrections:

- `_round_prbs` now caps at capacity instead of force-filling to it, so an allocator that
  requests fewer PRBs keeps them idle and `radio_util` becomes an informative metric.
- `C_ADMM` normalised the capacity vector rather than the demand vector, requesting ~1 PRB
  out of 160; its primal step also carried no objective gradient. It now uses closed-form
  derivatives of the scored utility.
- Jain's index is computed on rates, not on signed utilities.
- Confidence intervals use Student's *t*; comparisons use paired *t*-tests with
  Holm-Bonferroni correction, matching the common-random-numbers design.
- `load_scale` now scales offered traffic (previously it scaled only compute demand,
  despite every plot labelling its x-axis "Load Scale").

## Project Structure

- `src/environment/fiveg_env.py`: environment dynamics, enforcement, metrics, SAA.
- `src/algorithms/*.py`: all five algorithm implementations.
- `src/experiments/run_benchmark.py`: phase-1 benchmark + 14 plots.
- `src/experiments/run_benchmark_phase2.py`: phase-2 benchmark + CI/significance + 14 plots.

## How To Run

```bash
pip install -r requirements.txt
python -m src.experiments.run_benchmark
python -m src.experiments.run_benchmark_phase2
```

Outputs include:

- `outputs/benchmark_results.csv`
- `outputs/plots/*.png`
- `outputs_phase2/benchmark_results_phase2.csv`
- `outputs_phase2/summary_with_ci95.csv`
- `outputs_phase2/statistical_significance.csv`
- `outputs_phase2/plots/*.png`
