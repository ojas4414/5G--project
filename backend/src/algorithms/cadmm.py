"""Consensus ADMM allocator.

The slicing problem solved here is

    maximise   sum_s  U_s(b_s, c_s, tau_s)
    s.t.       sum_s b_{s,k} <= B_k    for every gNB k
               sum_s c_{s,m} <= C_m    for every MEC m
               sum_s tau_s   <= T_agg

with the same per-slice utility the environment scores (concave log-rate reward minus
delay cost). ADMM splits this into a per-slice primal step -- which sees only local
information plus the consensus variable -- and a projection onto the coupling
constraints, exchanged through scaled dual variables.

The primal step performs a few projected-gradient iterations on the true local objective
using closed-form derivatives of the M/M/1 delay model, so the algorithm optimises the
objective it is scored against.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.algorithms.base import AlgorithmOutput, BaseAllocator


class CADMMAllocator(BaseAllocator):
    def __init__(
        self,
        s: int,
        k: int,
        m: int,
        b_k: np.ndarray,
        c_m: np.ndarray,
        t_agg: float,
        r_min: np.ndarray,
        d_max: np.ndarray,
        omega: np.ndarray,
        alpha: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        w_prb: float = 180e3,
        n0: float = 1e-9,
        l_pkt: float = 12000.0,
        rounds: int = 12,
        rho: float = 0.8,
        tol_pri: float = 1e-3,
        tol_dual: float = 1e-3,
        adaptive_rho: bool = True,
        inner_steps: int = 3,
    ):
        super().__init__("C_ADMM")
        self.s = s
        self.k = k
        self.m = m
        self.b_k = np.asarray(b_k, dtype=float)
        self.c_m = np.asarray(c_m, dtype=float)
        self.t_agg = float(t_agg)
        self.r_min = np.asarray(r_min, dtype=float)
        self.d_max = np.asarray(d_max, dtype=float)
        self.omega = np.asarray(omega, dtype=float)
        self.alpha = np.ones(s) if alpha is None else np.asarray(alpha, dtype=float)
        self.beta = np.ones(s) if beta is None else np.asarray(beta, dtype=float)
        self.w_prb = float(w_prb)
        self.n0 = float(n0)
        self.l_pkt = float(l_pkt)
        self.rounds = rounds
        self.rho = rho
        self.tol_pri = tol_pri
        self.tol_dual = tol_dual
        self.adaptive_rho = adaptive_rho
        self.inner_steps = inner_steps
        self.pk = np.ones(self.k, dtype=float)

    # ------------------------------------------------------------- objective

    def _marginal_rates(self, channel: np.ndarray) -> np.ndarray:
        """Bit/s delivered by one PRB of gNB k to slice s."""
        signal = channel * self.pk
        inter = np.sum(signal, axis=-1, keepdims=True) - signal
        sinr = signal / np.maximum(inter + self.n0 * self.w_prb, 1e-12)
        return self.w_prb * np.log2(1.0 + sinr)

    def _grad_b(self, b: np.ndarray, marginal: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """d/db of  alpha*log1p(R/r_min) - beta*d_radio/d_max,  with R = sum_k b*marginal.

        d(log1p(R/r_min))/dR = 1/(r_min + R);  d_radio = L/(R - lambda) so
        d(d_radio)/dR = -L/(R - lambda)^2, i.e. more rate always shortens the queue.
        """
        rates = np.sum(b * marginal, axis=1)
        d_rate = self.alpha / np.maximum(self.r_min + rates, 1e-9)
        headroom = np.maximum(rates - lam, self.l_pkt)
        d_delay = self.beta / np.maximum(self.d_max, 1e-9) * self.l_pkt / headroom**2
        return (d_rate + d_delay)[:, None] * marginal

    def _grad_c(self, c: np.ndarray, lam: np.ndarray, assign: np.ndarray) -> np.ndarray:
        """d/dc of  -beta*d_comp/d_max  where d_comp = L/(c/omega - lambda)."""
        c_slice = np.sum(c, axis=1)
        service = c_slice / np.maximum(self.omega, 1e-9)
        headroom = np.maximum(service - lam, self.l_pkt)
        g = self.beta / np.maximum(self.d_max, 1e-9) * self.l_pkt / (headroom**2 * np.maximum(self.omega, 1e-9))
        return g[:, None] * assign

    def _grad_tau(self, tau: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """d/dtau of  -beta*d_trans/d_max  where d_trans = L/(tau - lambda)."""
        headroom = np.maximum(tau - lam, self.l_pkt)
        return self.beta / np.maximum(self.d_max, 1e-9) * self.l_pkt / headroom**2

    # ------------------------------------------------------------------ ADMM

    def act(self, state: Dict[str, np.ndarray]) -> AlgorithmOutput:
        lam = np.asarray(state["lambda"], dtype=float)
        channel = np.asarray(state["channel"], dtype=float)
        marginal = self._marginal_rates(channel)

        # Warm start: split each resource in proportion to offered traffic. Note this
        # normalises the *demand* vector and scales it by capacity -- the reverse would
        # hand each slice a fraction of a PRB.
        share = lam / max(float(np.sum(lam)), 1e-9)
        b = np.outer(share, self.b_k)
        tau = share * self.t_agg

        # Associate each slice with the MEC where it has the best proxy channel, then bid
        # for the whole host: association is exclusive, so asking for only a traffic-share
        # of a host the slice may have to itself would idle compute for no reason.
        x = np.zeros((self.s, self.m), dtype=int)
        for s_idx in range(self.s):
            x[s_idx, int(np.argmax(channel[s_idx]) % self.m)] = 1
        c = x * self.c_m[None, :]

        z_b, z_c, z_t = b.copy(), c.copy(), tau.copy()
        l_b = np.zeros_like(b)
        l_c = np.zeros_like(c)
        l_t = np.zeros_like(tau)

        # Gradient step sizes, scaled to each resource's capacity so the update is
        # dimensionless in the resource units.
        step_b = float(np.mean(self.b_k)) ** 2
        step_c = float(np.mean(self.c_m)) ** 2
        step_t = self.t_agg**2

        rho = float(self.rho)
        r_pri = r_dual = 0.0
        rounds_used = 0

        for r in range(self.rounds):
            rounds_used = r + 1

            # --- Step 1: local primal updates (maximise U_s - rho/2 ||v - z + l/rho||^2).
            for _ in range(self.inner_steps):
                b = np.maximum(0.0, b + step_b * (self._grad_b(b, marginal, lam) / max(rho, 1e-9)) - (b - z_b + l_b / max(rho, 1e-9)))
                c = np.maximum(0.0, c + step_c * (self._grad_c(c, lam, x) / max(rho, 1e-9)) - (c - z_c + l_c / max(rho, 1e-9)))
                tau = np.maximum(0.0, tau + step_t * (self._grad_tau(tau, lam) / max(rho, 1e-9)) - (tau - z_t + l_t / max(rho, 1e-9)))
            c = c * x

            # --- Step 2: consensus projection onto the coupling constraints.
            z_b_prev, z_c_prev, z_t_prev = z_b.copy(), z_c.copy(), z_t.copy()
            z_b = self._project_capacity(b + l_b / max(rho, 1e-9), self.b_k)
            z_c = self._project_capacity(c + l_c / max(rho, 1e-9), self.c_m) * x
            z_t = self._project_scalar(tau + l_t / max(rho, 1e-9), self.t_agg)

            # --- Step 3: dual updates (unscaled form).
            l_b = l_b + rho * (b - z_b)
            l_c = l_c + rho * (c - z_c)
            l_t = l_t + rho * (tau - z_t)

            # --- Step 4: residuals, normalised by capacity so the tolerances are relative.
            r_pri = float(
                np.sqrt(
                    np.sum(((b - z_b) / np.mean(self.b_k)) ** 2)
                    + np.sum(((c - z_c) / np.mean(self.c_m)) ** 2)
                    + np.sum(((tau - z_t) / self.t_agg) ** 2)
                )
            )
            r_dual = float(
                rho
                * np.sqrt(
                    np.sum(((z_b - z_b_prev) / np.mean(self.b_k)) ** 2)
                    + np.sum(((z_c - z_c_prev) / np.mean(self.c_m)) ** 2)
                    + np.sum(((z_t - z_t_prev) / self.t_agg) ** 2)
                )
            )
            if r_pri < self.tol_pri and r_dual < self.tol_dual:
                break

            # --- Step 5: adaptive penalty (residual balancing).
            if self.adaptive_rho and r_pri > 0 and r_dual > 0:
                if r_pri > 10.0 * r_dual:
                    rho *= 2.0
                    l_b /= 2.0
                    l_c /= 2.0
                    l_t /= 2.0
                elif r_dual > 10.0 * r_pri:
                    rho /= 2.0
                    l_b *= 2.0
                    l_c *= 2.0
                    l_t *= 2.0

        self.rho = float(np.clip(rho, 1e-3, 100.0))
        return AlgorithmOutput(
            actions={"b": z_b, "c": z_c, "tau": z_t, "x": x},
            aux={"r_pri": r_pri, "r_dual": r_dual, "rounds": float(rounds_used), "rho": float(self.rho)},
        )

    @staticmethod
    def _project_capacity(mat: np.ndarray, caps: np.ndarray) -> np.ndarray:
        out = np.maximum(0.0, mat.copy())
        for j in range(out.shape[1]):
            col_sum = np.sum(out[:, j])
            if col_sum > caps[j]:
                out[:, j] *= caps[j] / max(col_sum, 1e-9)
        return out

    @staticmethod
    def _project_scalar(vec: np.ndarray, cap: float) -> np.ndarray:
        out = np.maximum(0.0, vec.copy())
        total = np.sum(out)
        if total > cap:
            out *= cap / max(total, 1e-9)
        return out
