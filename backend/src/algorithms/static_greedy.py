from __future__ import annotations

from typing import Dict

import numpy as np

from src.algorithms.base import AlgorithmOutput, BaseAllocator


class StaticGreedyAllocator(BaseAllocator):
    def __init__(
        self,
        slice_weights: np.ndarray,
        s: int,
        k: int,
        m: int,
        b_k: np.ndarray,
        c_m: np.ndarray,
        t_agg: float,
        r_min: np.ndarray | None = None,
        d_max: np.ndarray | None = None,
        omega: np.ndarray | None = None,
        j_max: int = 16,
    ):
        super().__init__("Static_Greedy")
        self.slice_weights = slice_weights / np.sum(slice_weights)
        self.s = s
        self.k = k
        self.m = m
        self.b_k = b_k.astype(float)
        self.c_m = c_m.astype(float)
        self.t_agg = float(t_agg)
        self.r_min = np.full(s, 10e6) if r_min is None else r_min.astype(float)
        self.d_max = np.full(s, 0.02) if d_max is None else d_max.astype(float)
        self.omega = np.full(s, 60.0) if omega is None else omega.astype(float)
        self.priority = np.argsort(self.d_max)  # URLLC-like slices first.
        self.j_max = j_max
        self.b_step = 1.0
        self.c_step_max = 30.0
        self.t_step_max = 15.0
        self.safe = 1e-6
        self.w_prb = 180e3
        self.n0 = 1e-9
        self.t_tti = 1e-3
        self.pk = np.ones(self.k, dtype=float)
        self.prev_b = np.zeros((self.s, self.k), dtype=float)
        self.prev_c = np.zeros((self.s, self.m), dtype=float)
        self.prev_tau = np.zeros(self.s, dtype=float)
        self.has_prev = False

    def reset(self) -> None:
        self.prev_b.fill(0.0)
        self.prev_c.fill(0.0)
        self.prev_tau.fill(0.0)
        self.has_prev = False

    def act(self, state: Dict[str, np.ndarray]) -> AlgorithmOutput:
        lam = state["lambda"].astype(float)
        channel = state["channel"].astype(float)

        b = self._proportional_prb_allocation()
        x, c = self._delay_aware_mec_association(lam, channel)
        tau = self.slice_weights * self.t_agg

        if self.has_prev:
            b = np.clip(b, self.prev_b - 8.0, self.prev_b + 8.0)
            c = np.clip(c, self.prev_c - 40.0, self.prev_c + 40.0)
            tau = np.clip(tau, self.prev_tau - 20.0, self.prev_tau + 20.0)

        b = self._project_prb_capacity(b)
        c = self._project_domain_capacity(c, self.c_m)
        tau = self._project_transport_capacity(tau)

        for _ in range(self.j_max):
            rates, delays, d_radio, d_trans, d_comp, sinr = self._predict_metrics(lam, channel, b, c, tau)
            viol = np.where((rates < self.r_min) | (delays > self.d_max))[0]
            if viol.size == 0:
                break

            s_target = self._highest_priority_violator(viol)
            bottleneck = np.argmax([d_radio[s_target], d_comp[s_target], d_trans[s_target]])
            changed = False
            if bottleneck == 0:
                changed = self._repair_radio(s_target, b, sinr)
            elif bottleneck == 1:
                changed = self._repair_compute(s_target, c, x)
            else:
                changed = self._repair_transport(s_target, tau)
            if not changed:
                break

        b = self._project_prb_capacity(b)
        c = self._project_domain_capacity(c, self.c_m)
        tau = self._project_transport_capacity(tau)

        self.prev_b = b.copy()
        self.prev_c = c.copy()
        self.prev_tau = tau.copy()
        self.has_prev = True

        return AlgorithmOutput(actions={"b": b, "c": c, "tau": tau, "x": x}, aux={})

    def _proportional_prb_allocation(self) -> np.ndarray:
        b = np.zeros((self.s, self.k), dtype=float)
        for k_idx in range(self.k):
            cap = int(round(self.b_k[k_idx]))
            raw = self.slice_weights * cap
            alloc = np.floor(raw).astype(int)
            rem = cap - int(np.sum(alloc))
            if rem > 0:
                order = np.argsort(-self.slice_weights)
                for s_idx in order[:rem]:
                    alloc[s_idx] += 1
            b[:, k_idx] = alloc.astype(float)
        return b

    def _delay_aware_mec_association(self, lam: np.ndarray, channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.zeros((self.s, self.m), dtype=int)
        c = np.zeros((self.s, self.m), dtype=float)
        c_avail = self.c_m.copy()
        c_budget = self.slice_weights * np.sum(self.c_m)
        d_prop = np.zeros(self.m, dtype=float)
        for m_idx in range(self.m):
            ch_proxy = np.mean(channel[:, m_idx % self.k])
            d_prop[m_idx] = 1.0 / max(ch_proxy, self.safe)

        for s_idx in self.priority:
            score = d_prop + lam[s_idx] / np.maximum(c_avail, self.safe)
            m_idx = int(np.argmin(score))
            x[s_idx, m_idx] = 1
            take = min(c_budget[s_idx], c_avail[m_idx])
            c[s_idx, m_idx] = max(take, 0.0)
            c_avail[m_idx] = max(c_avail[m_idx] - take, 0.0)
        return x, c

    def _highest_priority_violator(self, viol: np.ndarray) -> int:
        rank = {int(s_idx): r for r, s_idx in enumerate(self.priority)}
        return int(sorted(list(map(int, viol)), key=lambda x: rank.get(x, 10_000))[0])

    def _repair_radio(self, s_target: int, b: np.ndarray, sinr: np.ndarray) -> bool:
        # To reduce radio delay we should add PRBs on the strongest link, not the weakest.
        k_idx = int(np.argmax(sinr[s_target]))
        donors = [r for r in range(self.s) if r != s_target and b[r, k_idx] >= 1.0]
        if not donors:
            return False
        donor = int(max(donors, key=lambda r: b[r, k_idx]))
        b[s_target, k_idx] += self.b_step
        b[donor, k_idx] -= self.b_step
        return True

    def _repair_compute(self, s_target: int, c: np.ndarray, x: np.ndarray) -> bool:
        m_idx = int(np.argmax(x[s_target]))
        donors = [r for r in range(self.s) if r != s_target and x[r, m_idx] == 1 and c[r, m_idx] > 1.0]
        if not donors:
            return False
        donor = int(max(donors, key=lambda r: c[r, m_idx] / max(self.d_max[r], self.safe)))
        delta = min(0.1 * c[donor, m_idx], self.c_step_max)
        if delta <= 0:
            return False
        c[s_target, m_idx] += delta
        c[donor, m_idx] -= delta
        return True

    def _repair_transport(self, s_target: int, tau: np.ndarray) -> bool:
        donors = [r for r in range(self.s) if r != s_target and tau[r] > 1.0]
        if not donors:
            return False
        donor = int(max(donors, key=lambda r: tau[r] / max(self.d_max[r], self.safe)))
        delta = min(0.1 * tau[donor], self.t_step_max)
        if delta <= 0:
            return False
        tau[s_target] += delta
        tau[donor] -= delta
        return True

    def _predict_metrics(
        self,
        lam: np.ndarray,
        channel: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        tau: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sinr = self._compute_sinr(channel)
        rates = np.sum(b * self.w_prb * np.log2(1.0 + sinr), axis=1)
        d_radio = 1.0 / np.maximum(rates - lam, 1e3) + self.t_tti
        c_slice = np.sum(c, axis=1)
        d_comp = self.omega / np.maximum(c_slice, self.safe)
        d_trans = lam / np.maximum(tau, self.safe)
        delays = d_radio + d_comp + d_trans
        return rates, delays, d_radio, d_trans, d_comp, sinr

    def _compute_sinr(self, gains: np.ndarray) -> np.ndarray:
        inter = np.sum(gains * self.pk[None, :], axis=1, keepdims=True) - gains * self.pk[None, :]
        return (self.pk[None, :] * gains) / np.maximum(inter + self.n0 * self.w_prb, 1e-12)

    def _project_prb_capacity(self, b: np.ndarray) -> np.ndarray:
        b_clip = np.maximum(0.0, b.copy())
        for k_idx in range(self.k):
            cap = int(round(self.b_k[k_idx]))
            col = np.floor(b_clip[:, k_idx]).astype(int)
            col_sum = int(np.sum(col))
            rem = cap - col_sum
            if rem > 0:
                order = np.argsort(-self.slice_weights)
                for s_idx in order[:rem]:
                    col[s_idx] += 1
            elif rem < 0:
                over = -rem
                order = np.argsort(-col)
                for s_idx in order:
                    if over <= 0:
                        break
                    take = min(over, col[s_idx])
                    col[s_idx] -= take
                    over -= take
            b_clip[:, k_idx] = col.astype(float)
        return b_clip

    @staticmethod
    def _project_domain_capacity(mat: np.ndarray, caps: np.ndarray) -> np.ndarray:
        out = np.maximum(0.0, mat.copy())
        for j in range(out.shape[1]):
            col_sum = np.sum(out[:, j])
            if col_sum > caps[j]:
                out[:, j] *= caps[j] / max(col_sum, 1e-9)
        return out

    def _project_transport_capacity(self, tau: np.ndarray) -> np.ndarray:
        out = np.maximum(0.0, tau.copy())
        total = float(np.sum(out))
        if total > self.t_agg:
            out *= self.t_agg / max(total, 1e-9)
        return out
