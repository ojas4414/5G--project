from __future__ import annotations

from typing import Dict

import numpy as np

from src.algorithms.base import AlgorithmOutput, BaseAllocator


class OMDBanditAllocator(BaseAllocator):
    """One-point bandit gradient allocator (Flaxman-Kalai-McMahan style).

    Each slot the parameter vector is perturbed along a random direction drawn uniformly
    from the unit sphere, and the single scalar reward that comes back is turned into the
    gradient estimate ``g = (d * r / delta) * v``, which is an unbiased estimate of the
    gradient of the delta-smoothed objective. The iterate is then projected back onto a
    box by clipping.

    Note this is projected online gradient ascent with bandit feedback, using a Euclidean
    projection -- there is no Bregman divergence or mirror map here, so it is *not* mirror
    descent despite the legacy ``OMD`` in the class name. The benchmark reports it as
    ``OGD_Bandit``.
    """

    def __init__(
        self,
        s: int,
        k: int,
        m: int,
        b_k: np.ndarray,
        c_m: np.ndarray,
        t_agg: float,
        d_max: np.ndarray | None = None,
    ):
        super().__init__("OGD_Bandit")
        self.s = s
        self.k = k
        self.m = m
        self.b_k = b_k.astype(float)
        self.c_m = c_m.astype(float)
        self.t_agg = float(t_agg)
        self.d_a = self.k + self.m + 1
        self.theta = np.zeros((self.s, self.d_a), dtype=float)
        self.t = 1
        self.eta0 = 0.25
        self.price_eta = 2e-3
        self.r_max = 5.0
        self.stab_coef = 0.02
        self.qos_coef = 1.0
        self.d_max = np.full(self.s, 0.02) if d_max is None else d_max.astype(float)

        self.mu_b = np.zeros(self.k, dtype=float)
        self.mu_c = np.zeros(self.m, dtype=float)
        self.mu_t = 0.0

        self.last_v = np.zeros((self.s, self.d_a), dtype=float)
        self.last_delta = 1.0
        self.prev_action_vec: np.ndarray | None = None
        self.last_action_vec = np.zeros((self.s, self.d_a), dtype=float)

    def reset(self) -> None:
        self.t = 1
        self.theta.fill(0.0)
        self.mu_b.fill(0.0)
        self.mu_c.fill(0.0)
        self.mu_t = 0.0
        self.last_v.fill(0.0)
        self.last_delta = 1.0
        self.prev_action_vec = None
        self.last_action_vec.fill(0.0)

    def act(self, state: Dict[str, np.ndarray]) -> AlgorithmOutput:
        self.last_delta = self.t ** (-0.25)
        v = np.random.normal(0.0, 1.0, size=(self.s, self.d_a))
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        self.last_v = v

        theta_perturbed = self.theta + self.last_delta * v
        actions, action_vec = self._actions_from_theta(theta_perturbed)
        self.last_action_vec = action_vec
        return AlgorithmOutput(actions=actions, aux={"delta": float(self.last_delta)})

    def observe(self, state: Dict[str, np.ndarray], metrics: Dict[str, np.ndarray]) -> None:
        eta = self.eta0 / np.sqrt(max(self.t, 1))
        utilities = metrics["utilities"].astype(float)
        delays = metrics["delays"].astype(float)
        b_act = metrics.get("b_act", np.zeros((self.s, self.k), dtype=float))
        c_act = metrics.get("c_act", np.zeros((self.s, self.m), dtype=float))
        tau_act = metrics.get("tau_act", np.zeros(self.s, dtype=float))

        for s_idx in range(self.s):
            price_cost = float(np.dot(self.mu_b, b_act[s_idx]) + np.dot(self.mu_c, c_act[s_idx]) + self.mu_t * tau_act[s_idx])
            stability = 0.0
            if self.prev_action_vec is not None:
                stability = float(np.sum(np.abs(self.last_action_vec[s_idx] - self.prev_action_vec[s_idx])))
            qos_pen = max(0.0, (delays[s_idx] - self.d_max[s_idx]) / max(self.d_max[s_idx], 1e-9))
            r_raw = utilities[s_idx] - price_cost - self.stab_coef * stability - self.qos_coef * qos_pen
            r = float(np.clip(r_raw, -self.r_max, self.r_max))
            g = (self.d_a * r / max(self.last_delta, 1e-9)) * self.last_v[s_idx]
            self.theta[s_idx] += eta * g

        self.theta = np.clip(self.theta, -8.0, 8.0)

        # Projected dual-price updates.
        ex_b = metrics["rate_utilization"].astype(float) - 1.0
        ex_c = metrics["compute_utilization"].astype(float) - 1.0
        ex_t = float(metrics["transport_utilization"] - 1.0)
        self.mu_b = np.clip(self.mu_b + self.price_eta * ex_b, 0.0, None)
        self.mu_c = np.clip(self.mu_c + self.price_eta * ex_c, 0.0, None)
        self.mu_t = max(0.0, self.mu_t + self.price_eta * ex_t)

        self.prev_action_vec = self.last_action_vec.copy()
        self.t += 1

    def _actions_from_theta(self, theta: np.ndarray) -> tuple[Dict[str, np.ndarray], np.ndarray]:
        b = np.zeros((self.s, self.k), dtype=float)
        c = np.zeros((self.s, self.m), dtype=float)
        tau_raw = np.zeros(self.s, dtype=float)
        x = np.zeros((self.s, self.m), dtype=int)

        for s_idx in range(self.s):
            b_score = self._positive(theta[s_idx, : self.k])
            b_share = b_score / np.maximum(np.sum(b_score), 1e-9)
            b[s_idx] = b_share * (np.sum(self.b_k) / self.s)

            c_score = self._positive(theta[s_idx, self.k : self.k + self.m])
            m_idx = int(np.argmax(c_score))
            x[s_idx, m_idx] = 1
            # Association is exclusive; bid for the whole host and let the capacity
            # projection split it if another slice picked the same one.
            c[s_idx, m_idx] = self.c_m[m_idx]

            tau_raw[s_idx] = self._softplus(theta[s_idx, -1]) + 1e-3

        tau = tau_raw / np.maximum(np.sum(tau_raw), 1e-9) * self.t_agg
        b = self._project_domain_capacity(b, self.b_k)
        c = self._project_domain_capacity(c, self.c_m)

        a_vec = np.zeros((self.s, self.d_a), dtype=float)
        for s_idx in range(self.s):
            a_vec[s_idx, : self.k] = b[s_idx]
            a_vec[s_idx, self.k : self.k + self.m] = c[s_idx]
            a_vec[s_idx, -1] = tau[s_idx]
        return {"b": b, "c": c, "tau": tau, "x": x}, a_vec

    @staticmethod
    def _positive(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(x)) + 1e-8

    @staticmethod
    def _softplus(x: float) -> float:
        return float(np.log1p(np.exp(x)))

    @staticmethod
    def _project_domain_capacity(mat: np.ndarray, caps: np.ndarray) -> np.ndarray:
        out = np.maximum(0.0, mat.copy())
        for j in range(out.shape[1]):
            col_sum = np.sum(out[:, j])
            if col_sum > caps[j]:
                out[:, j] *= caps[j] / max(col_sum, 1e-9)
        return out
