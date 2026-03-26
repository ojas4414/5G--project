from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.algorithms.base import AlgorithmOutput, BaseAllocator


@dataclass
class MAANConfig:
    eta_mu: float = 2e-3
    beta: float = 0.4
    mu_max: float = 10.0
    lr_policy: float = 0.02


class MAANAllocator(BaseAllocator):
    """
    Lightweight surrogate of MAAN:
    local policy + dual-price negotiation update (distributed flavor).
    """

    def __init__(self, s: int, k: int, m: int, b_k: np.ndarray, c_m: np.ndarray, t_agg: float, cfg: MAANConfig):
        super().__init__("MAAN")
        self.s = s
        self.k = k
        self.m = m
        self.b_k = b_k
        self.c_m = c_m
        self.t_agg = t_agg
        self.cfg = cfg
        self.price = np.zeros(k + m + 1, dtype=float)
        self.policy_bias = np.ones((s, k + m + 1), dtype=float)

    def reset(self) -> None:
        self.price[:] = 0.0

    def act(self, state: Dict[str, np.ndarray]) -> AlgorithmOutput:
        lam = state["lambda"]
        ch = state["channel"]
        demand = lam / np.maximum(np.mean(lam), 1.0)
        quality = ch / np.maximum(np.mean(ch, axis=1, keepdims=True), 1e-6)
        mu_b = self.price[: self.k]
        mu_c = self.price[self.k : self.k + self.m]
        mu_t = self.price[-1]

        b = np.zeros((self.s, self.k), dtype=float)
        c = np.zeros((self.s, self.m), dtype=float)
        tau = np.zeros(self.s, dtype=float)
        x = np.zeros((self.s, self.m), dtype=int)

        for s_idx in range(self.s):
            b_pref = demand[s_idx] * quality[s_idx] * self.policy_bias[s_idx, : self.k]
            b_pref = b_pref / np.maximum(1.0 + mu_b, 1e-6)
            c_pref = np.roll(np.ones(self.m), s_idx % self.m) * self.policy_bias[s_idx, self.k : self.k + self.m]
            c_pref = c_pref / np.maximum(1.0 + mu_c, 1e-6)

            best_m = int(np.argmax(c_pref))
            x[s_idx, best_m] = 1
            c[s_idx, best_m] = max(1.0, demand[s_idx] * 0.6 * np.sum(self.c_m) / self.s)
            b[s_idx, :] = np.maximum(0.0, b_pref) * np.sum(self.b_k) / (self.s * np.sum(np.maximum(b_pref, 1e-6)))
            tau[s_idx] = max(1.0, demand[s_idx] * self.t_agg / (self.s * max(1.0 + mu_t, 1e-6)))

        return AlgorithmOutput(actions={"b": b, "c": c, "tau": tau, "x": x}, aux={"mean_price": float(np.mean(self.price))})

    def observe(self, state: Dict[str, np.ndarray], metrics: Dict[str, np.ndarray]) -> None:
        reward = metrics["utilities"]
        rate_ok = metrics["qos_rate_ok"].astype(float)
        delay_ok = metrics["qos_delay_ok"].astype(float)
        signal = reward + 0.2 * rate_ok + 0.2 * delay_ok
        self.policy_bias += self.cfg.lr_policy * signal[:, None]
        self.policy_bias = np.clip(self.policy_bias, 0.5, 3.0)

        excess_b = np.mean(metrics["rate_utilization"]) - 1.0
        excess_c = np.mean(metrics["compute_utilization"]) - 1.0
        excess_t = metrics["transport_utilization"] - 1.0
        self.price[: self.k] = np.clip(
            (1 - self.cfg.beta) * self.price[: self.k] + self.cfg.beta * (self.price[: self.k] + self.cfg.eta_mu * excess_b),
            0.0,
            self.cfg.mu_max,
        )
        self.price[self.k : self.k + self.m] = np.clip(
            (1 - self.cfg.beta) * self.price[self.k : self.k + self.m]
            + self.cfg.beta * (self.price[self.k : self.k + self.m] + self.cfg.eta_mu * excess_c),
            0.0,
            self.cfg.mu_max,
        )
        self.price[-1] = float(
            np.clip((1 - self.cfg.beta) * self.price[-1] + self.cfg.beta * (self.price[-1] + self.cfg.eta_mu * excess_t), 0.0, self.cfg.mu_max)
        )
