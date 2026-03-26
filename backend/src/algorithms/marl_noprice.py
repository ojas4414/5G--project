from __future__ import annotations

from typing import Dict

import numpy as np

from src.algorithms.base import AlgorithmOutput, BaseAllocator


class IndependentMAPPOAllocator(BaseAllocator):
    def __init__(self, s: int, k: int, m: int, b_k: np.ndarray, c_m: np.ndarray, t_agg: float):
        super().__init__("Independent_MAPPO")
        self.s = s
        self.k = k
        self.m = m
        self.b_k = b_k
        self.c_m = c_m
        self.t_agg = t_agg
        self.bias = np.ones((s, k + m + 1), dtype=float)
        self.lr = 0.015

    def act(self, state: Dict[str, np.ndarray]) -> AlgorithmOutput:
        lam = state["lambda"]
        ch = state["channel"]
        demand = lam / np.maximum(np.mean(lam), 1.0)
        quality = ch / np.maximum(np.mean(ch, axis=1, keepdims=True), 1e-6)
        b = np.zeros((self.s, self.k), dtype=float)
        c = np.zeros((self.s, self.m), dtype=float)
        tau = np.zeros(self.s, dtype=float)
        x = np.zeros((self.s, self.m), dtype=int)

        for s_idx in range(self.s):
            b_pref = demand[s_idx] * quality[s_idx] * self.bias[s_idx, : self.k]
            best_m = int(np.argmax(self.bias[s_idx, self.k : self.k + self.m]))
            x[s_idx, best_m] = 1
            c[s_idx, best_m] = max(1.0, demand[s_idx] * 0.55 * np.sum(self.c_m) / self.s)
            b[s_idx, :] = np.maximum(0.0, b_pref) * np.sum(self.b_k) / (self.s * np.sum(np.maximum(b_pref, 1e-6)))
            tau[s_idx] = max(1.0, demand[s_idx] * self.t_agg / self.s)

        return AlgorithmOutput(actions={"b": b, "c": c, "tau": tau, "x": x}, aux={})

    def observe(self, state: Dict[str, np.ndarray], metrics: Dict[str, np.ndarray]) -> None:
        signal = metrics["utilities"] + 0.1 * metrics["qos_ok"].astype(float)
        self.bias += self.lr * signal[:, None]
        self.bias = np.clip(self.bias, 0.4, 3.5)
