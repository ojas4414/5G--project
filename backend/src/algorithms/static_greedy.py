from __future__ import annotations

from typing import Dict

import numpy as np

from src.algorithms.base import AlgorithmOutput, BaseAllocator


class StaticGreedyAllocator(BaseAllocator):
    def __init__(self, slice_weights: np.ndarray, s: int, k: int, m: int, b_k: np.ndarray, c_m: np.ndarray, t_agg: float):
        super().__init__("Static_Greedy")
        self.slice_weights = slice_weights / np.sum(slice_weights)
        self.s = s
        self.k = k
        self.m = m
        self.b_k = b_k
        self.c_m = c_m
        self.t_agg = t_agg
        self.priority = np.argsort(-self.slice_weights)

    def act(self, state: Dict[str, np.ndarray]) -> AlgorithmOutput:
        b = np.outer(self.slice_weights, self.b_k)
        c_raw = np.outer(self.slice_weights, self.c_m)
        tau = self.slice_weights * self.t_agg
        x = np.zeros((self.s, self.m), dtype=int)
        channel = state["channel"]

        for s_idx in self.priority:
            scores = np.argsort(-channel[s_idx])
            best_m = int(scores[0] % self.m)
            x[s_idx, best_m] = 1
        c = c_raw * x
        return AlgorithmOutput(actions={"b": b, "c": c, "tau": tau, "x": x}, aux={})
