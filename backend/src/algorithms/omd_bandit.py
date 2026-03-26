from __future__ import annotations

from typing import Dict

import numpy as np

from src.algorithms.base import AlgorithmOutput, BaseAllocator


class OMDBanditAllocator(BaseAllocator):
    def __init__(self, s: int, k: int, m: int, b_k: np.ndarray, c_m: np.ndarray, t_agg: float):
        super().__init__("OMD_BF")
        self.s = s
        self.k = k
        self.m = m
        self.b_k = b_k
        self.c_m = c_m
        self.t_agg = t_agg
        self.t = 1
        self.eta0 = 0.25
        self.theta_b = np.ones((s, k), dtype=float) / k
        self.theta_c = np.ones((s, m), dtype=float) / m
        self.theta_t = np.ones(s, dtype=float) / s

    def reset(self) -> None:
        self.t = 1

    def act(self, state: Dict[str, np.ndarray]) -> AlgorithmOutput:
        delta = self.t ** (-0.25)
        vb = np.random.normal(0.0, 1.0, size=self.theta_b.shape)
        vc = np.random.normal(0.0, 1.0, size=self.theta_c.shape)
        vt = np.random.normal(0.0, 1.0, size=self.theta_t.shape)
        b_prob = self._simplex_proj(self.theta_b + delta * vb)
        c_prob = self._simplex_proj(self.theta_c + delta * vc)
        t_prob = self._simplex_proj_vec(self.theta_t + delta * vt)

        b = b_prob * self.b_k[None, :]
        x = np.zeros((self.s, self.m), dtype=int)
        c = np.zeros((self.s, self.m), dtype=float)
        for s_idx in range(self.s):
            best_m = int(np.argmax(c_prob[s_idx]))
            x[s_idx, best_m] = 1
            c[s_idx, best_m] = c_prob[s_idx, best_m] * np.sum(self.c_m)
        tau = t_prob * self.t_agg
        return AlgorithmOutput(actions={"b": b, "c": c, "tau": tau, "x": x}, aux={"delta": float(delta)})

    def observe(self, state: Dict[str, np.ndarray], metrics: Dict[str, np.ndarray]) -> None:
        r = metrics["utilities"]
        eta = self.eta0 / np.sqrt(self.t)
        grad = r[:, None]
        self.theta_b = self._simplex_proj(self.theta_b + eta * grad)
        self.theta_c = self._simplex_proj(self.theta_c + eta * grad)
        self.theta_t = self._simplex_proj_vec(self.theta_t + eta * r)
        self.t += 1

    @staticmethod
    def _simplex_proj(mat: np.ndarray) -> np.ndarray:
        out = np.maximum(mat, 1e-8)
        out = out / np.sum(out, axis=1, keepdims=True)
        return out

    @staticmethod
    def _simplex_proj_vec(vec: np.ndarray) -> np.ndarray:
        out = np.maximum(vec, 1e-8)
        out = out / np.sum(out)
        return out
