from __future__ import annotations

from typing import Dict

import numpy as np

from src.algorithms.base import AlgorithmOutput, BaseAllocator


class CADMMAllocator(BaseAllocator):
    def __init__(self, s: int, k: int, m: int, b_k: np.ndarray, c_m: np.ndarray, t_agg: float, rounds: int = 8, rho: float = 0.8):
        super().__init__("C_ADMM")
        self.s = s
        self.k = k
        self.m = m
        self.b_k = b_k
        self.c_m = c_m
        self.t_agg = t_agg
        self.rounds = rounds
        self.rho = rho

    def act(self, state: Dict[str, np.ndarray]) -> AlgorithmOutput:
        lam = state["lambda"]
        demand = lam / np.maximum(np.mean(lam), 1.0)

        b = np.outer(demand, self.b_k / np.sum(self.b_k))
        c = np.outer(demand, self.c_m / np.sum(self.c_m))
        tau = demand * self.t_agg / np.sum(demand)
        x = np.zeros((self.s, self.m), dtype=int)
        for i in range(self.s):
            x[i, i % self.m] = 1
        c = c * x

        z_b = b.copy()
        z_c = c.copy()
        z_t = tau.copy()
        l_b = np.zeros_like(b)
        l_c = np.zeros_like(c)
        l_t = np.zeros_like(tau)
        r_pri = 0.0
        r_dual = 0.0

        for _ in range(self.rounds):
            b = np.maximum(0.0, z_b - l_b / self.rho + 0.1 * b)
            c = np.maximum(0.0, z_c - l_c / self.rho + 0.1 * c) * x
            tau = np.maximum(0.0, z_t - l_t / self.rho + 0.1 * tau)

            z_b_prev = z_b.copy()
            z_c_prev = z_c.copy()
            z_t_prev = z_t.copy()
            z_b = self._project_capacity(b + l_b / self.rho, self.b_k)
            z_c = self._project_capacity(c + l_c / self.rho, self.c_m)
            z_t = self._project_scalar(tau + l_t / self.rho, self.t_agg)

            l_b = l_b + self.rho * (b - z_b)
            l_c = l_c + self.rho * (c - z_c)
            l_t = l_t + self.rho * (tau - z_t)
            r_pri = float(np.sqrt(np.sum((b - z_b) ** 2) + np.sum((c - z_c) ** 2) + np.sum((tau - z_t) ** 2)))
            r_dual = float(
                self.rho
                * np.sqrt(np.sum((z_b - z_b_prev) ** 2) + np.sum((z_c - z_c_prev) ** 2) + np.sum((z_t - z_t_prev) ** 2))
            )
        return AlgorithmOutput(actions={"b": z_b, "c": z_c, "tau": z_t, "x": x}, aux={"r_pri": r_pri, "r_dual": r_dual})

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
