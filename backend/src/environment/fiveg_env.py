from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SliceConfig:
    name: str
    r_min: float
    d_max: float
    alpha: float
    beta: float
    gamma: float
    omega: float
    eps_urlcc: float = 0.01


class FiveGEnvironment:
    def __init__(
        self,
        slice_configs: list[SliceConfig],
        num_gnbs: int = 3,
        num_mecs: int = 3,
        b_k: float = 160.0,
        c_m: float = 350.0,
        t_agg: float = 420.0,
        w_prb: float = 180e3,
        n0: float = 1e-9,
        t_tti: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.slice_configs = slice_configs
        self.s = len(slice_configs)
        self.k = num_gnbs
        self.m = num_mecs
        self.b_k = np.full(self.k, b_k, dtype=float)
        self.c_m = np.full(self.m, c_m, dtype=float)
        self.t_agg = float(t_agg)
        self.w_prb = w_prb
        self.n0 = n0
        self.t_tti = t_tti
        self.pk = np.ones(self.k, dtype=float)
        self.prices = np.zeros(self.k + self.m + 1, dtype=float)
        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        self.t = 0
        self.lambda_s = self.rng.uniform(8e5, 4e6, size=self.s)
        self.g_sk = self.rng.exponential(scale=1.0, size=(self.s, self.k))
        self.prev_b = np.zeros((self.s, self.k), dtype=float)
        self.prev_c = np.zeros((self.s, self.m), dtype=float)
        self.prev_tau = np.zeros(self.s, dtype=float)
        return self.get_state()

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "lambda": self.lambda_s.copy(),
            "channel": self.g_sk.copy(),
            "prices": self.prices.copy(),
            "prev_b": self.prev_b.copy(),
            "prev_c": self.prev_c.copy(),
            "prev_tau": self.prev_tau.copy(),
        }

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
        b_prop = actions["b"].astype(float).copy()
        c_prop = actions["c"].astype(float).copy()
        tau_prop = actions["tau"].astype(float).copy()
        x = actions["x"].astype(int).copy()

        c_prop = c_prop * x
        b_act = self._enforce_domain_capacity(b_prop, self.b_k)
        c_act = self._enforce_domain_capacity(c_prop, self.c_m)
        tau_act = tau_prop * min(1.0, self.t_agg / max(np.sum(tau_prop), 1e-8))
        b_act = self._round_prbs(b_act, self.b_k)

        sinr = self._compute_sinr(self.g_sk)
        rates = np.sum(b_act * self.w_prb * np.log2(1.0 + sinr), axis=1)
        d_radio = 1.0 / np.maximum(rates - self.lambda_s, 1e3) + self.t_tti
        d_trans = self.lambda_s / np.maximum(tau_act, 1e-6)
        c_slice = np.sum(c_act, axis=1)
        omega = np.array([cfg.omega for cfg in self.slice_configs], dtype=float)
        d_comp = omega / np.maximum(c_slice, 1e-6)
        delays = d_radio + d_trans + d_comp

        utilities = self._compute_utility(rates, delays)
        qos_rate_ok = np.array([rates[i] >= cfg.r_min for i, cfg in enumerate(self.slice_configs)])
        qos_delay_ok = np.array([delays[i] <= cfg.d_max for i, cfg in enumerate(self.slice_configs)])
        violation = (~qos_rate_ok) | (~qos_delay_ok)

        self.prev_b = b_act.copy()
        self.prev_c = c_act.copy()
        self.prev_tau = tau_act.copy()
        self.lambda_s = self.rng.uniform(8e5, 4.2e6, size=self.s)
        self.g_sk = self.rng.exponential(scale=1.0, size=(self.s, self.k))
        self.t += 1

        metrics = {
            "rates": rates,
            "delays": delays,
            "utilities": utilities,
            "d_radio": d_radio,
            "d_trans": d_trans,
            "d_comp": d_comp,
            "qos_rate_ok": qos_rate_ok,
            "qos_delay_ok": qos_delay_ok,
            "qos_ok": ~(violation),
            "rate_utilization": np.sum(b_act, axis=0) / np.maximum(self.b_k, 1e-8),
            "compute_utilization": np.sum(c_act, axis=0) / np.maximum(self.c_m, 1e-8),
            "transport_utilization": np.sum(tau_act) / max(self.t_agg, 1e-8),
        }
        return self.get_state(), metrics

    def saa_urlcc_violation_probability(self, actions: Dict[str, np.ndarray], n_mc: int = 64, urlcc_idx: int = 1) -> float:
        """
        Monte Carlo estimate of Pr(D_urlcc > Dmax_urlcc) for the current slot dynamics.
        """
        b_prop = actions["b"].astype(float).copy()
        c_prop = actions["c"].astype(float).copy()
        tau_prop = actions["tau"].astype(float).copy()
        x = actions["x"].astype(int).copy()
        c_prop = c_prop * x
        b_act = self._round_prbs(self._enforce_domain_capacity(b_prop, self.b_k), self.b_k)
        c_act = self._enforce_domain_capacity(c_prop, self.c_m)
        tau_act = tau_prop * min(1.0, self.t_agg / max(np.sum(tau_prop), 1e-8))
        c_slice = np.sum(c_act, axis=1)
        omega = np.array([cfg.omega for cfg in self.slice_configs], dtype=float)
        exceed = 0
        for _ in range(n_mc):
            lam = self.rng.uniform(8e5, 4.2e6, size=self.s)
            gains = self.rng.exponential(scale=1.0, size=(self.s, self.k))
            sinr = self._compute_sinr(gains)
            rates = np.sum(b_act * self.w_prb * np.log2(1.0 + sinr), axis=1)
            d_radio = 1.0 / np.maximum(rates - lam, 1e3) + self.t_tti
            d_trans = lam / np.maximum(tau_act, 1e-6)
            d_comp = omega / np.maximum(c_slice, 1e-6)
            delays = d_radio + d_trans + d_comp
            if delays[urlcc_idx] > self.slice_configs[urlcc_idx].d_max:
                exceed += 1
        return float(exceed / max(n_mc, 1))

    def update_prices(self, eta_mu: float = 1e-3, beta: float = 0.4, mu_max: float = 10.0) -> None:
        excess_b = np.sum(self.prev_b, axis=0) - self.b_k
        excess_c = np.sum(self.prev_c, axis=0) - self.c_m
        excess_t = np.sum(self.prev_tau) - self.t_agg
        mu_b = self.prices[: self.k]
        mu_c = self.prices[self.k : self.k + self.m]
        mu_t = self.prices[-1]
        mu_b = np.clip((1.0 - beta) * mu_b + beta * (mu_b + eta_mu * excess_b), 0.0, mu_max)
        mu_c = np.clip((1.0 - beta) * mu_c + beta * (mu_c + eta_mu * excess_c), 0.0, mu_max)
        mu_t = float(np.clip((1.0 - beta) * mu_t + beta * (mu_t + eta_mu * excess_t), 0.0, mu_max))
        self.prices = np.concatenate([mu_b, mu_c, np.array([mu_t])])

    def _compute_sinr(self, gains: np.ndarray) -> np.ndarray:
        inter = np.sum(gains * self.pk[None, :], axis=1, keepdims=True) - gains * self.pk[None, :]
        return (self.pk[None, :] * gains) / np.maximum(inter + self.n0 * self.w_prb, 1e-12)

    def _compute_utility(self, rates: np.ndarray, delays: np.ndarray) -> np.ndarray:
        out = np.zeros(self.s, dtype=float)
        for i, cfg in enumerate(self.slice_configs):
            d_ratio = (delays[i] - cfg.d_max) / max(cfg.d_max, 1e-9)
            phi = 1.0 / (1.0 + np.exp(-20.0 * d_ratio))
            out[i] = cfg.alpha * (rates[i] / cfg.r_min) - cfg.beta * (delays[i] / cfg.d_max) - cfg.gamma * phi
        return out

    @staticmethod
    def _enforce_domain_capacity(prop: np.ndarray, cap: np.ndarray) -> np.ndarray:
        total = np.sum(prop, axis=0)
        scale = np.minimum(1.0, cap / np.maximum(total, 1e-9))
        return prop * scale[None, :]

    @staticmethod
    def _round_prbs(b: np.ndarray, cap: np.ndarray) -> np.ndarray:
        b_int = np.floor(b).astype(int)
        for k in range(b.shape[1]):
            rem = int(max(cap[k] - np.sum(b_int[:, k]), 0))
            if rem > 0:
                frac = b[:, k] - np.floor(b[:, k])
                order = np.argsort(-frac)
                for idx in order[:rem]:
                    b_int[idx, k] += 1
            elif rem < 0:
                over = -rem
                order = np.argsort(-b_int[:, k])
                for idx in order:
                    if over <= 0:
                        break
                    take = min(over, b_int[idx, k])
                    b_int[idx, k] -= take
                    over -= take
        return b_int.astype(float)