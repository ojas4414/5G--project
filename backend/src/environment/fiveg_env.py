"""End-to-end 5G slicing environment with an explicit, dimensionally consistent delay model.

Units (every quantity in this module carries one, and they are checked against each other):

    lambda_s   [bit/s]        offered traffic of slice s
    b_{s,k}    [PRB]          physical resource blocks of gNB k given to slice s
    W_prb      [Hz]           bandwidth of one PRB (12 subcarriers x 15 kHz)
    R_s        [bit/s]        achieved radio rate  = sum_k b_{s,k} * W_prb * log2(1+SINR)
    tau_s      [bit/s]        transport (fronthaul/midhaul) capacity given to slice s
    c_{s,m}    [cycle/s]      MEC compute capacity given to slice s on host m
    omega_s    [cycle/bit]    processing density of slice s
    C_s        [bit/s]        compute service rate = (sum_m c_{s,m}) / omega_s
    L_pkt      [bit]          mean packet size
    d_*        [s]            delay components

Each of the three domains is modelled as an M/M/1 queue fed by lambda_s and served at
mu [bit/s].  The mean sojourn time of an M/M/1 queue with packet size L is

    W = L / (mu - lambda)        [s]        for mu > lambda

which is dimensionally a time.  When mu <= lambda the queue is unstable; we saturate the
delay at ``D_UNSTABLE`` rather than returning inf so that gradients and plots stay finite.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# Delay reported for an unstable (overloaded) queue. Large enough to dominate any QoS
# budget, small enough to keep utilities and plots numerically well behaved.
D_UNSTABLE = 1.0  # [s]

# Saturation point of the linear delay cost in the utility, expressed as a multiple of the
# slice's delay budget. Keeps utility O(1) instead of O(D_UNSTABLE / d_max).
DELAY_PENALTY_CAP = 3.0


@dataclass
class SliceConfig:
    """Per-slice service level agreement and traffic characteristics.

    r_min  [bit/s]      minimum acceptable throughput
    d_max  [s]          end-to-end delay budget
    omega  [cycle/bit]  processing density (how much compute one bit of this slice costs)
    alpha/beta/gamma    utility weights for rate reward, delay cost, violation penalty
    """

    name: str
    r_min: float
    d_max: float
    alpha: float
    beta: float
    gamma: float
    omega: float
    eps_urlcc: float = 0.01


class FiveGEnvironment:
    # Base offered-load range per slice, in bit/s, before scenario load scaling.
    lam_low: float = 8e5
    lam_high: float = 4.2e6

    def __init__(
        self,
        slice_configs: list[SliceConfig],
        num_gnbs: int = 3,
        num_mecs: int = 3,
        b_k: float = 160.0,        # [PRB] per gNB  -> 160 * 180 kHz = 28.8 MHz
        c_m: float = 4.0e8,        # [cycle/s] per MEC host
        t_agg: float = 6.0e7,      # [bit/s] aggregate transport capacity
        w_prb: float = 180e3,      # [Hz] one PRB
        n0: float = 1e-9,          # [W/Hz] noise PSD
        t_tti: float = 1e-3,       # [s] transmission time interval
        l_pkt: float = 12000.0,    # [bit] mean packet (1500 B)
        seed: int = 42,
        lambda_trace: np.ndarray | None = None,
        channel_trace: np.ndarray | None = None,
    ) -> None:
        self.seed = int(seed)
        self.rng_dyn = np.random.default_rng(self.seed)
        self.rng_eval = np.random.default_rng(self.seed + 1)

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
        self.l_pkt = float(l_pkt)
        self.pk = np.ones(self.k, dtype=float)
        self.prices = np.zeros(self.k + self.m + 1, dtype=float)

        self.omega = np.array([cfg.omega for cfg in slice_configs], dtype=float)
        self.r_min = np.array([cfg.r_min for cfg in slice_configs], dtype=float)
        self.d_max = np.array([cfg.d_max for cfg in slice_configs], dtype=float)

        self.lambda_trace = None if lambda_trace is None else np.asarray(lambda_trace, dtype=float)
        self.channel_trace = None if channel_trace is None else np.asarray(channel_trace, dtype=float)
        if self.lambda_trace is not None and self.lambda_trace.shape[1] != self.s:
            raise ValueError(f"lambda_trace shape must be [T, {self.s}]")
        if self.channel_trace is not None and self.channel_trace.shape[1:] != (self.s, self.k):
            raise ValueError(f"channel_trace shape must be [T, {self.s}, {self.k}]")

        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        self.t = 0
        self.prev_b = np.zeros((self.s, self.k), dtype=float)
        self.prev_c = np.zeros((self.s, self.m), dtype=float)
        self.prev_tau = np.zeros(self.s, dtype=float)
        self.prev_rates = np.zeros(self.s, dtype=float)
        self.prev_delays = np.zeros(self.s, dtype=float)

        self.lambda_s, self.g_sk = self._exogenous_at(self.t)
        return self.get_state()

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "lambda": self.lambda_s.copy(),
            "channel": self.g_sk.copy(),
            "prices": self.prices.copy(),
            "prev_b": self.prev_b.copy(),
            "prev_c": self.prev_c.copy(),
            "prev_tau": self.prev_tau.copy(),
            "prev_rates": self.prev_rates.copy(),
            "prev_delays": self.prev_delays.copy(),
        }

    # ------------------------------------------------------------------ delays

    def _queue_delay(self, service_rate: np.ndarray, arrival_rate: np.ndarray) -> np.ndarray:
        """Mean M/M/1 sojourn time  W = L / (mu - lambda)  in seconds.

        Both arguments are in bit/s, ``l_pkt`` is in bit, so the quotient is a time.
        Unstable queues (mu <= lambda) saturate at ``D_UNSTABLE`` instead of diverging.
        """
        headroom = np.asarray(service_rate, dtype=float) - np.asarray(arrival_rate, dtype=float)
        # The smallest headroom that still yields a delay below the saturation cap.
        min_headroom = self.l_pkt / D_UNSTABLE
        return self.l_pkt / np.maximum(headroom, min_headroom)

    def _delay_components(
        self,
        rates: np.ndarray,
        tau: np.ndarray,
        c_slice: np.ndarray,
        lam: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Radio / transport / compute delay, all in seconds."""
        d_radio = self._queue_delay(rates, lam) + self.t_tti
        d_trans = self._queue_delay(tau, lam)
        # Compute service rate in bit/s: cycles/s available divided by cycles needed per bit.
        compute_service = c_slice / np.maximum(self.omega, 1e-9)
        d_comp = self._queue_delay(compute_service, lam)
        return d_radio, d_trans, d_comp

    # ------------------------------------------------------------------- step

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
        b_prop = actions["b"].astype(float).copy()
        c_prop = actions["c"].astype(float).copy()
        tau_prop = actions["tau"].astype(float).copy()
        x = self._to_one_hot(actions["x"].astype(float))

        c_prop = c_prop * x
        b_scaled = self._enforce_domain_capacity(b_prop, self.b_k)
        c_act = self._enforce_domain_capacity(c_prop, self.c_m)
        tau_act = tau_prop * min(1.0, self.t_agg / max(np.sum(tau_prop), 1e-8))

        sinr = self._compute_sinr(self.g_sk)
        marginal = self.w_prb * np.log2(1.0 + sinr)
        b_act = self._round_prbs(b_scaled, self.b_k, marginal)

        rates = np.sum(b_act * marginal, axis=1)
        c_slice = np.sum(c_act, axis=1)
        d_radio, d_trans, d_comp = self._delay_components(rates, tau_act, c_slice, self.lambda_s)
        delays = d_radio + d_trans + d_comp

        utilities = self._compute_utility(rates, delays)
        qos_rate_ok = rates >= self.r_min
        qos_delay_ok = delays <= self.d_max
        violation = (~qos_rate_ok) | (~qos_delay_ok)

        self.prev_b = b_act.copy()
        self.prev_c = c_act.copy()
        self.prev_tau = tau_act.copy()
        self.prev_rates = rates.copy()
        self.prev_delays = delays.copy()
        self.t += 1
        self.lambda_s, self.g_sk = self._exogenous_at(self.t)

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
            "b_act": b_act,
            "c_act": c_act,
            "tau_act": tau_act,
            "x": x,
        }
        return self.get_state(), metrics

    def saa_urlcc_violation_probability(self, actions: Dict[str, np.ndarray], n_mc: int = 64, urlcc_idx: int = 1) -> float:
        """Sample-average approximation of Pr(D_urlcc > d_max_urlcc) for the current slot.

        Uses a dedicated evaluation RNG so SAA never perturbs the rollout stream, and the
        same PRB rounding rule as :meth:`step` so the estimate refers to the allocation
        that is actually played.
        """
        if self.s <= urlcc_idx:
            return float("nan")

        b_prop = actions["b"].astype(float).copy()
        c_prop = actions["c"].astype(float).copy()
        tau_prop = actions["tau"].astype(float).copy()
        x = self._to_one_hot(actions["x"].astype(float))
        c_prop = c_prop * x

        # Round with the marginal rates of the *current* slot, matching step().
        marginal_now = self.w_prb * np.log2(1.0 + self._compute_sinr(self.g_sk))
        b_act = self._round_prbs(self._enforce_domain_capacity(b_prop, self.b_k), self.b_k, marginal_now)
        c_act = self._enforce_domain_capacity(c_prop, self.c_m)
        tau_act = tau_prop * min(1.0, self.t_agg / max(np.sum(tau_prop), 1e-8))
        c_slice = np.sum(c_act, axis=1)

        # Vectorised over the n_mc scenarios: shapes are [n_mc, s] and [n_mc, s, k].
        lam = self.rng_eval.uniform(self.lam_low, self.lam_high, size=(n_mc, self.s))
        gains = self.rng_eval.exponential(scale=1.0, size=(n_mc, self.s, self.k))
        sinr = self._compute_sinr(gains)
        rates = np.sum(b_act[None, :, :] * self.w_prb * np.log2(1.0 + sinr), axis=2)

        d_radio, d_trans, d_comp = self._delay_components(rates, tau_act[None, :], c_slice[None, :], lam)
        delays = d_radio + d_trans + d_comp
        return float(np.mean(delays[:, urlcc_idx] > self.slice_configs[urlcc_idx].d_max))

    def update_prices(self, eta_mu: float = 1e-3, beta: float = 0.4, mu_max: float = 10.0) -> None:
        """Dual ascent on the capacity constraints, with step size damped by ``beta``."""
        excess_b = np.sum(self.prev_b, axis=0) / np.maximum(self.b_k, 1e-9) - 1.0
        excess_c = np.sum(self.prev_c, axis=0) / np.maximum(self.c_m, 1e-9) - 1.0
        excess_t = np.sum(self.prev_tau) / max(self.t_agg, 1e-9) - 1.0
        step = beta * eta_mu
        mu_b = np.clip(self.prices[: self.k] + step * excess_b, 0.0, mu_max)
        mu_c = np.clip(self.prices[self.k : self.k + self.m] + step * excess_c, 0.0, mu_max)
        mu_t = float(np.clip(self.prices[-1] + step * excess_t, 0.0, mu_max))
        self.prices = np.concatenate([mu_b, mu_c, np.array([mu_t])])

    # -------------------------------------------------------------- exogenous

    def _exogenous_at(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self.lambda_trace is not None and idx < self.lambda_trace.shape[0]:
            lam = self.lambda_trace[idx].copy()
        else:
            lam = self.rng_dyn.uniform(self.lam_low, self.lam_high, size=self.s)

        if self.channel_trace is not None and idx < self.channel_trace.shape[0]:
            ch = self.channel_trace[idx].copy()
        else:
            ch = self.rng_dyn.exponential(scale=1.0, size=(self.s, self.k))
        return lam, ch

    def _compute_sinr(self, gains: np.ndarray) -> np.ndarray:
        """SINR per (slice, gNB). Accepts [s, k] or a batched [n, s, k]."""
        signal = gains * self.pk
        inter = np.sum(signal, axis=-1, keepdims=True) - signal
        return signal / np.maximum(inter + self.n0 * self.w_prb, 1e-12)

    def _compute_utility(self, rates: np.ndarray, delays: np.ndarray) -> np.ndarray:
        """Concave (proportional-fair) rate reward minus delay cost and violation penalty.

        Two shaping choices matter here:

        * The rate term is logarithmic, not linear. A linear term makes the optimum
          degenerate -- hand every PRB to whichever slice has the largest alpha/r_min --
          whereas log utility has diminishing returns and yields balanced allocations.
        * The linear delay cost is saturated at ``DELAY_PENALTY_CAP``. An unstable queue
          against a tight budget gives D/d_max ~ 125, which would swamp the O(1) rate term
          and reduce the objective to "minimise delay" alone. The sigmoid term ``phi``
          already registers *that* a violation happened; the linear term only needs to
          grade severity within a bounded range.
        """
        alpha = np.array([cfg.alpha for cfg in self.slice_configs], dtype=float)
        beta = np.array([cfg.beta for cfg in self.slice_configs], dtype=float)
        gamma = np.array([cfg.gamma for cfg in self.slice_configs], dtype=float)

        d_ratio = delays / np.maximum(self.d_max, 1e-9)
        phi = 1.0 / (1.0 + np.exp(-np.clip(20.0 * (d_ratio - 1.0), -60.0, 60.0)))
        delay_cost = np.minimum(d_ratio, DELAY_PENALTY_CAP)
        return alpha * np.log1p(rates / np.maximum(self.r_min, 1e-9)) - beta * delay_cost - gamma * phi

    @staticmethod
    def _to_one_hot(x: np.ndarray) -> np.ndarray:
        x_out = np.zeros_like(x, dtype=int)
        if x.ndim != 2 or x.shape[1] == 0:
            return x_out
        for s_idx in range(x.shape[0]):
            m_idx = int(np.argmax(x[s_idx]))
            x_out[s_idx, m_idx] = 1
        return x_out

    @staticmethod
    def _enforce_domain_capacity(prop: np.ndarray, cap: np.ndarray) -> np.ndarray:
        total = np.sum(prop, axis=0)
        scale = np.minimum(1.0, cap / np.maximum(total, 1e-9))
        return prop * scale[None, :]

    @staticmethod
    def _round_prbs(b: np.ndarray, cap: np.ndarray, marginal: np.ndarray | None = None) -> np.ndarray:
        """Round a fractional PRB request to integers, never exceeding capacity.

        PRBs are indivisible, so the request is floored and the leftover fractional demand
        is handed out largest-remainder-first. Crucially this only distributes PRBs the
        allocator actually asked for: an allocator that requests less than capacity keeps
        the unused PRBs idle, which is what makes ``rate_utilization`` an informative
        metric rather than a constant 1.0.
        """
        b = np.maximum(b, 0.0)
        b_int = np.floor(b).astype(int)
        for k in range(b.shape[1]):
            target = min(int(np.floor(np.sum(b[:, k]))), int(round(cap[k])))
            rem = target - int(np.sum(b_int[:, k]))

            if rem > 0:
                # Largest-remainder apportionment, tie-broken by channel quality.
                frac = b[:, k] - np.floor(b[:, k])
                score = frac if marginal is None else frac + 1e-9 * marginal[:, k]
                for s_idx in np.argsort(-score)[:rem]:
                    b_int[s_idx, k] += 1
            elif rem < 0:
                over = -rem
                # Reclaim from the weakest links first.
                order = np.argsort(b_int[:, k]) if marginal is None else np.argsort(marginal[:, k])
                for s_idx in order:
                    if over <= 0:
                        break
                    take = min(over, b_int[s_idx, k])
                    b_int[s_idx, k] -= take
                    over -= take
        return b_int.astype(float)
