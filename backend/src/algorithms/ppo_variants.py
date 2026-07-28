from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.base import AlgorithmOutput, BaseAllocator


def _softplus(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable softplus, keeping resource shares strictly positive."""
    x = np.asarray(x, dtype=float)
    return np.logaddexp(0.0, x) + 1e-6


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, out_dim)
        self.log_std = nn.Parameter(torch.full((out_dim,), -0.7))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mean = self.mean(h)
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)
        return mean, std


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    lr_actor: float = 3e-4
    lr_critic: float = 5e-4
    update_every: int = 64
    ppo_epochs: int = 5
    minibatch: int = 64
    ent_coef: float = 0.01
    stab_coef: float = 0.02
    qos_coef: float = 1.0
    price_eta: float = 2e-3
    price_beta: float = 0.4
    price_max: float = 10.0


class _BasePPO(BaseAllocator):
    def __init__(
        self,
        name: str,
        s: int,
        k: int,
        m: int,
        b_k: np.ndarray,
        c_m: np.ndarray,
        t_agg: float,
        use_prices: bool,
        cfg: PPOConfig,
        r_min: np.ndarray | None = None,
        d_max: np.ndarray | None = None,
    ):
        super().__init__(name)
        self.s = s
        self.k = k
        self.m = m
        self.b_k = b_k
        self.c_m = c_m
        self.t_agg = t_agg
        self.use_prices = use_prices
        self.cfg = cfg
        self.r_min = np.full(self.s, 10e6) if r_min is None else r_min.astype(float)
        self.d_max = np.full(self.s, 0.02) if d_max is None else d_max.astype(float)

        self.act_dim = k + m + 1 + m  # b, c, tau, x_logits
        local_obs_dim = 2 * k + m + 4 + (k + m + 1 if use_prices else 0)
        self.obs_dim = local_obs_dim
        self.actors = [Actor(self.obs_dim, self.act_dim) for _ in range(s)]
        self.opt_actors = [optim.Adam(a.parameters(), lr=cfg.lr_actor) for a in self.actors]
        self.critic = Critic(self.obs_dim * s + self.act_dim * s)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self._buf: List[Dict] = []
        self._last = None
        self.prices = np.zeros(k + m + 1, dtype=float)
        self.prev_action_compact = np.zeros((self.s, self.k + self.m + 1), dtype=float)

    def reset(self) -> None:
        self._buf = []
        self._last = None
        self.prices[:] = 0.0
        self.prev_action_compact.fill(0.0)

    def _obs_per_agent(self, state: Dict[str, np.ndarray], s_idx: int) -> np.ndarray:
        lam_all = state["lambda"]
        lam = np.array([lam_all[s_idx] / max(np.mean(lam_all), 1.0)], dtype=float)
        chan = state["channel"][s_idx] / max(np.mean(state["channel"][s_idx]), 1e-6)

        prev_rates = state.get("prev_rates", np.zeros(self.s, dtype=float))
        prev_delays = state.get("prev_delays", np.zeros(self.s, dtype=float))
        prev_r = np.array([prev_rates[s_idx] / max(self.r_min[s_idx], 1e-9)], dtype=float)
        prev_d = np.array([prev_delays[s_idx] / max(self.d_max[s_idx], 1e-9)], dtype=float)

        prev_b = state["prev_b"][s_idx] / np.maximum(self.b_k, 1e-9)
        prev_c = state["prev_c"][s_idx] / np.maximum(self.c_m, 1e-9)
        prev_tau = np.array([state["prev_tau"][s_idx] / max(self.t_agg, 1e-9)], dtype=float)

        base = np.concatenate([lam, chan, prev_d, prev_r, prev_b, prev_c, prev_tau])
        if self.use_prices:
            base = np.concatenate([base, self.prices / max(self.cfg.price_max, 1.0)])
        return base.astype(np.float32)

    def act(self, state: Dict[str, np.ndarray]) -> AlgorithmOutput:
        obs = [self._obs_per_agent(state, i) for i in range(self.s)]
        b = np.zeros((self.s, self.k), dtype=float)
        c = np.zeros((self.s, self.m), dtype=float)
        tau_raw = np.zeros(self.s, dtype=float)
        x = np.zeros((self.s, self.m), dtype=int)
        info = {"obs": obs, "logp": [], "act_vec": [], "compact": np.zeros((self.s, self.k + self.m + 1), dtype=float)}

        for i in range(self.s):
            o = torch.tensor(obs[i]).unsqueeze(0)
            mean, std = self.actors[i](o)
            dist = torch.distributions.Normal(mean, std)
            z = dist.sample()
            logp = dist.log_prob(z).sum(dim=1)
            a = z.squeeze(0).detach().numpy()

            # Softplus rather than max(0, .): with a rectifier any negative sample
            # collapses to ~1e-6, so a slice can be handed essentially zero transport or
            # radio and drop into an unstable queue purely from exploration noise.
            # Softplus keeps every share strictly positive and smoothly differentiable.
            raw_b = _softplus(a[: self.k])
            raw_c = _softplus(a[self.k : self.k + self.m])
            raw_tau = float(_softplus(a[self.k + self.m]))
            x_logits = a[self.k + self.m + 1 :]
            m_idx = int(np.argmax(x_logits))
            x[i, m_idx] = 1

            b_share = raw_b / np.maximum(np.sum(raw_b), 1e-9)
            b[i] = b_share * (np.sum(self.b_k) / self.s)
            # MEC association is exclusive (one-hot x), so the slice bids for the whole
            # host and the environment's capacity projection arbitrates between slices
            # that picked the same one. Bidding for only 1/m of a host it may have to
            # itself would idle compute for no reason -- the real decision here is
            # *which* MEC, and MAAN's price term is what discourages over-bidding.
            c[i, m_idx] = self.c_m[m_idx]
            tau_raw[i] = raw_tau

            info["logp"].append(float(logp.item()))
            info["act_vec"].append(a)

        tau = tau_raw / np.maximum(np.sum(tau_raw), 1e-9) * self.t_agg
        info["compact"][:, : self.k] = b
        info["compact"][:, self.k : self.k + self.m] = c
        info["compact"][:, -1] = tau
        self._last = info
        return AlgorithmOutput(actions={"b": b, "c": c, "tau": tau, "x": x}, aux={})

    def observe(self, state: Dict[str, np.ndarray], metrics: Dict[str, np.ndarray]) -> None:
        if self._last is None:
            return
        utilities = metrics["utilities"].astype(float)
        delays = metrics["delays"].astype(float)
        b_used = metrics.get("b_act", self._last["compact"][:, : self.k]).astype(float)
        c_used = metrics.get("c_act", self._last["compact"][:, self.k : self.k + self.m]).astype(float)
        tau_used = metrics.get("tau_act", self._last["compact"][:, -1]).astype(float)

        rewards = np.zeros(self.s, dtype=float)
        mu_b = self.prices[: self.k]
        mu_c = self.prices[self.k : self.k + self.m]
        mu_t = self.prices[-1]
        for s_idx in range(self.s):
            stability = float(np.sum(np.abs(self._last["compact"][s_idx] - self.prev_action_compact[s_idx])))
            qos_pen = max(0.0, (delays[s_idx] - self.d_max[s_idx]) / max(self.d_max[s_idx], 1e-9))
            local_pen = self.cfg.stab_coef * stability + self.cfg.qos_coef * qos_pen
            r = utilities[s_idx] - local_pen
            if self.use_prices:
                r -= float(np.dot(mu_b, b_used[s_idx]) + np.dot(mu_c, c_used[s_idx]) + mu_t * tau_used[s_idx])
            rewards[s_idx] = r

        self._buf.append({"obs": self._last["obs"], "act": self._last["act_vec"], "logp": self._last["logp"], "rew": rewards})
        self.prev_action_compact = self._last["compact"].copy()

        if self.use_prices:
            ex_b = metrics["rate_utilization"].astype(float) - 1.0
            ex_c = metrics["compute_utilization"].astype(float) - 1.0
            ex_t = float(metrics["transport_utilization"] - 1.0)
            new_b = self.prices[: self.k] + self.cfg.price_eta * ex_b
            new_c = self.prices[self.k : self.k + self.m] + self.cfg.price_eta * ex_c
            new_t = self.prices[-1] + self.cfg.price_eta * ex_t
            self.prices[: self.k] = np.clip(
                (1.0 - self.cfg.price_beta) * self.prices[: self.k] + self.cfg.price_beta * new_b,
                0.0,
                self.cfg.price_max,
            )
            self.prices[self.k : self.k + self.m] = np.clip(
                (1.0 - self.cfg.price_beta) * self.prices[self.k : self.k + self.m] + self.cfg.price_beta * new_c,
                0.0,
                self.cfg.price_max,
            )
            self.prices[-1] = float(
                np.clip((1.0 - self.cfg.price_beta) * self.prices[-1] + self.cfg.price_beta * new_t, 0.0, self.cfg.price_max)
            )

        if len(self._buf) >= self.cfg.update_every:
            self._update_ppo()
            self._buf.clear()

    def _update_ppo(self) -> None:
        n = len(self._buf)
        joint_obs = []
        joint_act = []
        joint_rew = []
        for item in self._buf:
            joint_obs.append(np.concatenate(item["obs"]))
            joint_act.append(np.concatenate(item["act"]))
            joint_rew.append(np.mean(item["rew"]))
        obs_t = torch.tensor(np.array(joint_obs), dtype=torch.float32)
        act_t = torch.tensor(np.array(joint_act), dtype=torch.float32)
        rew = np.array(joint_rew, dtype=np.float32)

        with torch.no_grad():
            val = self.critic(torch.cat([obs_t, act_t], dim=1)).squeeze(1).numpy()
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        nxt = 0.0
        for t in reversed(range(n)):
            delta = rew[t] + self.cfg.gamma * nxt - val[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * gae
            adv[t] = gae
            nxt = val[t]
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_t = torch.tensor(adv, dtype=torch.float32)
        ret_t = torch.tensor(ret, dtype=torch.float32)

        old_logp_agents = []
        obs_agents = []
        act_agents = []
        for s_idx in range(self.s):
            o = torch.tensor(np.array([it["obs"][s_idx] for it in self._buf]), dtype=torch.float32)
            a = torch.tensor(np.array([it["act"][s_idx] for it in self._buf]), dtype=torch.float32)
            lp = torch.tensor(np.array([it["logp"][s_idx] for it in self._buf]), dtype=torch.float32)
            obs_agents.append(o)
            act_agents.append(a)
            old_logp_agents.append(lp)

        idx = np.arange(n)
        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idx)
            for st in range(0, n, self.cfg.minibatch):
                mb = idx[st : st + self.cfg.minibatch]
                pred_v = self.critic(torch.cat([obs_t[mb], act_t[mb]], dim=1)).squeeze(1)
                l_v = ((pred_v - ret_t[mb]) ** 2).mean()
                self.opt_critic.zero_grad()
                l_v.backward()
                self.opt_critic.step()

                for s_idx in range(self.s):
                    mean, std = self.actors[s_idx](obs_agents[s_idx][mb])
                    dist = torch.distributions.Normal(mean, std)
                    new_logp = dist.log_prob(act_agents[s_idx][mb]).sum(dim=1)
                    ratio = torch.exp(new_logp - old_logp_agents[s_idx][mb])
                    surr1 = ratio * adv_t[mb]
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv_t[mb]
                    ent = dist.entropy().sum(dim=1).mean()
                    l_pi = -(torch.min(surr1, surr2).mean() + self.cfg.ent_coef * ent)
                    self.opt_actors[s_idx].zero_grad()
                    l_pi.backward()
                    self.opt_actors[s_idx].step()


class MAANPPOAllocator(_BasePPO):
    def __init__(
        self,
        s: int,
        k: int,
        m: int,
        b_k: np.ndarray,
        c_m: np.ndarray,
        t_agg: float,
        cfg: PPOConfig | None = None,
        r_min: np.ndarray | None = None,
        d_max: np.ndarray | None = None,
    ):
        super().__init__("MAAN_PPO", s, k, m, b_k, c_m, t_agg, use_prices=True, cfg=cfg or PPOConfig(), r_min=r_min, d_max=d_max)


class IndependentMAPPOPPOAllocator(_BasePPO):
    def __init__(
        self,
        s: int,
        k: int,
        m: int,
        b_k: np.ndarray,
        c_m: np.ndarray,
        t_agg: float,
        cfg: PPOConfig | None = None,
        r_min: np.ndarray | None = None,
        d_max: np.ndarray | None = None,
    ):
        super().__init__(
            "Independent_MAPPO_PPO",
            s,
            k,
            m,
            b_k,
            c_m,
            t_agg,
            use_prices=False,
            cfg=cfg or PPOConfig(),
            r_min=r_min,
            d_max=d_max,
        )
