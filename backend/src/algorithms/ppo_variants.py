from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.base import AlgorithmOutput, BaseAllocator


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


class _BasePPO(BaseAllocator):
    def __init__(self, name: str, s: int, k: int, m: int, b_k: np.ndarray, c_m: np.ndarray, t_agg: float, use_prices: bool, cfg: PPOConfig):
        super().__init__(name)
        self.s = s
        self.k = k
        self.m = m
        self.b_k = b_k
        self.c_m = c_m
        self.t_agg = t_agg
        self.use_prices = use_prices
        self.cfg = cfg
        self.act_dim = k + m + 1 + m
        self.obs_dim = 2 * s + s * k + (k + m + 1 if use_prices else 0)
        self.actors = [Actor(self.obs_dim, self.act_dim) for _ in range(s)]
        self.opt_actors = [optim.Adam(a.parameters(), lr=cfg.lr_actor) for a in self.actors]
        self.critic = Critic(self.obs_dim * s + self.act_dim * s)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self._buf: List[Dict] = []
        self._last = None
        self.prices = np.zeros(k + m + 1, dtype=float)

    def reset(self) -> None:
        self._buf = []
        self._last = None
        self.prices[:] = 0.0

    def _obs_per_agent(self, state: Dict[str, np.ndarray], s_idx: int) -> np.ndarray:
        lam = state["lambda"] / max(np.mean(state["lambda"]), 1.0)
        chan = state["channel"].reshape(-1)
        base = np.concatenate([lam, state["prev_tau"] / max(np.mean(state["prev_tau"]) + 1e-6, 1.0), chan])
        if self.use_prices:
            base = np.concatenate([base, self.prices])
        return base.astype(np.float32)

    def act(self, state: Dict[str, np.ndarray]) -> AlgorithmOutput:
        obs = [self._obs_per_agent(state, i) for i in range(self.s)]
        b = np.zeros((self.s, self.k), dtype=float)
        c = np.zeros((self.s, self.m), dtype=float)
        tau = np.zeros(self.s, dtype=float)
        x = np.zeros((self.s, self.m), dtype=int)
        info = {"obs": obs, "logp": [], "act_vec": []}
        for i in range(self.s):
            o = torch.tensor(obs[i]).unsqueeze(0)
            mean, std = self.actors[i](o)
            dist = torch.distributions.Normal(mean, std)
            z = dist.sample()
            logp = dist.log_prob(z).sum(dim=1)
            a = z.squeeze(0).detach().numpy()
            raw_b = np.maximum(0.0, a[: self.k])
            raw_c = np.maximum(0.0, a[self.k : self.k + self.m])
            raw_tau = max(0.0, float(a[self.k + self.m]))
            x_logits = a[self.k + self.m + 1 :]
            m_idx = int(np.argmax(x_logits))
            x[i, m_idx] = 1
            b[i] = raw_b + 1e-3
            c[i, m_idx] = raw_c[m_idx] + 1e-3
            tau[i] = raw_tau + 1e-3
            info["logp"].append(float(logp.item()))
            info["act_vec"].append(a)
        self._last = info
        return AlgorithmOutput(actions={"b": b, "c": c, "tau": tau, "x": x}, aux={})

    def observe(self, state: Dict[str, np.ndarray], metrics: Dict[str, np.ndarray]) -> None:
        reward = metrics["utilities"].astype(float)
        if self._last is None:
            return
        self._buf.append({"obs": self._last["obs"], "act": self._last["act_vec"], "logp": self._last["logp"], "rew": reward})
        if self.use_prices:
            ex_b = np.mean(metrics["rate_utilization"]) - 1.0
            ex_c = np.mean(metrics["compute_utilization"]) - 1.0
            ex_t = metrics["transport_utilization"] - 1.0
            self.prices = np.clip(self.prices + 0.002 * np.array([*([ex_b] * self.k), *([ex_c] * self.m), ex_t]), 0.0, 10.0)
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
                # Critic
                pred_v = self.critic(torch.cat([obs_t[mb], act_t[mb]], dim=1)).squeeze(1)
                l_v = ((pred_v - ret_t[mb]) ** 2).mean()
                self.opt_critic.zero_grad()
                l_v.backward()
                self.opt_critic.step()

                # Actors
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
    def __init__(self, s: int, k: int, m: int, b_k: np.ndarray, c_m: np.ndarray, t_agg: float, cfg: PPOConfig | None = None):
        super().__init__("MAAN_PPO", s, k, m, b_k, c_m, t_agg, use_prices=True, cfg=cfg or PPOConfig())


class IndependentMAPPOPPOAllocator(_BasePPO):
    def __init__(self, s: int, k: int, m: int, b_k: np.ndarray, c_m: np.ndarray, t_agg: float, cfg: PPOConfig | None = None):
        super().__init__("Independent_MAPPO_PPO", s, k, m, b_k, c_m, t_agg, use_prices=False, cfg=cfg or PPOConfig())
