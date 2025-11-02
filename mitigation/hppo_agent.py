import argparse
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = ["rate_limit", "quarantine", "reroute", "drop"]


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int, act_dim: int):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, act_dim))
        self.v = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        logits = self.pi(x)
        v = self.v(x).squeeze(-1)
        return logits, v


@dataclass
class HPPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    steps: int = 64
    minibatches: int = 2
    epochs: int = 2


class HierarchicalPPO:
    def __init__(self, cfg: HPPOConfig, obs_dim: int, act_dim: int = len(ACTIONS)):
        self.cfg = cfg
        self.high = PolicyNet(obs_dim, 64, act_dim)
        self.low = PolicyNet(obs_dim + act_dim, 64, act_dim)  # conditioned on high-level action
        self.opt = torch.optim.Adam(list(self.high.parameters()) + list(self.low.parameters()), lr=cfg.lr)

    def _policy_loss(self, logits_old, logits_new, acts, adv):
        logp_old = F.log_softmax(logits_old, dim=-1).gather(1, acts.view(-1, 1)).squeeze(1)
        logp_new = F.log_softmax(logits_new, dim=-1).gather(1, acts.view(-1, 1)).squeeze(1)
        ratio = torch.exp(logp_new - logp_old)
        clip = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv
        loss = -(torch.min(ratio * adv, clip)).mean()
        ent = (-(F.softmax(logits_new, -1) * F.log_softmax(logits_new, -1)).sum(-1)).mean()
        return loss - self.cfg.entropy_coef * ent

    def _value_loss(self, v, ret):
        return F.mse_loss(v, ret)

    def step_env(self, env_fn, steps: int) -> Tuple[dict, dict]:
        # env_fn returns (obs, reward, done, info) given action
        obs = env_fn(None)  # get initial obs
        buf = {k: [] for k in ["obs", "acts_h", "acts_l", "logits_h", "logits_l", "vals_h", "vals_l", "rews", "done"]}
        for _ in range(steps):
            x = torch.tensor(obs, dtype=torch.float32)
            logits_h, v_h = self.high(x)
            a_h = torch.distributions.Categorical(logits=logits_h).sample().item()
            x_low = torch.cat([x, F.one_hot(torch.tensor(a_h), num_classes=len(ACTIONS)).float()])
            logits_l, v_l = self.low(x_low)
            a_l = torch.distributions.Categorical(logits=logits_l).sample().item()
            obs2, r, done, _ = env_fn((a_h, a_l))
            # store
            buf["obs"].append(obs)
            buf["acts_h"].append(a_h)
            buf["acts_l"].append(a_l)
            buf["logits_h"].append(logits_h.detach().numpy())
            buf["logits_l"].append(logits_l.detach().numpy())
            buf["vals_h"].append(v_h.item())
            buf["vals_l"].append(v_l.item())
            buf["rews"].append(r)
            buf["done"].append(done)
            obs = obs2
            if done:
                obs = env_fn(None)
        return buf, {"steps": steps}

    def update(self, buf: dict):
        obs = torch.tensor(np.array(buf["obs"]), dtype=torch.float32)
        acts_h = torch.tensor(buf["acts_h"], dtype=torch.long)
        acts_l = torch.tensor(buf["acts_l"], dtype=torch.long)
        logits_h_old = torch.tensor(np.array(buf["logits_h"]), dtype=torch.float32)
        logits_l_old = torch.tensor(np.array(buf["logits_l"]), dtype=torch.float32)
        vals_h = torch.tensor(buf["vals_h"], dtype=torch.float32)
        vals_l = torch.tensor(buf["vals_l"], dtype=torch.float32)
        rews = torch.tensor(buf["rews"], dtype=torch.float32)
        done = torch.tensor(buf["done"], dtype=torch.float32)

        # GAE-Lambda for advantages
        def compute_adv(vals):
            adv = torch.zeros_like(rews)
            lastgaelam = 0
            for t in reversed(range(len(rews))):
                nextnonterminal = 1 - done[t]
                delta = rews[t] + self.cfg.gamma * nextnonterminal * (vals[t + 1] if t + 1 < len(vals) else 0) - vals[t]
                lastgaelam = delta + self.cfg.gamma * self.cfg.lam * nextnonterminal * lastgaelam
                adv[t] = lastgaelam
            ret = adv + vals
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            return adv, ret

        adv_h, ret_h = compute_adv(vals_h)
        adv_l, ret_l = compute_adv(vals_l)

        for _ in range(self.cfg.epochs):
            # High-level update
            logits_h_new, v_h_new = self.high(obs)
            loss_pi_h = self._policy_loss(logits_h_old, logits_h_new, acts_h, adv_h)
            loss_v_h = self._value_loss(v_h_new, ret_h)
            # Low-level update
            onehot_h = F.one_hot(acts_h, num_classes=len(ACTIONS)).float()
            obs_l = torch.cat([obs, onehot_h], dim=-1)
            logits_l_new, v_l_new = self.low(obs_l)
            loss_pi_l = self._policy_loss(logits_l_old, logits_l_new, acts_l, adv_l)
            loss_v_l = self._value_loss(v_l_new, ret_l)

            loss = loss_pi_h + loss_pi_l + self.cfg.value_coef * (loss_v_h + loss_v_l)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.high.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.low.parameters(), 0.5)
            self.opt.step()
        return {"loss": float(loss.item())}


def demo_env(action: Tuple[int, int] | None):
    # Observation: [flow_bytes, packets, duration, anomaly_score]
    rng = np.random.default_rng(0)
    if action is None:
        return np.array([100.0, 10.0, 1.0, 0.1], dtype=np.float32)
    a_h, a_l = action
    # reward: encourage drop/quarantine for high anomaly, penalize reroute overhead
    base_obs = np.array([rng.normal(100, 10), rng.integers(1, 50), rng.random(), rng.random()])
    reward = 0.1
    if a_h in [1, 3]:
        reward += 0.5
    if a_h == 2:
        reward -= 0.05
    done = bool(rng.random() < 0.05)
    return base_obs.astype(np.float32), float(reward), done, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/hppo.yaml")
    ap.add_argument("--rollout_steps", type=int, default=64)
    args = ap.parse_args()

    import yaml
    from utils.common import load_yaml

    cfg = load_yaml(args.config)
    hppo = HierarchicalPPO(HPPOConfig(**cfg), obs_dim=4)
    buf, info = hppo.step_env(demo_env, steps=args.rollout_steps)
    out = hppo.update(buf)
    print("HPPO updated:", out)


if __name__ == "__main__":
    main()
