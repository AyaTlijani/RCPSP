"""
ppo.py
======
PPO for DI-GNN-PPO.

The objective is undiscounted makespan minimisation (gamma = 1): for a complete
episode  return = -makespan / reward_scale.

    - GAE(lambda), episode boundaries stop the recursion
    - advantages normalised over the rollout
    - clipped policy objective
    - plain (unclipped) value loss, divided by the return std so its gradient
      stays O(1) and does not dominate the shared gradient-norm clip
    - entropy bonus, gradient clipping, target-KL early stopping
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from config import Config
from di_gnn_model import DIGNNActorCritic, collate_observations


class RolloutBuffer:
    """Complete on-policy episodes (one or more) stored transition by transition."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.obs: List[dict] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def add(self, obs: dict, action: int, log_prob: float, value: float,
            reward: float, done: bool) -> None:
        self.obs.append(obs)
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def __len__(self) -> int:
        return len(self.actions)

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float
                                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GAE:  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
              A_t     = delta_t + gamma * lambda * A_{t+1}
        Terminal transitions bootstrap with 0 and cut the recursion.
        """
        T = len(self)
        if T == 0:
            raise ValueError("cannot compute advantages from an empty buffer")
        if not self.dones[-1]:
            raise ValueError("buffer must end at a completed episode")

        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        advantages = np.zeros(T, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(T)):
            non_terminal = 0.0 if self.dones[t] else 1.0
            next_value = values[t + 1] if t < T - 1 else 0.0
            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[t] = gae

        return advantages, advantages + values


class PPO:
    def __init__(self, model: DIGNNActorCritic, cfg: Config, device: torch.device) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    def update(self, buffer: RolloutBuffer, rng: np.random.Generator) -> Dict[str, float]:
        cfg, dev = self.cfg, self.device
        T = len(buffer)
        if T == 0:
            raise ValueError("cannot update PPO with an empty buffer")

        adv_np, ret_np = buffer.compute_returns_and_advantages(cfg.gamma, cfg.gae_lambda)
        value_scale = max(float(ret_np.std()), 1.0)
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        actions = torch.as_tensor(buffer.actions, dtype=torch.long, device=dev)
        old_logp = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=dev)
        advantages = torch.as_tensor(adv_np, dtype=torch.float32, device=dev)
        returns = torch.as_tensor(ret_np, dtype=torch.float32, device=dev)

        logs: Dict[str, List[float]] = {k: [] for k in (
            "policy_loss", "value_loss", "entropy", "approx_kl",
            "clip_frac", "explained_variance", "grad_norm")}

        for _ in range(cfg.ppo_epochs):
            perm = rng.permutation(T)
            epoch_kl: List[float] = []

            for start in range(0, T, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=dev)
                gb = collate_observations([buffer.obs[i] for i in idx], dev)

                new_logp, entropy, values = self.model.evaluate_actions(gb, actions[idx_t])

                log_ratio = new_logp - old_logp[idx_t]
                ratio = log_ratio.exp()
                adv = advantages[idx_t]
                target = returns[idx_t]

                policy_loss = -torch.min(
                    ratio * adv,
                    ratio.clamp(1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * adv,
                ).mean()
                value_loss = 0.5 * ((values - target) / value_scale).pow(2).mean()
                entropy_mean = entropy.mean()

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl = float(((ratio - 1.0) - log_ratio).mean().clamp(min=0.0).item())
                    target_var = target.var(unbiased=False)
                    ev = (1.0 - (target - values).var(unbiased=False) / target_var
                          if target_var > 1e-8 else torch.zeros((), device=dev))

                    logs["policy_loss"].append(float(policy_loss.item()))
                    logs["value_loss"].append(float(value_loss.item()))
                    logs["entropy"].append(float(entropy_mean.item()))
                    logs["approx_kl"].append(kl)
                    logs["clip_frac"].append(
                        float(((ratio - 1.0).abs() > cfg.clip_epsilon).float().mean().item()))
                    logs["explained_variance"].append(float(ev.item()))
                    logs["grad_norm"].append(float(grad_norm.item()))
                    epoch_kl.append(kl)

            if epoch_kl and float(np.mean(epoch_kl)) > cfg.target_kl:
                break

        return {k: float(np.mean(v)) for k, v in logs.items()}