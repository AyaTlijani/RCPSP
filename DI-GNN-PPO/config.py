"""
config.py
=========
Central configuration for DI-GNN-PPO. The config is saved with every checkpoint
so evaluation rebuilds the exact feature set and architecture used in training.

Node features (27 with the defaults, K = 4 renewable resources):
    base dynamic/static state        9
    resource demand                  K + 3
    criticality (CPM)                3
    structural (GRPW, downstream)    4
    resource pressure (free ratios)  K

The PPO objective is undiscounted: the reward is elapsed scheduling time, so the
episode return is exactly -makespan / reward_scale.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_DIR.parent / "data"
DEFAULT_RUNS_ROOT = PROJECT_DIR / "runs"

PROBLEM_SIZES = ("j30", "j60", "j90")


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    seed: int = 1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    num_resources: int = 4

    # --------------------------------------------------------------- features
    use_demand_ratio: bool = True
    use_criticality: bool = True
    use_structural: bool = True
    use_resource_pressure: bool = True

    # GRPW variant used by the GRPW baseline: "immediate" or "all".
    grpw_variant: str = "immediate"

    # Propagate precedence information both upstream and downstream.
    bidirectional_edges: bool = True

    # ----------------------------------------------------------------- model
    gin_layers: int = 3          # stacked multi-hop (0/1/2/3-hop) graph blocks
    hidden_dim: int = 96
    actor_hidden_dim: int = 192
    critic_hidden_dim: int = 192

    # ------------------------------------------------------------------- PPO
    learning_rate: float = 1e-4
    gamma: float = 1.0           # makespan objective is undiscounted
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 10.0
    ppo_epochs: int = 4
    # One rollout holds ~32 x (#activities) transitions (~960 for j30). A small
    # minibatch gives enough gradient steps per rollout (256 gave only 4 per epoch).
    minibatch_size: int = 64
    target_kl: float = 0.03

    # -------------------------------------------------------------- training
    num_episodes: int = 8000
    episodes_per_update: int = 32
    val_every_updates: int = 5

    # ------------------------------------------------------------ properties
    def node_feature_dim(self) -> int:
        k = self.num_resources
        dim = 9
        if self.use_demand_ratio:
            dim += k + 3
        if self.use_criticality:
            dim += 3
        if self.use_structural:
            dim += 4
        if self.use_resource_pressure:
            dim += k
        return dim

    def num_updates(self) -> int:
        return self.num_episodes // self.episodes_per_update

    def validate(self) -> None:
        if self.grpw_variant not in ("immediate", "all"):
            raise ValueError(f"grpw_variant must be 'immediate' or 'all', got {self.grpw_variant!r}")
        if not (0 < self.train_ratio < 1 and 0 < self.val_ratio < 1
                and self.train_ratio + self.val_ratio < 1):
            raise ValueError("train/val ratios must be in (0,1) and leave room for a test split")
        if self.episodes_per_update < 1 or self.num_episodes < self.episodes_per_update:
            raise ValueError("num_episodes must be >= episodes_per_update >= 1")
        if self.num_episodes % self.episodes_per_update != 0:
            raise ValueError("num_episodes must be divisible by episodes_per_update")
        if self.num_resources < 1:
            raise ValueError("num_resources must be >= 1")
        if self.gin_layers < 1 or self.hidden_dim < 1:
            raise ValueError("gin_layers and hidden_dim must be >= 1")
        if self.actor_hidden_dim < 1 or self.critic_hidden_dim < 1:
            raise ValueError("actor/critic hidden dimensions must be >= 1")
        if self.ppo_epochs < 1 or self.minibatch_size < 1:
            raise ValueError("ppo_epochs and minibatch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not (0 < self.clip_epsilon < 1):
            raise ValueError("clip_epsilon must be in (0, 1)")
        if self.entropy_coef < 0 or self.value_coef < 0:
            raise ValueError("entropy_coef and value_coef must be >= 0")
        if self.max_grad_norm <= 0 or self.reward_scale <= 0 or self.target_kl <= 0:
            raise ValueError("max_grad_norm, reward_scale and target_kl must be > 0")
        if not np.isclose(self.gamma, 1.0):
            raise ValueError("DI-GNN-PPO uses an undiscounted makespan objective; gamma must be 1.0")
        if not (0 < self.gae_lambda <= 1):
            raise ValueError("gae_lambda must be in (0, 1]")

    # --------------------------------------------------------- (de)serialise
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        valid = {f.name for f in dataclasses.fields(cls)}
        cfg = cls(**{k: v for k, v in d.items() if k in valid})
        cfg.validate()
        return cfg