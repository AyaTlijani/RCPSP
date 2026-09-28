from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ============================================================
# ACTIVITY SCORER
# ============================================================

class ActivityScorer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        layers: int = 3,
    ):
        super().__init__()

        modules = []

        current_dim = input_dim

        for _ in range(layers):

            modules.append(
                nn.Linear(
                    current_dim,
                    hidden_dim
                )
            )

            modules.append(
                nn.ReLU()
            )

            current_dim = hidden_dim

        modules.append(
            nn.Linear(
                current_dim,
                1
            )
        )

        self.network = nn.Sequential(
            *modules
        )

    def forward(self, x):

        return self.network(x).squeeze(-1)


# ============================================================
# CRITIC
# ============================================================

class Critic(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        layers: int = 3,
    ):
        super().__init__()

        modules = []

        current_dim = input_dim

        for _ in range(layers):

            modules.append(
                nn.Linear(
                    current_dim,
                    hidden_dim
                )
            )

            modules.append(
                nn.ReLU()
            )

            current_dim = hidden_dim

        modules.append(
            nn.Linear(
                current_dim,
                1
            )
        )

        self.network = nn.Sequential(
            *modules
        )

    def forward(self, x):

        return self.network(x).squeeze(-1)


# ============================================================
# ACTOR-CRITIC
# ============================================================

class MLPActorCritic(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        global_dim: int,
        hidden_dim: int = 128,
        layers: int = 3,
    ):
        super().__init__()

        self.actor = ActivityScorer(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            layers=layers,
        )

        self.critic = Critic(
            input_dim=global_dim,
            hidden_dim=hidden_dim,
            layers=layers,
        )

    def action_distribution(
        self,
        candidate_features: torch.Tensor,
    ):

        scores = self.actor(
            candidate_features
        )

        return Categorical(
            logits=scores
        )

    def value(
        self,
        global_features: torch.Tensor,
    ):

        return self.critic(
            global_features
        )


# ============================================================
# TRANSITION
# ============================================================

@dataclass
class Transition:

    features: np.ndarray

    global_features: np.ndarray

    action_index: int

    log_prob: float

    value: float

    reward: float

    done: bool


# ============================================================
# PPO AGENT
# ============================================================

class RCPSPMLPPPOAgent:

    def __init__(
        self,
        feature_dim: int,
        global_dim: int,
        hidden_dim: int = 128,
        layers: int = 3,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.025,
        value_coef: float = 0.5,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        device=None,
    ):

        if device is None:

            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.model = MLPActorCritic(
            feature_dim=feature_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            layers=layers,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

    # ========================================================
    # ACTION
    # ========================================================

    def select_action(
        self,
        candidate_features: np.ndarray,
        global_features: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:

        candidate_tensor = torch.as_tensor(
            candidate_features,
            dtype=torch.float32,
            device=self.device,
        )

        global_tensor = torch.as_tensor(
            global_features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            distribution = (
                self.model.action_distribution(
                    candidate_tensor
                )
            )

            value = self.model.value(
                global_tensor
            ).item()

            if deterministic:

                action = torch.argmax(
                    distribution.logits
                )

            else:

                action = distribution.sample()

            log_prob = distribution.log_prob(
                action
            ).item()

        return (
            int(action.item()),
            float(log_prob),
            float(value),
        )

    # ========================================================
    # PPO UPDATE
    # ========================================================

    def update(
        self,
        transitions: List[Transition],
        update_epochs: int = 2,
    ):

        if not transitions:
            return

        rewards = np.asarray(
            [
                transition.reward
                for transition in transitions
            ],
            dtype=np.float32,
        )

        old_values = np.asarray(
            [
                transition.value
                for transition in transitions
            ],
            dtype=np.float32,
        )

        dones = np.asarray(
            [
                transition.done
                for transition in transitions
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # GAE
        # ----------------------------------------------------

        advantages = np.zeros_like(
            rewards,
            dtype=np.float32,
        )

        last_advantage = 0.0

        for t in reversed(
            range(len(transitions))
        ):

            if t == len(transitions) - 1:

                next_value = 0.0

            else:

                next_value = old_values[t + 1]

            non_terminal = 1.0 - dones[t]

            delta = (
                rewards[t]
                + self.gamma
                * next_value
                * non_terminal
                - old_values[t]
            )

            last_advantage = (
                delta
                + self.gamma
                * self.gae_lambda
                * non_terminal
                * last_advantage
            )

            advantages[t] = last_advantage

        returns = (
            advantages
            + old_values
        )

        # Normalize advantages.
        if len(advantages) > 1:

            advantages = (
                advantages
                - advantages.mean()
            ) / (
                advantages.std() + 1e-8
            )

        # ----------------------------------------------------
        # PPO epochs
        # ----------------------------------------------------

        for _ in range(update_epochs):

            total_loss = 0.0

            for index, transition in enumerate(
                transitions
            ):

                features = torch.as_tensor(
                    transition.features,
                    dtype=torch.float32,
                    device=self.device,
                )

                global_features = torch.as_tensor(
                    transition.global_features,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                distribution = (
                    self.model.action_distribution(
                        features
                    )
                )

                action = torch.tensor(
                    transition.action_index,
                    dtype=torch.long,
                    device=self.device,
                )

                new_log_prob = (
                    distribution.log_prob(
                        action
                    )
                )

                entropy = (
                    distribution.entropy()
                )

                value = self.model.value(
                    global_features
                ).squeeze()

                old_log_prob = torch.tensor(
                    transition.log_prob,
                    dtype=torch.float32,
                    device=self.device,
                )

                advantage = torch.tensor(
                    advantages[index],
                    dtype=torch.float32,
                    device=self.device,
                )

                target_return = torch.tensor(
                    returns[index],
                    dtype=torch.float32,
                    device=self.device,
                )

                ratio = torch.exp(
                    new_log_prob
                    - old_log_prob
                )

                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                )

                policy_loss = -torch.min(
                    ratio * advantage,
                    clipped_ratio * advantage,
                )

                value_loss = (
                    value - target_return
                ).pow(2)

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                total_loss = (
                    total_loss
                    + loss
                )

            total_loss = (
                total_loss
                / len(transitions)
            )

            self.optimizer.zero_grad()

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

            self.optimizer.step()

    # ========================================================
    # SAVE
    # ========================================================

    def save(self, path: str):

        checkpoint = {
            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),
        }

        torch.save(
            checkpoint,
            path,
        )

    # ========================================================
    # LOAD
    # ========================================================

    def load(self, path: str):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if "optimizer_state_dict" in checkpoint:

            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )