import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from gnn import GINEncoder


class RCPSPPPOAgent(nn.Module):
    """
    PPO agent for RCPSP.

    The agent:
    - uses GINEncoder from gnn.py
    - scores only currently available activities
    - uses free resources as part of the decision
    - uses the global graph embedding for project-level context
    - has a separate critic for state-value estimation

    Works with j30, j60 and j90.
    """

    def __init__(
        self,
        node_feature_dim: int,
        resource_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()

        # =========================
        # GNN
        # =========================

        self.gnn = GINEncoder(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
        )

        # =========================
        # Actor
        # =========================
        #
        # For each available activity:
        #
        # activity embedding
        # + free resources
        # + global graph embedding
        #
        # -> action score
        #

        actor_input_dim = hidden_dim + resource_dim + hidden_dim

        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # =========================
        # Critic
        # =========================
        #
        # State representation:
        #
        # global graph embedding
        # + free resources
        #
        # -> state value
        #

        critic_input_dim = hidden_dim + resource_dim

        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    # ============================================================
    # GNN forward
    # ============================================================

    def forward(self, graph):
        h_j_K, h_G = self.gnn(
            graph.x,
            graph.edge_index,
        )

        return h_j_K, h_G

    # ============================================================
    # Candidate action scoring
    # ============================================================

    def _candidate_scores(
        self,
        graph,
        r_free,
        available_actions,
    ):
        """
        Compute logits for the currently available activities.

        Only activities in available_actions are considered.
        """

        h_j_K, h_G = self.forward(graph)

        # Convert graph node indices -> activity IDs
        graph_indexes = [
            idx
            for idx, activity_id in graph.node_to_activity.items()
            if activity_id in available_actions
        ]

        activity_ids = [
            graph.node_to_activity[idx]
            for idx in graph_indexes
        ]

        if len(activity_ids) == 0:
            raise ValueError(
                "No graph nodes correspond to available actions."
            )

        # Embeddings of candidate activities
        action_embeddings = h_j_K[graph_indexes]

        n_actions = action_embeddings.size(0)

        # Repeat free-resource vector for every candidate
        r_free_expanded = r_free.unsqueeze(0).expand(
            n_actions, -1
        )

        # Repeat global project embedding
        h_G_expanded = h_G.unsqueeze(0).expand(
            n_actions, -1
        )

        # Actor input
        actor_input = torch.cat(
            [
                action_embeddings,
                r_free_expanded,
                h_G_expanded,
            ],
            dim=-1,
        )

        logits = self.actor(actor_input).squeeze(-1)

        return logits, activity_ids, h_G

    # ============================================================
    # Critic
    # ============================================================

    def get_value(
        self,
        graph,
        r_free,
    ):
        """
        Estimate V(s).
        """

        _, h_G = self.forward(graph)

        critic_input = torch.cat(
            [
                h_G,
                r_free,
            ],
            dim=-1,
        )

        return self.critic(critic_input)

    # ============================================================
    # Action selection
    # ============================================================

    def select_action(
        self,
        graph,
        r_free,
        available_actions,
        greedy=False,
    ):
        """
        Select an activity.

        Training:
            greedy=False -> sample from policy

        Evaluation:
            greedy=True -> choose highest-probability action

        Returns:
            action
            log_probability
            state_value
        """

        if not available_actions:
            raise ValueError(
                "No available actions."
            )

        logits, activity_ids, h_G = self._candidate_scores(
            graph,
            r_free,
            available_actions,
        )

        probs = F.softmax(logits, dim=0)

        distribution = Categorical(probs)

        if greedy:
            local_index = torch.argmax(probs)
        else:
            local_index = distribution.sample()

        action = activity_ids[local_index.item()]

        log_prob = distribution.log_prob(local_index)

        # Critic
        critic_input = torch.cat(
            [
                h_G,
                r_free,
            ],
            dim=-1,
        )

        value = self.critic(critic_input)

        return action, log_prob, value

    # ============================================================
    # PPO action evaluation
    # ============================================================

    def evaluate_action(
        self,
        graph,
        r_free,
        available_actions,
        action,
    ):
        """
        Re-evaluate an action using the current policy.

        Used during PPO updates to calculate:

            new log probability
            entropy
            state value
        """

        if not available_actions:
            raise ValueError(
                "No available actions."
            )

        logits, activity_ids, h_G = self._candidate_scores(
            graph,
            r_free,
            available_actions,
        )

        probs = F.softmax(logits, dim=0)

        distribution = Categorical(probs)

        if action not in activity_ids:
            raise ValueError(
                f"Action {action} is not in available actions."
            )

        local_index = activity_ids.index(action)

        local_index = torch.tensor(
            local_index,
            device=logits.device,
        )

        log_prob = distribution.log_prob(local_index)

        entropy = distribution.entropy()

        # Critic
        critic_input = torch.cat(
            [
                h_G,
                r_free,
            ],
            dim=-1,
        )

        value = self.critic(critic_input)

        return log_prob, entropy, value