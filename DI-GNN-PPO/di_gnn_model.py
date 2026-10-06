"""
di_gnn_model.py
===============
Multi-hop (MixHop-style) GNN actor-critic for DI-GNN-PPO.

    node features -> multi-hop graph blocks -> node embeddings
    actor : score each node from [node, project, candidate-set context, resources]
    critic: value from       [project, candidate-set context, resources]

The actor is candidate-set aware: each startable activity is scored relative to
the other currently startable activities and the whole project state.

No dropout is used: PPO needs the log-probability computed at rollout time and
at update time to be identical for identical weights.

No torch-geometric dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

MASK_VALUE = -1e9


# ==========================================================================
# GRAPH BATCH
# ==========================================================================
@dataclass
class GraphBatch:
    """Variable-size RCPSP graphs stored as one disconnected graph."""
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor      # graph id of every node
    pos: torch.Tensor        # node index inside its own graph
    mask: torch.Tensor       # startable activities
    free: torch.Tensor       # (G, K) free-resource ratios
    sizes: List[int]
    num_graphs: int


def collate_observations(obs_list: Sequence[dict], device: torch.device) -> GraphBatch:
    """Collate environment observations (x, edge_index, mask, free_ratio, num_nodes)."""
    if not obs_list:
        raise ValueError("obs_list must not be empty")

    xs, edges, masks, frees, batch_ids, positions, sizes = [], [], [], [], [], [], []
    offset = 0

    for g, obs in enumerate(obs_list):
        n = int(obs["num_nodes"])
        x = np.asarray(obs["x"], dtype=np.float32)
        ei = np.asarray(obs["edge_index"], dtype=np.int64)
        mask = np.asarray(obs["mask"], dtype=bool)
        free = np.asarray(obs["free_ratio"], dtype=np.float32)

        if n <= 0 or x.ndim != 2 or x.shape[0] != n:
            raise ValueError(f"graph {g}: bad node features {x.shape} for num_nodes={n}")
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError(f"graph {g}: edge_index must be (2, E), got {ei.shape}")
        if mask.shape != (n,):
            raise ValueError(f"graph {g}: mask shape {mask.shape} != {(n,)}")
        if not mask.any():
            raise ValueError(f"graph {g} has no valid action")
        if not np.isfinite(x).all() or not np.isfinite(free).all():
            raise ValueError(f"graph {g}: non-finite values")
        if ei.shape[1] > 0:
            if ei.min() < 0 or ei.max() >= n:
                raise ValueError(f"graph {g}: edge index outside graph")
            edges.append(ei + offset)

        xs.append(x)
        masks.append(mask)
        frees.append(free)
        batch_ids.append(np.full(n, g, dtype=np.int64))
        positions.append(np.arange(n, dtype=np.int64))
        sizes.append(n)
        offset += n

    edge_all = np.concatenate(edges, axis=1) if edges else np.zeros((2, 0), dtype=np.int64)

    def tensor(a: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)

    return GraphBatch(
        x=tensor(np.concatenate(xs), torch.float32),
        edge_index=tensor(edge_all, torch.long),
        batch=tensor(np.concatenate(batch_ids), torch.long),
        pos=tensor(np.concatenate(positions), torch.long),
        mask=tensor(np.concatenate(masks), torch.bool),
        free=tensor(np.stack(frees), torch.float32),
        sizes=sizes,
        num_graphs=len(sizes),
    )


# ==========================================================================
# POOLING
# ==========================================================================
def global_mean_pool(h: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    summed = h.new_zeros(num_graphs, h.shape[1]).index_add(0, batch, h)
    counts = h.new_zeros(num_graphs).index_add(0, batch, h.new_ones(h.shape[0]))
    return summed / counts.clamp(min=1.0).unsqueeze(1)


def masked_mean_pool(h: torch.Tensor, mask: torch.Tensor, batch: torch.Tensor,
                     num_graphs: int) -> torch.Tensor:
    w = mask.to(h.dtype).unsqueeze(1)
    summed = h.new_zeros(num_graphs, h.shape[1]).index_add(0, batch, h * w)
    counts = h.new_zeros(num_graphs).index_add(0, batch, w.squeeze(1))
    return summed / counts.clamp(min=1.0).unsqueeze(1)


def masked_max_pool(h: torch.Tensor, mask: torch.Tensor, batch: torch.Tensor,
                    num_graphs: int) -> torch.Tensor:
    result = h.new_full((num_graphs, h.shape[1]), -torch.inf)
    cand_h, cand_batch = h[mask], batch[mask]
    if cand_h.numel() == 0:
        return h.new_zeros(num_graphs, h.shape[1])
    for g in range(num_graphs):
        sel = cand_h[cand_batch == g]
        if sel.numel() > 0:
            result[g] = sel.max(dim=0).values
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


# ==========================================================================
# GRAPH MESSAGE PASSING
# ==========================================================================
def mean_message_passing(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean over incoming neighbours. edge_index[0]=source, edge_index[1]=destination."""
    if edge_index.shape[1] == 0:
        return torch.zeros_like(h)
    src, dst = edge_index[0], edge_index[1]
    agg = h.new_zeros(h.shape).index_add(0, dst, h[src])
    deg = h.new_zeros(h.shape[0]).index_add(0, dst, h.new_ones(src.shape[0]))
    return agg / deg.clamp(min=1.0).unsqueeze(1)


class MultiHopGraphBlock(nn.Module):
    """
    MixHop-style block: 0/1/2/3-hop representations are built explicitly, each gets
    its own projection, and all scales are fused together (not just stacked).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(4)])
        self.fusion = nn.Sequential(
            nn.Linear(4 * out_dim, 2 * out_dim),
            nn.GELU(),
            nn.Linear(2 * out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.residual = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hops = [h]
        for _ in range(3):
            hops.append(mean_message_passing(hops[-1], edge_index))
        multi_hop = torch.cat([proj(x) for proj, x in zip(self.projs, hops)], dim=1)
        return F.gelu(self.norm(self.fusion(multi_hop) + self.residual(h)))


class ResourceEncoder(nn.Module):
    """Encode the global free-capacity ratios."""

    def __init__(self, num_resources: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_resources, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, free: torch.Tensor) -> torch.Tensor:
        return self.net(free)


def _head(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, 1),
    )


# ==========================================================================
# ACTOR-CRITIC
# ==========================================================================
class DIGNNActorCritic(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        self.gin_layers = nn.ModuleList([
            MultiHopGraphBlock(cfg.node_feature_dim() if i == 0 else d, d)
            for i in range(cfg.gin_layers)
        ])
        self.resource_encoder = ResourceEncoder(cfg.num_resources, d)

        # candidate context = [mean, max] of startable-node embeddings -> 2d
        self.actor = _head(d + d + 2 * d + d, cfg.actor_hidden_dim)
        self.critic = _head(d + 2 * d + d, cfg.critic_hidden_dim)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _forward(self, gb: GraphBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns log_probs [num_graphs, max_nodes] and values [num_graphs]."""
        if gb.x.ndim != 2 or gb.x.shape[1] != self.cfg.node_feature_dim():
            raise ValueError(
                f"expected (N, {self.cfg.node_feature_dim()}) node features, got {tuple(gb.x.shape)}")

        h = gb.x
        for layer in self.gin_layers:
            h = layer(h, gb.edge_index)

        global_emb = global_mean_pool(h, gb.batch, gb.num_graphs)
        cand_ctx = torch.cat([
            masked_mean_pool(h, gb.mask, gb.batch, gb.num_graphs),
            masked_max_pool(h, gb.mask, gb.batch, gb.num_graphs),
        ], dim=1)
        res_emb = self.resource_encoder(gb.free)

        actor_in = torch.cat(
            [h, global_emb[gb.batch], cand_ctx[gb.batch], res_emb[gb.batch]], dim=1)
        logits = self.actor(actor_in).squeeze(-1).masked_fill(~gb.mask, MASK_VALUE)

        dense = logits.new_full((gb.num_graphs, max(gb.sizes)), MASK_VALUE)
        dense[gb.batch, gb.pos] = logits
        log_probs = F.log_softmax(dense, dim=1)

        values = self.critic(torch.cat([global_emb, cand_ctx, res_emb], dim=1)).squeeze(-1)
        return log_probs, values

    @torch.no_grad()
    def act(self, obs: dict, greedy: bool = False) -> Tuple[int, float, float]:
        """Select one action. Returns (action, log_prob, value)."""
        gb = collate_observations([obs], self.device)
        log_probs, values = self._forward(gb)
        lp = log_probs[0]

        if greedy:
            action = int(torch.argmax(lp).item())
        else:
            action = int(torch.multinomial(lp.exp(), num_samples=1).item())

        if not bool(np.asarray(obs["mask"], dtype=bool)[action]):
            raise RuntimeError("policy selected a non-startable activity")
        return action, float(lp[action].item()), float(values[0].item())

    def evaluate_actions(self, gb: GraphBatch, actions: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (log_prob of actions, entropy, values) for a PPO minibatch."""
        actions = actions.reshape(-1)
        if actions.shape[0] != gb.num_graphs:
            raise ValueError(f"expected {gb.num_graphs} actions, got {actions.shape[0]}")

        log_probs, values = self._forward(gb)
        selected = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=1)
        return selected, entropy, values