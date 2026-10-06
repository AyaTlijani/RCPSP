"""
domain_features.py
==================

Static and dynamic node features for the DI-GNN-PPO RCPSP agent.

The representation is built for the single-mode PSPLIB RCPSP instances
(j30, j60, j90, j120) and uses:

    Base dynamic/static features        9
    Resource demand features            7
    Criticality features                3
    Structural features                 4
    Dynamic resource-pressure features  4
                                      ----
                                       27

All arrays in StaticFeatures refer to REAL activities only.
PSPLIB dummy source and sink activities are excluded from the node matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from config import Config
from psplib_loader import RCPSPInstance


_EPS = 1e-8


# ---------------------------------------------------------------------------
# Environment status constants
# ---------------------------------------------------------------------------

STATUS_NOT_STARTED = 0
STATUS_RUNNING = 1
STATUS_COMPLETED = 2


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

def _safe_div(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
) -> np.ndarray:
    return np.asarray(numerator, dtype=np.float32) / np.maximum(
        np.asarray(denominator, dtype=np.float32),
        _EPS,
    )


def _normalize(
    values: np.ndarray,
    scale: float | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)

    if scale is None:
        scale = float(np.max(values)) if values.size else 1.0

    scale = max(float(scale), 1.0)
    return values / scale


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def _topological_order(instance: RCPSPInstance) -> List[int]:
    """
    Return a topological ordering of all activities.

    The loader already validates the graph as a DAG, but computing the order
    here makes the feature code independent of the loader's internal ordering.
    """

    n = instance.num_activities

    indegree = np.zeros(n, dtype=np.int64)

    for node in range(n):
        indegree[node] = len(instance.predecessors[node])

    queue = [i for i in range(n) if indegree[i] == 0]
    order: List[int] = []

    head = 0

    while head < len(queue):
        node = queue[head]
        head += 1

        order.append(node)

        for successor in instance.successors[node]:
            indegree[successor] -= 1

            if indegree[successor] == 0:
                queue.append(successor)

    if len(order) != n:
        raise ValueError(
            f"Instance {instance.name} does not contain a valid DAG."
        )

    return order


def _successor_closure(
    instance: RCPSPInstance,
    node: int,
) -> List[int]:
    """
    Return all transitive successors of one activity.

    The returned node IDs are the original PSPLIB activity IDs.
    """

    visited = set()
    stack = list(instance.successors[node])

    while stack:
        current = stack.pop()

        if current in visited:
            continue

        visited.add(current)
        stack.extend(instance.successors[current])

    return list(visited)


# ---------------------------------------------------------------------------
# CPM features
# ---------------------------------------------------------------------------

def compute_cpm(instance: RCPSPInstance) -> Dict[str, np.ndarray | int]:
    """
    Compute deterministic CPM quantities for the instance.

    ES = earliest start
    EF = earliest finish
    LS = latest start
    LF = latest finish
    slack = LS - ES

    The dummy source/sink are included in the computation.
    """

    n = instance.num_activities
    durations = np.asarray(instance.durations, dtype=np.float64)

    order = _topological_order(instance)

    es = np.zeros(n, dtype=np.float64)
    ef = np.zeros(n, dtype=np.float64)

    # Forward pass.
    for node in order:
        predecessors = instance.predecessors[node]

        if predecessors:
            es[node] = max(ef[p] for p in predecessors)

        ef[node] = es[node] + durations[node]

    sink = n - 1
    horizon = int(round(ef[sink]))

    # Backward pass.
    lf = np.full(n, float(horizon), dtype=np.float64)
    ls = np.full(n, float(horizon), dtype=np.float64)

    reverse_order = list(reversed(order))

    for node in reverse_order:
        successors = instance.successors[node]

        if successors:
            lf[node] = min(ls[s] for s in successors)

        ls[node] = lf[node] - durations[node]

    slack = ls - es

    return {
        "ES": es,
        "EF": ef,
        "LS": ls,
        "LF": lf,
        "slack": slack,
        "horizon": horizon,
    }


# ---------------------------------------------------------------------------
# GRPW features
# ---------------------------------------------------------------------------

def compute_grpw(
    instance: RCPSPInstance,
) -> Dict[str, np.ndarray]:
    """
    Compute GRPW-style structural priority features.

    immediate:
        duration + durations of immediate successors

    all:
        duration + duration of every transitive successor

    These are static precedence-structure descriptors. They are not used as
    the action rule itself; they are supplied to the GNN as node information.
    """

    n = instance.num_activities
    durations = np.asarray(instance.durations, dtype=np.int64)

    immediate = np.zeros(n, dtype=np.int64)
    all_successors = np.zeros(n, dtype=np.int64)

    for node in range(n):
        immediate[node] = int(durations[node])

        for successor in instance.successors[node]:
            immediate[node] += int(durations[successor])

        total = int(durations[node])

        for successor in _successor_closure(instance, node):
            total += int(durations[successor])

        all_successors[node] = total

    return {
        "immediate": immediate,
        "all": all_successors,
    }


# ---------------------------------------------------------------------------
# Static feature container
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class StaticFeatures:
    """
    Everything fixed for one RCPSP instance.

    All node-level arrays are indexed over REAL activities only:
        local node 0 -> PSPLIB activity 1
        local node 1 -> PSPLIB activity 2
        ...
        local node n-1 -> PSPLIB activity n

    The dummy source and sink are excluded.
    """

    instance_name: str
    num_real: int
    num_resources: int

    # Raw instance information.
    durations: np.ndarray
    demands: np.ndarray
    capacities: np.ndarray

    # CPM.
    es: np.ndarray
    ef: np.ndarray
    ls: np.ndarray
    lf: np.ndarray
    slack: np.ndarray

    # Structural priority information.
    grpw_immediate: np.ndarray
    grpw_all: np.ndarray

    # Global normalization quantities.
    horizon: int
    duration_scale: float

    # Normalized structural features.
    pred_count_norm: np.ndarray
    succ_count_norm: np.ndarray
    demand_ratio: np.ndarray
    slack_norm: np.ndarray
    lf_norm: np.ndarray
    grpw_norm_immediate: np.ndarray
    grpw_norm_all: np.ndarray

    # Graph.
    edge_index: np.ndarray

    # Real-activity precedence relationships.
    real_predecessors: List[List[int]]
    real_successors: List[List[int]]

    # Extra structural/resource features used by the 27-feature representation.
    max_demand_ratio: np.ndarray
    mean_demand_ratio: np.ndarray
    resource_tightness: np.ndarray

    downstream_critical_workload: np.ndarray
    downstream_remaining_workload: np.ndarray
    downstream_remaining_activity_ratio: np.ndarray

    immediate_grpw_norm: np.ndarray
    all_successor_grpw_norm: np.ndarray


# ---------------------------------------------------------------------------
# Static feature computation
# ---------------------------------------------------------------------------

def compute_static_features(
    instance: RCPSPInstance,
) -> StaticFeatures:
    """
    Compute all static features from THIS instance only.

    This function never shares features between different PSPLIB instances.
    """

    n_all = int(instance.num_activities)

    if n_all < 3:
        raise ValueError(
            f"Instance {instance.name} must contain source, sink, "
            f"and at least one real activity."
        )

    n = n_all - 2
    k = int(instance.num_resources)

    real = slice(1, n_all - 1)

    durations_all = np.asarray(instance.durations)
    demands_all = np.asarray(instance.resource_demands)
    capacities = np.asarray(instance.resource_capacities)

    if durations_all.shape[0] != n_all:
        raise ValueError(
            f"{instance.name}: duration vector has length "
            f"{durations_all.shape[0]}, expected {n_all}."
        )

    if demands_all.shape[0] != n_all:
        raise ValueError(
            f"{instance.name}: resource demand matrix has "
            f"{demands_all.shape[0]} rows, expected {n_all}."
        )

    if demands_all.shape[1] != k:
        raise ValueError(
            f"{instance.name}: demand matrix has {demands_all.shape[1]} "
            f"resources, expected {k}."
        )

    if capacities.shape[0] != k:
        raise ValueError(
            f"{instance.name}: capacity vector has length "
            f"{capacities.shape[0]}, expected {k}."
        )

    cpm = compute_cpm(instance)
    grpw = compute_grpw(instance)

    horizon = int(max(int(cpm["horizon"]), 1))

    durations = durations_all[real].astype(np.int64, copy=True)
    demands = demands_all[real].astype(np.int64, copy=True)
    caps = capacities.astype(np.int64, copy=True)

    # ------------------------------------------------------------------
    # Real-activity precedence graph.
    # ------------------------------------------------------------------

    real_pred: List[List[int]] = []
    real_succ: List[List[int]] = []

    src: List[int] = []
    dst: List[int] = []

    for local_node in range(n):
        activity_id = local_node + 1

        predecessors = [
            p - 1
            for p in instance.predecessors[activity_id]
            if 1 <= p <= n
        ]

        successors = [
            s - 1
            for s in instance.successors[activity_id]
            if 1 <= s <= n
        ]

        real_pred.append(predecessors)
        real_succ.append(successors)

        for predecessor in predecessors:
            src.append(predecessor)
            dst.append(local_node)

    if src:
        edge_index = np.asarray(
            [src, dst],
            dtype=np.int64,
        )
    else:
        edge_index = np.zeros(
            (2, 0),
            dtype=np.int64,
        )

    # ------------------------------------------------------------------
    # Basic structural quantities.
    # ------------------------------------------------------------------

    pred_count = np.asarray(
        [len(p) for p in real_pred],
        dtype=np.float32,
    )

    succ_count = np.asarray(
        [len(s) for s in real_succ],
        dtype=np.float32,
    )

    pred_count_norm = _normalize(pred_count)
    succ_count_norm = _normalize(succ_count)

    duration_scale = float(
        max(
            int(np.max(durations)) if durations.size else 1,
            1,
        )
    )

    demand_ratio = _safe_div(
        demands.astype(np.float32),
        caps.astype(np.float32)[None, :],
    )

    max_demand_ratio = np.max(
        demand_ratio,
        axis=1,
    ).astype(np.float32)

    mean_demand_ratio = np.mean(
        demand_ratio,
        axis=1,
    ).astype(np.float32)

    # A resource-tightness descriptor. This is intentionally kept as a
    # separate feature so the representation remains explicit and debuggable.
    resource_tightness = max_demand_ratio.copy()

    # ------------------------------------------------------------------
    # CPM-derived quantities.
    # ------------------------------------------------------------------

    slack_real = np.asarray(
        cpm["slack"],
        dtype=np.float32,
    )[real]

    lf_real = np.asarray(
        cpm["LF"],
        dtype=np.float32,
    )[real]

    slack_norm = _safe_div(
        slack_real,
        float(horizon),
    ).astype(np.float32)

    lf_norm = _safe_div(
        lf_real,
        float(horizon),
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # GRPW-derived quantities.
    # ------------------------------------------------------------------

    grpw_immediate = np.asarray(
        grpw["immediate"],
        dtype=np.int64,
    )[real]

    grpw_all = np.asarray(
        grpw["all"],
        dtype=np.int64,
    )[real]

    grpw_immediate_norm = _normalize(
        grpw_immediate.astype(np.float32)
    ).astype(np.float32)

    grpw_all_norm = _normalize(
        grpw_all.astype(np.float32)
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Downstream structural features.
    # ------------------------------------------------------------------

    downstream_critical_workload = np.zeros(
        n,
        dtype=np.float32,
    )

    downstream_remaining_workload = np.zeros(
        n,
        dtype=np.float32,
    )

    downstream_remaining_activity_ratio = np.zeros(
        n,
        dtype=np.float32,
    )

    for local_node in range(n):
        activity_id = local_node + 1

        descendants = _successor_closure(
            instance,
            activity_id,
        )

        # Keep only real activities.
        descendants_real = [
            node
            for node in descendants
            if 1 <= node <= n
        ]

        if descendants_real:
            descendant_local = [
                node - 1
                for node in descendants_real
            ]

            # Total downstream workload.
            downstream_work = float(
                np.sum(
                    durations[
                        descendant_local
                    ]
                )
            )

            downstream_remaining_workload[local_node] = (
                downstream_work
            )

            downstream_remaining_activity_ratio[local_node] = (
                len(descendant_local) / max(float(n), 1.0)
            )

            # Critical downstream workload:
            # prioritize descendants with low CPM slack.
            critical_work = 0.0

            for descendant_id in descendants_real:
                descendant_local_id = descendant_id - 1

                descendant_duration = float(
                    durations[descendant_local_id]
                )

                descendant_slack = max(
                    float(
                        np.asarray(
                            cpm["slack"],
                            dtype=np.float32,
                        )[descendant_id]
                    ),
                    0.0,
                )

                critical_weight = 1.0 / (
                    1.0 + descendant_slack
                )

                critical_work += (
                    descendant_duration
                    * critical_weight
                )

            downstream_critical_workload[local_node] = (
                critical_work
            )

    downstream_critical_workload = _normalize(
        downstream_critical_workload
    ).astype(np.float32)

    downstream_remaining_workload = _normalize(
        downstream_remaining_workload
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Return complete static representation.
    # ------------------------------------------------------------------

    return StaticFeatures(
        instance_name=instance.name,
        num_real=n,
        num_resources=k,

        durations=durations,
        demands=demands,
        capacities=caps,

        es=np.asarray(cpm["ES"], dtype=np.float32)[real].copy(),
        ef=np.asarray(cpm["EF"], dtype=np.float32)[real].copy(),
        ls=np.asarray(cpm["LS"], dtype=np.float32)[real].copy(),
        lf=lf_real.copy(),
        slack=slack_real.copy(),

        grpw_immediate=grpw_immediate.copy(),
        grpw_all=grpw_all.copy(),

        horizon=horizon,
        duration_scale=duration_scale,

        pred_count_norm=pred_count_norm.astype(np.float32),
        succ_count_norm=succ_count_norm.astype(np.float32),

        demand_ratio=demand_ratio.astype(np.float32),

        slack_norm=slack_norm.astype(np.float32),
        lf_norm=lf_norm.astype(np.float32),

        grpw_norm_immediate=grpw_immediate_norm,
        grpw_norm_all=grpw_all_norm,

        edge_index=edge_index,

        real_predecessors=real_pred,
        real_successors=real_succ,

        max_demand_ratio=max_demand_ratio,
        mean_demand_ratio=mean_demand_ratio,
        resource_tightness=resource_tightness,

        downstream_critical_workload=(
            downstream_critical_workload
        ),

        downstream_remaining_workload=(
            downstream_remaining_workload
        ),

        downstream_remaining_activity_ratio=(
            downstream_remaining_activity_ratio
        ),

        immediate_grpw_norm=grpw_immediate_norm.copy(),
        all_successor_grpw_norm=grpw_all_norm.copy(),
    )


# ---------------------------------------------------------------------------
# Static feature cache
# ---------------------------------------------------------------------------

class StaticFeatureCache:
    """
    Cache static features PER INSTANCE.

    The environment uses:

        cache.get(instance)

    The cache therefore must not represent one fixed instance. It stores
    the static representation of every instance that has been requested.

    Instances are keyed by their unique instance name and verified before
    returning cached features.
    """

    def __init__(
        self,
        instance: RCPSPInstance | None = None,
    ) -> None:
        self._cache: Dict[str, StaticFeatures] = {}

        # Backward compatibility with code that constructs:
        #
        #     StaticFeatureCache(instance)
        #
        # The cache remains reusable for all later instances.
        if instance is not None:
            self.get(instance)

    def get(
        self,
        instance: RCPSPInstance,
    ) -> StaticFeatures:
        """
        Return static features for `instance`.

        Features are computed once and reused afterwards.
        """

        key = str(instance.name)

        static = self._cache.get(key)

        if static is None:
            static = compute_static_features(instance)
            self._cache[key] = static

        # Safety checks prevent accidental feature/instance mixing.
        expected_durations = np.asarray(
            instance.durations[1:-1],
            dtype=np.int64,
        )

        expected_capacities = np.asarray(
            instance.resource_capacities,
            dtype=np.int64,
        )

        if static.instance_name != instance.name:
            raise RuntimeError(
                f"Static features do not match instance "
                f"{instance.name}: cached name is "
                f"{static.instance_name}."
            )

        if static.num_real != instance.num_activities - 2:
            raise RuntimeError(
                f"Static features for {instance.name} have "
                f"num_real={static.num_real}, expected "
                f"{instance.num_activities - 2}."
            )

        if not np.array_equal(
            static.durations,
            expected_durations,
        ):
            raise RuntimeError(
                f"Static duration features do not match "
                f"instance {instance.name}."
            )

        if not np.array_equal(
            static.capacities,
            expected_capacities,
        ):
            raise RuntimeError(
                f"Static resource capacities do not match "
                f"instance {instance.name}."
            )

        return static

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def features(self) -> StaticFeatures:
        """
        Backward-compatible access for code that previously expected:

            cache.features

        This is valid only when the cache contains exactly one instance.
        """

        if len(self._cache) != 1:
            raise AttributeError(
                "'features' is only available when the cache "
                "contains exactly one instance"
            )

        return next(iter(self._cache.values()))

    def __getattr__(self, name: str):
        """
        Backward-compatible forwarding to the single cached StaticFeatures
        object.

        This is intentionally only used for old code accessing attributes
        such as cache.edge_index or cache.duration_scale.
        """

        if name.startswith("_"):
            raise AttributeError(name)

        cache = self.__dict__.get("_cache")

        if cache is not None and len(cache) == 1:
            static = next(iter(cache.values()))
            return getattr(static, name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

def feature_names(
    cfg: Config,
    num_resources: int,
) -> List[str]:
    """
    Return feature names in exactly the same order as build_node_features().
    """

    k = int(num_resources)

    names = [
        # Base: 9
        "status_not_started",
        "status_running",
        "status_completed",
        "startable",
        "remaining_duration",
        "duration_norm",
        "num_pred",
        "num_succ",
        "pred_completion_ratio",
    ]

    if cfg.use_demand_ratio:
        names += [
            f"demand_ratio_r{i + 1}"
            for i in range(k)
        ]

        names += [
            "max_demand_ratio",
            "mean_demand_ratio",
            "resource_tightness",
        ]

    if cfg.use_criticality:
        names += [
            "slack_norm",
            "lf_norm",
            "downstream_critical_workload",
        ]

    if cfg.use_structural:
        names += [
            "grpw_immediate",
            "grpw_all_successor",
            "downstream_remaining_workload",
            "downstream_remaining_activity_ratio",
        ]

    if cfg.use_resource_pressure:
        names += [
            f"free_ratio_r{i + 1}"
            for i in range(k)
        ]

    return names


# ---------------------------------------------------------------------------
# Dynamic node representation
# ---------------------------------------------------------------------------

def build_node_features(
    static: StaticFeatures,
    status: np.ndarray,
    remaining: np.ndarray,
    startable: np.ndarray,
    free_ratio: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    Assemble the current (n, F) node-feature matrix.

    Parameters
    ----------
    static:
        StaticFeatures for the CURRENT RCPSP instance.

    status:
        Shape (n,), using:
            0 = not started
            1 = running
            2 = completed

    remaining:
        Shape (n,), remaining processing duration.

    startable:
        Shape (n,), boolean mask indicating activities that can currently
        be started.

    free_ratio:
        Shape (K,), current free-resource / total-capacity ratios.

    cfg:
        DI-GNN-PPO configuration.

    Returns
    -------
    np.ndarray
        Float32 matrix with exactly cfg.node_feature_dim() columns.
    """

    n = static.num_real
    k = static.num_resources

    status = np.asarray(status, dtype=np.int64)
    remaining = np.asarray(remaining, dtype=np.float32)
    startable = np.asarray(startable, dtype=bool)
    free_ratio = np.asarray(free_ratio, dtype=np.float32)

    if status.shape != (n,):
        raise ValueError(
            f"status shape {status.shape} != {(n,)}"
        )

    if remaining.shape != (n,):
        raise ValueError(
            f"remaining shape {remaining.shape} != {(n,)}"
        )

    if startable.shape != (n,):
        raise ValueError(
            f"startable shape {startable.shape} != {(n,)}"
        )

    if free_ratio.shape != (k,):
        raise ValueError(
            f"free_ratio shape {free_ratio.shape} != {(k,)}"
        )

    if np.any(
        (status < STATUS_NOT_STARTED)
        | (status > STATUS_COMPLETED)
    ):
        raise ValueError(
            "status contains values outside "
            "{STATUS_NOT_STARTED, STATUS_RUNNING, STATUS_COMPLETED}"
        )

    cols: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Base features: 9
    # ------------------------------------------------------------------

    one_hot_status = np.zeros(
        (n, 3),
        dtype=np.float32,
    )

    one_hot_status[
        np.arange(n),
        status,
    ] = 1.0

    cols.append(one_hot_status)

    # 4. Startable.
    cols.append(
        startable.astype(np.float32)[:, None]
    )

    # 5. Remaining duration.
    cols.append(
        _safe_div(
            remaining,
            static.duration_scale,
        )[:, None]
    )

    # 6. Static normalized duration.
    cols.append(
        _safe_div(
            static.durations.astype(np.float32),
            static.duration_scale,
        )[:, None]
    )

    # 7. Number of predecessors.
    cols.append(
        static.pred_count_norm[:, None]
    )

    # 8. Number of successors.
    cols.append(
        static.succ_count_norm[:, None]
    )

    # 9. Fraction of immediate predecessors already completed.
    pred_completion_ratio = np.zeros(
        n,
        dtype=np.float32,
    )

    for node in range(n):
        predecessors = static.real_predecessors[node]

        if not predecessors:
            pred_completion_ratio[node] = 1.0
        else:
            pred_completion_ratio[node] = float(
                np.mean(
                    status[
                        predecessors
                    ]
                    == STATUS_COMPLETED
                )
            )

    cols.append(
        pred_completion_ratio[:, None]
    )

    # ------------------------------------------------------------------
    # Resource-demand features: 7
    # ------------------------------------------------------------------

    if cfg.use_demand_ratio:
        cols.append(
            static.demand_ratio.astype(np.float32)
        )

        cols.append(
            static.max_demand_ratio[:, None]
        )

        cols.append(
            static.mean_demand_ratio[:, None]
        )

        cols.append(
            static.resource_tightness[:, None]
        )

    # ------------------------------------------------------------------
    # Criticality features: 3
    # ------------------------------------------------------------------

    if cfg.use_criticality:
        cols.append(
            static.slack_norm[:, None]
        )

        cols.append(
            static.lf_norm[:, None]
        )

        cols.append(
            static.downstream_critical_workload[:, None]
        )

    # ------------------------------------------------------------------
    # Structural features: 4
    # ------------------------------------------------------------------

    if cfg.use_structural:
        cols.append(
            static.immediate_grpw_norm[:, None]
        )

        cols.append(
            static.all_successor_grpw_norm[:, None]
        )

        cols.append(
            static.downstream_remaining_workload[:, None]
        )

        cols.append(
            static.downstream_remaining_activity_ratio[:, None]
        )

    # ------------------------------------------------------------------
    # Dynamic resource-pressure features: K
    # ------------------------------------------------------------------

    if cfg.use_resource_pressure:
        free_resource_features = np.broadcast_to(
            free_ratio.astype(np.float32)[None, :],
            (n, k),
        )

        cols.append(
            free_resource_features
        )

    x = np.concatenate(
        cols,
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )

    expected_dim = int(
        cfg.node_feature_dim()
    )

    if x.shape != (n, expected_dim):
        raise AssertionError(
            f"Node feature shape {x.shape} does not match "
            f"expected {(n, expected_dim)}."
        )

    return np.ascontiguousarray(x)


# ---------------------------------------------------------------------------
# Graph edge representation
# ---------------------------------------------------------------------------

def get_edge_index(
    static: StaticFeatures,
    cfg: Config,
) -> np.ndarray:
    """
    Return the graph edge index.

    Original edges are:
        predecessor -> successor

    If bidirectional_edges=True, the reverse edges are added as well.

    The returned indices are LOCAL real-activity indices.
    """

    edge_index = np.asarray(
        static.edge_index,
        dtype=np.int64,
    )

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape (2, E), "
            f"got {edge_index.shape}."
        )

    if (
        cfg.bidirectional_edges
        and edge_index.shape[1] > 0
    ):
        edge_index = np.concatenate(
            [
                edge_index,
                edge_index[::-1],
            ],
            axis=1,
        )

    return np.ascontiguousarray(
        edge_index,
        dtype=np.int64,
    )