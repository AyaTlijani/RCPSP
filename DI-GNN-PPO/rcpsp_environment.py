"""
rcpsp_environment.py
====================
Event-driven (non-delay / "parallel"-style) scheduling environment for the
deterministic single-mode RCPSP.

Decision process
----------------
At every decision point the agent picks ONE currently *startable* activity
(precedence-eligible AND demand <= free resources).  The activity starts at the
current time.  When nothing else is startable, time jumps to the next activity
completion event: finished activities are completed, their resources released,
eligibility is updated, and this repeats until something is startable again or the
project is complete.  Time is never advanced one unit at a time, and the agent
cannot "wait" while something is startable.

Reward
------
Whenever time advances by dt, reward -= dt / reward_scale.  The final advance (from
the last activity start to the completion of the last running activity) is produced
by the same loop, so the undiscounted return is EXACTLY  -makespan / reward_scale.
(Checked by `check_return_matches_makespan`-style tests; see `info["return_check"]`.)

Indexing
--------
Actions and node indices are over REAL activities: action r  <->  activity index r+1
of the `RCPSPInstance` (dummy source = 0, dummy sink = N-1).  The dummy source is
completed at t = 0; the dummy sink completes when all real activities are completed,
so makespan = time at which the last real activity finishes.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from config import Config
from domain_features import (STATUS_COMPLETED, STATUS_NOT_STARTED, STATUS_RUNNING,
                             StaticFeatureCache, StaticFeatures, build_node_features,
                             get_edge_index)
from psplib_loader import RCPSPInstance

Observation = Dict[str, np.ndarray]


class RCPSPEnv:
    def __init__(self, cfg: Config, feature_cache: Optional[StaticFeatureCache] = None) -> None:
        self.cfg = cfg
        self.feature_cache = feature_cache if feature_cache is not None else StaticFeatureCache()
        self.instance: Optional[RCPSPInstance] = None
        self.static: Optional[StaticFeatures] = None

    # ------------------------------------------------------------------ reset
    def reset(self, instance: RCPSPInstance) -> Observation:
        if instance.num_resources != self.cfg.num_resources:
            raise ValueError(f"{instance.name}: {instance.num_resources} resources, "
                             f"config expects {self.cfg.num_resources}")
        self.instance = instance
        self.static = self.feature_cache.get(instance)          # THIS instance's features
        st = self.static
        self.n = st.num_real
        self.d = st.durations
        self.dem = st.demands
        self.cap = st.capacities
        self.edge_index = get_edge_index(st, self.cfg)

        self.time = 0
        self.status = np.full(self.n, STATUS_NOT_STARTED, dtype=np.int64)
        self.start_time = np.full(self.n, -1, dtype=np.int64)
        self.finish_time = np.full(self.n, -1, dtype=np.int64)
        self.unfinished_preds = np.array([len(p) for p in st.real_predecessors], dtype=np.int64)
        self.available = self.cap.copy()
        self.num_completed = 0
        self.total_reward = 0.0
        self.done = False
        self._update_startable()
        if not self.startable.any():
            raise RuntimeError(f"{instance.name}: no activity can start at t=0")
        return self.observe()

    # ------------------------------------------------------------ observation
    def _update_startable(self) -> None:
        fits = (self.dem <= self.available[None, :]).all(axis=1)
        self.precedence_eligible = (self.status == STATUS_NOT_STARTED) & (self.unfinished_preds == 0)
        self.startable = self.precedence_eligible & fits

    def remaining_durations(self) -> np.ndarray:
        rem = np.zeros(self.n, dtype=np.float64)
        ns = self.status == STATUS_NOT_STARTED
        run = self.status == STATUS_RUNNING
        rem[ns] = self.d[ns]
        rem[run] = self.finish_time[run] - self.time
        return rem                                               # completed -> 0

    def free_ratio(self) -> np.ndarray:
        return (self.available / np.maximum(self.cap, 1)).astype(np.float32)

    def observe(self) -> Observation:
        free = self.free_ratio()
        x = build_node_features(self.static, self.status, self.remaining_durations(),
                                self.startable, free, self.cfg)
        return {
            "x": x,                                   # (n, F) float32
            "mask": self.startable.copy(),            # (n,)  bool  valid actions
            "free_ratio": free,                       # (K,)  float32 current free/capacity
            "edge_index": self.edge_index,            # (2,E) int64 (shared, do not modify)
            "num_nodes": self.n,
        }

    # ------------------------------------------------------------------- step
    def _advance_to_next_event(self) -> float:
        """Jump to the next completion event; returns the (scaled) reward of that jump."""
        running = np.flatnonzero(self.status == STATUS_RUNNING)
        if running.size == 0:
            raise RuntimeError(f"{self.instance.name}: deadlock (nothing running, nothing startable)")
        next_t = int(self.finish_time[running].min())
        dt = next_t - self.time
        self.time = next_t
        for r in running[self.finish_time[running] <= next_t]:
            self.status[r] = STATUS_COMPLETED
            self.available += self.dem[r]
            self.num_completed += 1
            for s in self.static.real_successors[r]:
                self.unfinished_preds[s] -= 1
        return -float(dt) / self.cfg.reward_scale

    def step(self, action: int) -> Tuple[Observation, float, bool, dict]:
        if self.done:
            raise RuntimeError("step() called on a finished episode")
        action = int(action)
        if not (0 <= action < self.n) or not self.startable[action]:
            raise ValueError(f"action {action} is not startable (valid: {np.flatnonzero(self.startable).tolist()})")

        # start the chosen activity now
        self.status[action] = STATUS_RUNNING
        self.start_time[action] = self.time
        self.finish_time[action] = self.time + self.d[action]
        self.available -= self.dem[action]
        if np.any(self.available < 0):
            raise AssertionError("resource over-allocation")

        # advance time only while no activity can start and the project is unfinished
        reward = 0.0
        self._update_startable()
        while self.num_completed < self.n and not self.startable.any():
            reward += self._advance_to_next_event()
            self._update_startable()
        self.total_reward += reward
        self.done = self.num_completed == self.n
        info: dict = {}
        if self.done:
            info["makespan"] = int(self.time)
            info["return_check"] = abs(self.total_reward + self.time / self.cfg.reward_scale) < 1e-6
        return self.observe(), reward, self.done, info

    # ---------------------------------------------------------------- results
    @property
    def makespan(self) -> int:
        if not self.done:
            raise RuntimeError("episode not finished")
        return int(self.time)

    def check_schedule(self) -> None:
        """Independent feasibility check of the finished schedule (precedence + resources)."""
        if not self.done:
            raise RuntimeError("episode not finished")
        st, ft, d = self.start_time, self.finish_time, self.d
        if np.any(ft - st != d):
            raise AssertionError("finish - start != duration")
        for r in range(self.n):
            for p in self.static.real_predecessors[r]:
                if st[r] < ft[p]:
                    raise AssertionError(f"precedence violated: {p} -> {r}")
        for t in np.unique(st):                                  # usage only changes at start times
            active = (st <= t) & (ft > t)
            use = self.dem[active].sum(axis=0)
            if np.any(use > self.cap):
                raise AssertionError(f"resource capacity exceeded at t={t}")
        if int(ft.max()) != self.time:
            raise AssertionError("makespan != max finish time")