from collections import deque
from typing import Dict, List

import numpy as np

from parser import Project
from environment import RCPSPEnvironment


class RCPSPFeatureExtractor:

    def __init__(self, project: Project):
        self.project = project

        self.activities = {
            activity.id: activity
            for activity in project.real_activities
        }

        self.resource_ids = [
            resource.id
            for resource in project.resources
        ]

        self.resource_capacities = {
            resource.id: float(resource.capacity)
            for resource in project.resources
        }

        self.num_resources = len(self.resource_ids)

        # Static scheduling features.
        self.es = {}
        self.ef = {}
        self.ls = {}
        self.lf = {}
        self.slack = {}
        self.total_successors = {}
        self.grpw = {}

        self.project_duration = 1.0

        self._compute_cpm_features()
        self._compute_successor_features()
        self._compute_grpw()

    # ========================================================
    # TOPOLOGICAL ORDER
    # ========================================================

    def _topological_order(self) -> List[int]:

        indegree = {
            activity_id: 0
            for activity_id in self.activities
        }

        for activity in self.activities.values():

            for successor in activity.successors:

                if successor in indegree:
                    indegree[successor] += 1

        queue = deque(
            activity_id
            for activity_id, degree in indegree.items()
            if degree == 0
        )

        order = []

        while queue:

            activity_id = queue.popleft()
            order.append(activity_id)

            activity = self.activities[activity_id]

            for successor in activity.successors:

                if successor not in indegree:
                    continue

                indegree[successor] -= 1

                if indegree[successor] == 0:
                    queue.append(successor)

        # Safety fallback.
        if len(order) != len(self.activities):

            remaining = [
                activity_id
                for activity_id in self.activities
                if activity_id not in order
            ]

            order.extend(remaining)

        return order

    # ========================================================
    # CPM FEATURES
    # ========================================================

    def _compute_cpm_features(self):

        order = self._topological_order()

        # Forward pass.
        for activity_id in order:

            activity = self.activities[activity_id]

            predecessor_ids = [
                pred
                for pred in activity.predecessors
                if pred in self.activities
            ]

            if predecessor_ids:

                self.es[activity_id] = max(
                    self.ef[pred]
                    for pred in predecessor_ids
                )

            else:

                self.es[activity_id] = 0.0

            self.ef[activity_id] = (
                self.es[activity_id]
                + float(activity.duration)
            )

        if self.ef:

            self.project_duration = max(
                self.ef.values()
            )

        self.project_duration = max(
            self.project_duration,
            1.0
        )

        # Backward pass.
        project_end = self.project_duration

        for activity_id in reversed(order):

            activity = self.activities[activity_id]

            successor_ids = [
                successor
                for successor in activity.successors
                if successor in self.activities
            ]

            if successor_ids:

                self.lf[activity_id] = min(
                    self.ls[successor]
                    for successor in successor_ids
                )

            else:

                self.lf[activity_id] = project_end

            self.ls[activity_id] = (
                self.lf[activity_id]
                - float(activity.duration)
            )

            self.slack[activity_id] = max(
                0.0,
                self.ls[activity_id]
                - self.es[activity_id]
            )

    # ========================================================
    # SUCCESSOR FEATURES
    # ========================================================

    def _compute_successor_features(self):

        for activity_id in self.activities:

            visited = set()
            queue = deque(
                self.activities[activity_id].successors
            )

            while queue:

                successor = queue.popleft()

                if successor in visited:
                    continue

                if successor not in self.activities:
                    continue

                visited.add(successor)

                for next_successor in self.activities[
                    successor
                ].successors:

                    if next_successor not in visited:
                        queue.append(next_successor)

            self.total_successors[activity_id] = len(
                visited
            )

    # ========================================================
    # GRPW
    # ========================================================

    def _compute_grpw(self):

        for activity_id in self.activities:

            total = float(
                self.activities[activity_id].duration
            )

            visited = set()
            queue = deque(
                self.activities[activity_id].successors
            )

            while queue:

                successor = queue.popleft()

                if successor in visited:
                    continue

                if successor not in self.activities:
                    continue

                visited.add(successor)

                total += float(
                    self.activities[successor].duration
                )

                for next_successor in self.activities[
                    successor
                ].successors:

                    if next_successor not in visited:
                        queue.append(next_successor)

            self.grpw[activity_id] = total

    # ========================================================
    # SINGLE ACTIVITY FEATURES
    # ========================================================

    def _activity_features(
        self,
        env: RCPSPEnvironment,
        activity_id: int,
    ) -> List[float]:

        activity = self.activities[activity_id]

        duration = float(activity.duration)

        # Normalize time-related quantities.
        duration_norm = duration / self.project_duration

        es_norm = (
            self.es[activity_id]
            / self.project_duration
        )

        ef_norm = (
            self.ef[activity_id]
            / self.project_duration
        )

        ls_norm = (
            self.ls[activity_id]
            / self.project_duration
        )

        lf_norm = (
            self.lf[activity_id]
            / self.project_duration
        )

        slack_norm = (
            self.slack[activity_id]
            / self.project_duration
        )

        direct_successors = len(
            [
                successor
                for successor in activity.successors
                if successor in self.activities
            ]
        )

        total_successors = (
            self.total_successors[activity_id]
        )

        # Normalize graph counts by number of activities.
        denominator = max(
            1,
            len(self.activities)
        )

        direct_successors_norm = (
            direct_successors / denominator
        )

        total_successors_norm = (
            total_successors / denominator
        )

        grpw_norm = (
            self.grpw[activity_id]
            / self.project_duration
        )

        features = [
            duration_norm,
            es_norm,
            ef_norm,
            ls_norm,
            lf_norm,
            slack_norm,
            direct_successors_norm,
            total_successors_norm,
            grpw_norm,
        ]

        # Resource demands normalized by capacity.
        for resource_id in self.resource_ids:

            capacity = self.resource_capacities[
                resource_id
            ]

            demand = float(
                activity.resource_demands.get(
                    resource_id,
                    0
                )
            )

            if capacity > 0:
                normalized_demand = demand / capacity
            else:
                normalized_demand = 0.0

            features.append(normalized_demand)

        # Current available resources.
        free_resources = env.get_free_resources()

        for resource_id in self.resource_ids:

            capacity = self.resource_capacities[
                resource_id
            ]

            free = float(
                free_resources.get(
                    resource_id,
                    0
                )
            )

            if capacity > 0:
                normalized_free = free / capacity
            else:
                normalized_free = 0.0

            features.append(normalized_free)

        return features

    # ========================================================
    # CANDIDATE FEATURES
    # ========================================================

    def get_candidate_features(
        self,
        env: RCPSPEnvironment,
    ) -> np.ndarray:

        candidate_ids = env.get_available_actions()

        if not candidate_ids:

            return np.empty(
                (0, self.get_feature_dim()),
                dtype=np.float32
            )

        rows = [
            self._activity_features(
                env,
                activity_id
            )
            for activity_id in candidate_ids
        ]

        return np.asarray(
            rows,
            dtype=np.float32
        )

    # ========================================================
    # GLOBAL FEATURES FOR CRITIC
    # ========================================================

    def get_global_features(
        self,
        env: RCPSPEnvironment,
    ) -> np.ndarray:

        total_activities = max(
            1,
            len(env.real_activities)
        )

        completed_ratio = (
            len(env.completed_activities)
            / total_activities
        )

        running_ratio = (
            len(env.running_activities)
            / total_activities
        )

        current_time_norm = (
            env.current_time
            / self.project_duration
        )

        features = [
            current_time_norm,
            completed_ratio,
            running_ratio,
        ]

        free_resources = env.get_free_resources()

        for resource_id in self.resource_ids:

            capacity = self.resource_capacities[
                resource_id
            ]

            free = float(
                free_resources.get(
                    resource_id,
                    0
                )
            )

            if capacity > 0:
                normalized_free = free / capacity
            else:
                normalized_free = 0.0

            features.append(normalized_free)

        return np.asarray(
            features,
            dtype=np.float32
        )

    # ========================================================
    # DIMENSIONS
    # ========================================================

    def get_feature_dim(self) -> int:
        return 9 + 2 * self.num_resources

    def get_global_dim(self) -> int:
        return 3 + self.num_resources