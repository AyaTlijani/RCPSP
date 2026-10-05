from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data

from parser import Project


class RCPSPGraphBuilder:

    def __init__(
        self,
        project: Project,
        resource_ids: List[int],
        state: Optional[Dict] = None,
    ):
        self.project = project
        self.state = state
        self.resource_ids = resource_ids
        self.activities = project.real_activities

        self.id_mapping = {
            activity.id: idx
            for idx, activity in enumerate(self.activities)
        }

        self.reverse_mapping = {
            idx: activity.id
            for idx, activity in enumerate(self.activities)
        }

        self.resource_capacities = {
            resource.id: resource.capacity
            for resource in project.resources
        }

        durations = [a.duration for a in self.activities]
        self.max_duration = max(durations) if durations else 1

        pred_counts = [
            len([
                p for p in activity.predecessors
                if p in self.id_mapping
            ])
            for activity in self.activities
        ]

        succ_counts = [
            len([
                s for s in activity.successors
                if s in self.id_mapping
            ])
            for activity in self.activities
        ]

        self.max_pred_count = max(pred_counts) if pred_counts else 1
        self.max_succ_count = max(succ_counts) if succ_counts else 1

    def build_node_features(self):

        features = []

        for activity in self.activities:

            node_features = []

            # Activity status
            status = (
                self.state["status"][activity.id]
                if self.state
                else 0
            )

            node_features.extend([
                1.0 if status == 0 else 0.0,
                1.0 if status == 1 else 0.0,
                1.0 if status == 2 else 0.0,
            ])

            # Whether the activity can currently be selected
            available = (
                1.0
                if self.state
                and activity.id in self.state.get(
                    "available_actions", []
                )
                else 0.0
            )

            node_features.append(available)

            # Remaining duration
            if self.state:
                duration = self.state["residual_duration"][activity.id]
            else:
                duration = activity.duration

            node_features.append(
                duration / self.max_duration
            )

            # Predecessor / successor counts
            pred_count = len([
                p for p in activity.predecessors
                if p in self.id_mapping
            ])

            succ_count = len([
                s for s in activity.successors
                if s in self.id_mapping
            ])

            node_features.extend([
                pred_count / self.max_pred_count,
                succ_count / self.max_succ_count,
            ])

            # Resource demands normalized by capacity
            for resource_id in self.resource_ids:

                demand = activity.resource_demands.get(
                    resource_id, 0
                )

                capacity = self.resource_capacities.get(
                    resource_id, 1
                )

                node_features.append(
                    demand / capacity
                    if capacity > 0
                    else 0.0
                )

            features.append(node_features)

        return torch.tensor(
            features,
            dtype=torch.float,
        )

    def build_edge_index(self):

        edges = []

        for activity in self.activities:

            source = self.id_mapping[activity.id]

            for successor_id in activity.successors:

                if successor_id in self.id_mapping:

                    target = self.id_mapping[successor_id]

                    # successor -> predecessor
                    edges.append([target, source])

        if not edges:
            return torch.empty(
                (2, 0),
                dtype=torch.long,
            )

        return torch.tensor(
            edges,
            dtype=torch.long,
        ).t().contiguous()

    def build_graph(self):

        graph = Data(
            x=self.build_node_features(),
            edge_index=self.build_edge_index(),
        )

        if self.state and "r_free" in self.state:

            graph.r_free = torch.tensor(
                [
                    self.state["r_free"][resource_id]
                    for resource_id in self.resource_ids
                ],
                dtype=torch.float,
            )

        graph.node_to_activity = self.reverse_mapping

        return graph