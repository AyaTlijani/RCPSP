from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from psplib import parse


@dataclass
class Activity:
    id: int
    duration: int
    resource_demands: Dict[int, int]
    predecessors: List[int]
    successors: List[int]

    def is_dummy(self) -> bool:
        return self.duration == 0


@dataclass
class Resource:
    id: int
    capacity: int


@dataclass
class Project:
    project_id: str
    activities: List[Activity]
    resources: List[Resource]

    @property
    def num_activities(self) -> int:
        return len(self.activities)

    @property
    def num_resources(self) -> int:
        return len(self.resources)

    @property
    def real_activities(self) -> List[Activity]:
        return [
            activity
            for activity in self.activities
            if not activity.is_dummy()
        ]


class RCPSPParser:

    @staticmethod
    def parse_file(file_path: Path) -> Project:

        project_data = parse(file_path)

        # Resources
        resources = [
            Resource(
                id=index,
                capacity=resource.capacity
            )
            for index, resource in enumerate(project_data.resources)
        ]

        # Build predecessor lists from successors
        predecessors = [
            []
            for _ in range(len(project_data.activities))
        ]

        for activity_id, activity in enumerate(project_data.activities):

            for successor in activity.successors:
                predecessors[successor].append(activity_id)

        # Activities
        activities = []

        for activity_id, activity in enumerate(project_data.activities):

            mode = activity.modes[0]

            activities.append(
                Activity(
                    id=activity_id,
                    duration=mode.duration,
                    resource_demands={
                        resource_id: demand
                        for resource_id, demand in enumerate(mode.demands)
                    },
                    predecessors=predecessors[activity_id],
                    successors=list(activity.successors)
                )
            )

        return Project(
            project_id=file_path.stem,
            activities=activities,
            resources=resources
        )