from typing import List, Dict
from enum import Enum

from parser import Project


class ActivityStatus(Enum):
    NOT_STARTED = 0
    RUNNING = 1
    COMPLETED = 2


class RCPSPEnvironment:
    """
    RCPSP scheduling environment.

    Handles:
    - activity status
    - precedence constraints
    - resource constraints
    - simulation time
    - available actions
    - makespan
    """

    def __init__(self, project: Project):
        self.project = project

        self.activities = {
            activity.id: activity
            for activity in project.activities
        }

        self.real_activities = {
            activity.id: activity
            for activity in project.real_activities
        }

        self.resources = {
            resource.id: resource.capacity
            for resource in project.resources
        }

        self.reset()

    def reset(self):
        self.current_time = 0

        self.status = {
            activity.id: ActivityStatus.NOT_STARTED
            for activity in self.real_activities.values()
        }

        self.start_times = {}
        self.finish_times = {}

        self.running_activities = set()
        self.completed_activities = set()

        self.resource_usage = {
            resource_id: 0
            for resource_id in self.resources
        }

        return self.get_state()

    def get_free_resources(self) -> Dict[int, int]:
        return {
            resource_id: capacity - self.resource_usage[resource_id]
            for resource_id, capacity in self.resources.items()
        }

    def get_state(self):
        residual_duration = {}

        for activity in self.real_activities.values():

            if self.status[activity.id] == ActivityStatus.RUNNING:
                residual_duration[activity.id] = (
                    self.finish_times[activity.id] - self.current_time
                )

            elif self.status[activity.id] == ActivityStatus.COMPLETED:
                residual_duration[activity.id] = 0

            else:
                residual_duration[activity.id] = activity.duration

        return {
            "time": self.current_time,
            "status": {
                activity_id: status.value
                for activity_id, status in self.status.items()
            },
            "resource_usage": self.resource_usage.copy(),
            "r_free": self.get_free_resources(),
            "residual_duration": residual_duration,
            "available_actions": self.get_available_actions(),
        }

    def get_available_actions(self) -> List[int]:
        available = []

        for activity in self.real_activities.values():

            if self.status[activity.id] != ActivityStatus.NOT_STARTED:
                continue

            predecessors_done = all(
                pred in self.completed_activities
                for pred in activity.predecessors
                if pred in self.real_activities
            )

            if not predecessors_done:
                continue

            if not self.check_resources(activity):
                continue

            available.append(activity.id)

        return available

    def check_resources(self, activity) -> bool:
        for resource_id, demand in activity.resource_demands.items():
            if (
                self.resource_usage[resource_id] + demand
                > self.resources[resource_id]
            ):
                return False

        return True

    def start_activity(self, activity_id: int):
        activity = self.real_activities[activity_id]

        self.status[activity_id] = ActivityStatus.RUNNING

        self.start_times[activity_id] = self.current_time

        self.finish_times[activity_id] = (
            self.current_time + activity.duration
        )

        self.running_activities.add(activity_id)

        for resource_id, demand in activity.resource_demands.items():
            self.resource_usage[resource_id] += demand

    def advance_time(self):
        if not self.running_activities:
            return

        next_finish = min(
            self.finish_times[activity_id]
            for activity_id in self.running_activities
        )

        self.current_time = next_finish

        finished = [
            activity_id
            for activity_id in self.running_activities
            if self.finish_times[activity_id] <= self.current_time
        ]

        for activity_id in finished:
            activity = self.real_activities[activity_id]

            self.status[activity_id] = ActivityStatus.COMPLETED
            self.completed_activities.add(activity_id)
            self.running_activities.remove(activity_id)

            for resource_id, demand in activity.resource_demands.items():
                self.resource_usage[resource_id] -= demand

    def step(self, activity_id: int):
        if activity_id not in self.get_available_actions():
            raise ValueError(
                f"Activity {activity_id} cannot start"
            )

        self.start_activity(activity_id)

        return self.get_state()

    def is_done(self):
        return (
            len(self.completed_activities)
            == len(self.real_activities)
        )

    def get_makespan(self):
        if not self.finish_times:
            return 0

        return max(self.finish_times.values())