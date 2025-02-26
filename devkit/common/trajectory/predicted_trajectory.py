from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Union

from devkit.common.actor_state.ego_state import EgoState
from devkit.common.actor_state.waypoint import Waypoint
from devkit.common.trajectory.abstract_trajectory import AbstractTrajectory
from devkit.common.trajectory.interpolated_trajectory import InterpolatedTrajectory

WaypointTypes = Union[Waypoint, EgoState]


@dataclass
class PredictedTrajectory:
    """Stores a predicted trajectory, along with its probability."""

    # Probability assigned to this trajectory prediction
    probability: float

    # List of predicted waypoints, if None, we appended the predictions to have desired length
    waypoints: List[Optional[WaypointTypes]]

    @property
    def valid_waypoints(self) -> List[WaypointTypes]:
        """
        Interface to get only valid waypoints
        :return: waypoints which are not None
        """
        return [w for w in self.waypoints if w]

    @cached_property
    def trajectory(self) -> AbstractTrajectory:
        """
        Interface to compute trajectory from waypoints
        :return: trajectory from waypoints
        """
        return InterpolatedTrajectory(self.valid_waypoints)

    def __len__(self) -> int:
        """
        :return: number of waypoints in trajectory
        """
        return len(self.waypoints)
