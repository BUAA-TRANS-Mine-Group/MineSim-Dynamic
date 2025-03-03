from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Generator, List, Optional, Set, Tuple, Union, cast

from devkit.common.actor_state.agent import Agent
from devkit.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from devkit.common.actor_state.waypoint import Waypoint
from devkit.common.trajectory.predicted_trajectory import PredictedTrajectory
from devkit.common.trajectory.trajectory_sampling import TrajectorySampling

logger = logging.getLogger(__name__)

LIDAR_PC_CACHE = 16 * 2**10  # 16K

DEFAULT_SCENARIO_NAME = "unknown"  # name of scenario (e.g. ego overtaking)
DEFAULT_SCENARIO_DURATION = 20.0  # [s] duration of the scenario (e.g. extract 20s from when the event occurred)
DEFAULT_EXTRACTION_OFFSET = 0.0  # [s] offset of the scenario (e.g. start at -5s from when the event occurred)
DEFAULT_SUBSAMPLE_RATIO = 1.0  # ratio used sample the scenario (e.g. a 0.1 ratio means sample from 20Hz to 2Hz)

NUPLAN_DATA_ROOT = os.getenv("NUPLAN_DATA_ROOT", "/data/sets/devkit/")


@dataclass(frozen=True)
class ScenarioExtractionInfo:
    """
    Structure containing information used to extract a scenario (lidarpc sequence).
    """

    scenario_name: str = DEFAULT_SCENARIO_NAME  # name of the scenario
    scenario_duration: float = DEFAULT_SCENARIO_DURATION  # [s] duration of the scenario
    extraction_offset: float = DEFAULT_EXTRACTION_OFFSET  # [s] offset of the scenario
    subsample_ratio: float = DEFAULT_SUBSAMPLE_RATIO  # ratio to sample the scenario

    def __post_init__(self) -> None:
        """Sanitize class attributes."""
        assert 0.0 < self.scenario_duration, f"Scenario duration has to be greater than 0, got {self.scenario_duration}"
        assert 0.0 < self.subsample_ratio <= 1.0, f"Subsample ratio has to be between 0 and 1, got {self.subsample_ratio}"


class ScenarioMapping:
    """
    Structure that maps each scenario type to instructions used in extracting it.
    """

    def __init__(
        self,
        scenario_map: Dict[str, Union[Tuple[float, float, float], Tuple[float, float]]],
        subsample_ratio_override: Optional[float],
    ) -> None:
        """
        Initializes the scenario mapping class.
        :param scenario_map: Dictionary with scenario name/type as keys and
                             tuples of (scenario duration, extraction offset, subsample ratio) as values.
        :subsample_ratio_override: The override for the subsample ratio if not provided.
        """
        self.mapping: Dict[str, ScenarioExtractionInfo] = {}
        self.subsample_ratio_override = subsample_ratio_override if subsample_ratio_override is not None else DEFAULT_SUBSAMPLE_RATIO

        for name in scenario_map:
            this_ratio: float = scenario_map[name][2] if len(scenario_map[name]) == 3 else self.subsample_ratio_override  # type: ignore

            self.mapping[name] = ScenarioExtractionInfo(
                scenario_name=name,
                scenario_duration=scenario_map[name][0],
                extraction_offset=scenario_map[name][1],
                subsample_ratio=this_ratio,
            )

    def get_extraction_info(self, scenario_type: str) -> Optional[ScenarioExtractionInfo]:
        """
        Accesses the scenario mapping using a query scenario type.
        If the scenario type is not found, a default extraction info object is returned.
        :param scenario_type: Scenario type to query for.
        :return: Scenario extraction information for the queried scenario type.
        """
        return self.mapping[scenario_type] if scenario_type in self.mapping else ScenarioExtractionInfo(subsample_ratio=self.subsample_ratio_override)

 
 

def _process_future_trajectories_for_windowed_agents(
    log_file: str,
    tracked_objects: List[TrackedObject],
    agent_indexes: Dict[int, Dict[str, int]],
    future_trajectory_sampling: TrajectorySampling,
) -> List[TrackedObject]:
    """
    A helper method to interpolate and parse the future trajectories for windowed agents.
    :param log_file: The log file to query.
    :param tracked_objects: The tracked objects to parse.
    :param agent_indexes: A mapping of [timestamp, [track_token, tracked_object_idx]]
    :param future_trajectory_sampling: The future trajectory sampling to use for future waypoints.
    :return: The tracked objects with predicted trajectories included.
    """
    agent_future_trajectories: Dict[int, Dict[str, List[Waypoint]]] = {}
    for timestamp in agent_indexes:
        agent_future_trajectories[timestamp] = {}

        for token in agent_indexes[timestamp]:
            agent_future_trajectories[timestamp][token] = []

    for timestamp_time in agent_future_trajectories:
        end_time = timestamp_time + int(1e6 * (future_trajectory_sampling.time_horizon + future_trajectory_sampling.interval_length))

        # TODO: This is somewhat inefficient because the resampling should happen in SQL layer

        for track_token, waypoint in get_future_waypoints_for_agents_from_db(
            log_file, list(agent_indexes[timestamp_time].keys()), timestamp_time, end_time
        ):
            agent_future_trajectories[timestamp_time][track_token].append(waypoint)

    for timestamp in agent_future_trajectories:
        for key in agent_future_trajectories[timestamp]:
            # We can only interpolate waypoints if there is more than one in the future.
            if len(agent_future_trajectories[timestamp][key]) == 1:
                tracked_objects[agent_indexes[timestamp][key]]._predictions = [PredictedTrajectory(1.0, agent_future_trajectories[timestamp][key])]
            elif len(agent_future_trajectories[timestamp][key]) > 1:
                tracked_objects[agent_indexes[timestamp][key]]._predictions = [
                    PredictedTrajectory(
                        1.0,
                        interpolate_future_waypoints(
                            agent_future_trajectories[timestamp][key],
                            future_trajectory_sampling.time_horizon,
                            future_trajectory_sampling.interval_length,
                        ),
                    )
                ]

    return tracked_objects


def extract_tracked_objects_within_time_window(
    token: str,
    log_file: str,
    past_time_horizon: float,
    future_time_horizon: float,
    filter_track_tokens: Optional[Set[str]] = None,
    future_trajectory_sampling: Optional[TrajectorySampling] = None,
) -> TrackedObjects:
    """
    Extracts the tracked objects in a time window centered on a token.
    :param token: The token on which to center the time window.
    :param past_time_horizon: The time in the past for which to search.
    :param future_time_horizon: The time in the future for which to search.
    :param filter_track_tokens: If provided, objects with track_tokens missing from the set will be excluded.
    :param future_trajectory_sampling: If provided, the future trajectory sampling to use for future waypoints.
    :return: The retrieved TrackedObjects.
    """
    tracked_objects: List[TrackedObject] = []
    agent_indexes: Dict[int, Dict[str, int]] = {}

    token_timestamp = get_sensor_data_token_timestamp_from_db(log_file, get_lidarpc_sensor_data(), token)
    start_time = int(token_timestamp - (1e6 * past_time_horizon))
    end_time = int(token_timestamp + (1e6 * future_time_horizon))

    for idx, tracked_object in enumerate(get_tracked_objects_within_time_interval_from_db(log_file, start_time, end_time, filter_track_tokens)):
        if future_trajectory_sampling and isinstance(tracked_object, Agent):
            if tracked_object.metadata.timestamp_us not in agent_indexes:
                agent_indexes[tracked_object.metadata.timestamp_us] = {}

            agent_indexes[tracked_object.metadata.timestamp_us][tracked_object.metadata.track_token] = idx
        tracked_objects.append(tracked_object)

    if future_trajectory_sampling:
        _process_future_trajectories_for_windowed_agents(log_file, tracked_objects, agent_indexes, future_trajectory_sampling)

    return TrackedObjects(tracked_objects=tracked_objects)


def extract_tracked_objects(
    token: str,
    log_file: str,
    future_trajectory_sampling: Optional[TrajectorySampling] = None,
) -> TrackedObjects:
    """
    Extracts all boxes from a lidarpc.
    :param lidar_pc: Input lidarpc.
    :param future_trajectory_sampling: If provided, the future trajectory sampling to use for future waypoints.
    :return: Tracked objects contained in the lidarpc.
    """
    tracked_objects: List[TrackedObject] = []
    agent_indexes: Dict[str, int] = {}
    agent_future_trajectories: Dict[str, List[Waypoint]] = {}

    for idx, tracked_object in enumerate(get_tracked_objects_for_lidarpc_token_from_db(log_file, token)):
        if future_trajectory_sampling and isinstance(tracked_object, Agent):
            agent_indexes[tracked_object.metadata.track_token] = idx
            agent_future_trajectories[tracked_object.metadata.track_token] = []
        tracked_objects.append(tracked_object)

    if future_trajectory_sampling and len(tracked_objects) > 0:
        timestamp_time = get_sensor_data_token_timestamp_from_db(log_file, get_lidarpc_sensor_data(), token)
        end_time = timestamp_time + int(1e6 * (future_trajectory_sampling.time_horizon + future_trajectory_sampling.interval_length))

        # TODO: This is somewhat inefficient because the resampling should happen in SQL layer
        for track_token, waypoint in get_future_waypoints_for_agents_from_db(log_file, list(agent_indexes.keys()), timestamp_time, end_time):
            agent_future_trajectories[track_token].append(waypoint)

        for key in agent_future_trajectories:
            # We can only interpolate waypoints if there is more than one in the future.
            if len(agent_future_trajectories[key]) == 1:
                tracked_objects[agent_indexes[key]]._predictions = [PredictedTrajectory(1.0, agent_future_trajectories[key])]
            elif len(agent_future_trajectories[key]) > 1:
                tracked_objects[agent_indexes[key]]._predictions = [
                    PredictedTrajectory(
                        1.0,
                        interpolate_future_waypoints(
                            agent_future_trajectories[key],
                            future_trajectory_sampling.time_horizon,
                            future_trajectory_sampling.interval_length,
                        ),
                    )
                ]

    return TrackedObjects(tracked_objects=tracked_objects)


def absolute_path_to_log_name(absolute_path: str) -> str:
    """
    Gets the log name from the absolute path to a log file.
    E.g.
        input: data/sets/devkit/devkit-v1.1/splits/mini/2021.10.11.02.57.41_veh-50_01522_02088.db
        output: 2021.10.11.02.57.41_veh-50_01522_02088

        input: /tmp/abcdef
        output: abcdef
    :param absolute_path: The absolute path to a log file.
    :return: The log name.
    """
    filename = os.path.basename(absolute_path)

    # Files generated during caching do not end with ".db"
    # They have no extension.
    if filename.endswith(".db"):
        filename = os.path.splitext(filename)[0]
    return filename
