from __future__ import annotations

import abc
import logging
from typing import Generator, List, Optional, Set

from devkit.common.actor_state.ego_state import EgoState
from devkit.common.actor_state.state_representation import TimePoint
from devkit.common.actor_state.vehicle_parameters import VehicleParameters
from devkit.sim_engine.map_manager.abstract_map import AbstractMap
from devkit.sim_engine.observation_manager.observation_type import DetectionsTracks
from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTasks
from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTask
from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTaskFinalPose

logger = logging.getLogger(__name__)


class AbstractScenario(abc.ABC):
    """Interface for a generic scenarios from any database."""

    @property
    @abc.abstractmethod
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """get "planning_problem" : "ego_info"
        Query the vehicle parameters of ego
        :return: VehicleParameters struct.
        """
        pass

    @property
    @abc.abstractmethod
    def log_name(self) -> str:
        """
        Log name for from which this scenario was created
        :return: str representing log name.
        """
        pass

    @property
    @abc.abstractmethod
    def scenario_name(self) -> str:
        """
        Name of this scenario, e.g. extraction_xxxx
        :return: str representing name of this scenario.
        """
        pass

    @property
    @abc.abstractmethod
    def scenario_type(self) -> str:
        """
        # :return: type of scenario e.g. [lane_change, lane_follow, ...].
        :return: type of scenario e.g. [mine_intersection_mix, mine_static_obstacle, ...].
        """
        pass

    @property
    @abc.abstractmethod
    def map_api(self) -> AbstractMap:
        """
        Return the Map API for this scenario
        :return: AbstractMap.
        """
        pass

    @property
    @abc.abstractmethod
    def database_interval(self) -> float:
        """
        Database interval in seconds
        :return: [s] database interval.
        """
        pass

    @abc.abstractmethod
    def get_time_point(self, iteration: int) -> TimePoint:
        """
        Get time point of the iteration
        :param iteration: iteration in scenario 0 <= iteration < number_of_iterations
        :return: global time point.
        """
        pass

    @abc.abstractmethod
    def get_number_of_iterations(self) -> int:
        """
        Get how many frames does this scenario contain
        :return: [int] representing number of scenarios.
        """
        pass

    @abc.abstractmethod
    def get_tracked_objects_at_iteration(self, iteration: int) -> DetectionsTracks:
        """
        Return tracked objects from iteration
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :param future_trajectory_sampling: sampling parameters of agent future ground truth predictions if desired.
        :return: DetectionsTracks.

        从迭代中返回跟踪对象
        :param iteration: 在场景0 <=迭代< number_of_iterations
        :param future_trajectory_sampling：如果需要，agent未来地面真值预测的采样参数。
        :return: DetectionsTracks。
        """
        pass

    @property
    def initial_tracked_objects(self) -> DetectionsTracks:
        """
        Get initial tracked objects
        :return: DetectionsTracks.
        """
        return self.get_tracked_objects_at_iteration(0)

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """
        Return ego (expert) state in a dataset
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return: EgoState of ego.
        """
        """Inherited, see superclass."""
        if iteration > 1:
            logging.warning("#log# 目前ego的未来轨迹没有保存，后续改进！")
        return None

    @property
    def initial_ego_state(self) -> EgoState:
        """
        Return the initial ego state
        :return: EgoState of ego.
        """
        pass

    @property
    def planning_problem_goal_task(self) -> PlanningProblemGoalTasks:
        """PlanningProblemGoalTaskFinalPose
        Return the initial ego state
        :return: EgoState of ego.
        """
        pass
