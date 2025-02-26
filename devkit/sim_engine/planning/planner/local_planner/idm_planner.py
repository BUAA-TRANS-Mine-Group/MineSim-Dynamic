import logging
import math
from typing import List, Tuple

from devkit.common.actor_state.ego_state import EgoState
from devkit.common.trajectory.abstract_trajectory import AbstractTrajectory
from devkit.sim_engine.planning.planner.abstract_idm_planner import AbstractIDMPlanner
from devkit.sim_engine.planning.planner.abstract_planner import PlannerInitialization
from devkit.sim_engine.planning.planner.abstract_planner import PlannerInput

logger = logging.getLogger(__name__)


class IDMPlanner(AbstractIDMPlanner):
    """
    The IDM planner is composed of two parts:
        1. Path planner that constructs a route to the goal pose.
        2. IDM policy controller to control the longitudinal movement of the ego along the planned route.
    """

    # Inherited property, see superclass.
    requires_scenario: bool = False
    requires_init_ego_vehicle_parameters: bool = False

    def __init__(
        self,
        target_velocity: float,
        min_gap_to_lead_agent: float,
        headway_time: float,
        accel_max: float,
        decel_max: float,
        planned_trajectory_samples: int,
        planned_trajectory_sample_interval: float,
        occupancy_map_radius: float,
        truck_lateral_expansion_factor: float,
    ):
        """
        Constructor for IDMPlanner
        :param target_velocity: [m/s] Desired velocity in free traffic.
        :param min_gap_to_lead_agent: [m] Minimum relative distance to lead vehicle.
        :param headway_time: [s] Desired time headway. The minimum possible time to the vehicle in front.
        :param accel_max: [m/s^2] maximum acceleration.
        :param decel_max: [m/s^2] maximum deceleration (positive value).
        :param planned_trajectory_samples: number of elements to sample for the planned trajectory.
        :param planned_trajectory_sample_interval: [s] time interval of sequence to sample from.
        :param occupancy_map_radius: [m] The range around the ego to add objects to be considered.
        """
        super(IDMPlanner, self).__init__(
            target_velocity=target_velocity,
            min_gap_to_lead_agent=min_gap_to_lead_agent,
            headway_time=headway_time,
            accel_max=accel_max,
            decel_max=decel_max,
            planned_trajectory_samples=planned_trajectory_samples,
            planned_trajectory_sample_interval=planned_trajectory_sample_interval,
            occupancy_map_radius=occupancy_map_radius,
            truck_lateral_expansion_factor=truck_lateral_expansion_factor,
        )

        self._initialized = False
        logger.info("#log# Created a IMDPlanner.")

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass.
        初始化
        """
        self.scenario_name = initialization.scenario_name
        self._map_api = initialization.map_api
        self._initialize_search_ego_route_path(
            initial_ego_state=initialization.initial_ego_state, goal_task=initialization.planning_problem_goal_task
        )

        logger.info("#log# Initialized a IMDPlanner: initialize ego_route_planner and search ego_route_path.")
        self._initialized = True

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Inherited, see superclass."""
        # Ego current state
        ego_state, observations = current_input.history.current_state

        # Create occupancy map: 记录
        occupancy_map, unique_observations = self._construct_occupancy_map(ego_state, observations)
        # logger.info("#log# Constructed occupancy_map using observations objects inside IMDPlanner._occupancy_map_radius.")

        # Traffic light handling
        # traffic_light_data = current_input.traffic_light_data
        # self._annotate_occupancy_map(traffic_light_data, occupancy_map)

        return self._get_planned_trajectory(ego_state, occupancy_map, unique_observations)
