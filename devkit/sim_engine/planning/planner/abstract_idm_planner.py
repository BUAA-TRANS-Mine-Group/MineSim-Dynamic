import logging
from abc import ABC
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import CAP_STYLE
from shapely.ops import unary_union


from devkit.common.actor_state.agent import Agent
from devkit.common.actor_state.ego_state import EgoState
from devkit.common.actor_state.scene_object import SceneObject
from devkit.common.actor_state.state_representation import StateSE2
from devkit.common.actor_state.state_representation import StateVector2D
from devkit.common.actor_state.state_representation import TimePoint
from devkit.common.actor_state.vehicle_parameters import VehicleParameters
from devkit.common.geometry.transform import transform
from devkit.common.trajectory.interpolated_trajectory import InterpolatedTrajectory
from devkit.metrics_tool.utils.expert_comparisons import principal_value

from devkit.sim_engine.map_manager.abstract_map import AbstractMap
from devkit.sim_engine.map_manager.occupancy_map.abstract_occupancy_map import OccupancyMap
from devkit.sim_engine.map_manager.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMapFactory
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import SemanticMapLayer
from devkit.sim_engine.path.path import AbstractPath
from devkit.sim_engine.path.utils import trim_path
from devkit.sim_engine.planning.planner.abstract_planner import AbstractPlanner
from devkit.sim_engine.planning.planner.route_planner.global_route_path_planner import GlobalRoutePathPlanner
from devkit.sim_engine.planning.planner.route_planner.global_route_path_planner import create_route_path_planner
from devkit.sim_engine.planning.planner.route_planner.utils.minesim_graph import create_minesim_graph

from devkit.sim_engine.observation_manager.agent_update_policy.idm.idm_policy import IDMPolicy
from devkit.sim_engine.observation_manager.agent_update_policy.idm.idm_states import IDMAgentState
from devkit.sim_engine.observation_manager.agent_update_policy.idm.idm_states import IDMLeadAgentState
from devkit.sim_engine.observation_manager.agent_update_policy.idm.utils import path_to_linestring
from devkit.sim_engine.observation_manager.agent_update_policy.idm.utils import create_path_from_se2

from devkit.sim_engine.observation_manager.observation_type import DetectionsTracks
from devkit.sim_engine.observation_manager.observation_type import Observation
from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTasks


UniqueObjects = Dict[str, SceneObject]

logger = logging.getLogger(__name__)


class AbstractIDMPlanner(AbstractPlanner, ABC):
    """
    An interface for IDM based planners. Inherit from this class to use IDM policy to control the longitudinal
    behaviour of the ego.
    """

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

        - 注意： 因为矿车较宽，增加一个侧向宽度膨胀系数, [m] float; 大于1.0;
        - Note: Due to the width of the mine truck, a lateral width expansion factor is added.
        """
        self._policy = IDMPolicy(target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max)
        self._planned_trajectory_samples = planned_trajectory_samples
        self._planned_trajectory_sample_interval = planned_trajectory_sample_interval
        self._planned_horizon = planned_trajectory_samples * planned_trajectory_sample_interval
        self._occupancy_map_radius = occupancy_map_radius
        assert truck_lateral_expansion_factor > 0, "truck_lateral_expansion_factor 必须为正数"  # 参数校验
        self._truck_lateral_expansion_factor = truck_lateral_expansion_factor

        self._max_path_length = self._policy.target_velocity * self._planned_horizon
        self._ego_token = "ego_token"

        # To be lazy loaded
        self._candidate_lane_edge_ids: Optional[List[str]] = None
        self._map_api: Optional[AbstractMap] = None  # simple_planner is None

        # To be lazy loaded
        self._route_path_planner: Optional[GlobalRoutePathPlanner] = None  # simple_planner is None
        # To be intialized by inherited classes
        self._ego_path: Optional[AbstractPath] = None
        self._ego_path_linestring: Optional[LineString] = None

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def _initialize_search_ego_route_path(self, initial_ego_state: EgoState, goal_task: PlanningProblemGoalTasks):
        """Initializes the ego route path from mine map;"""
        assert self._map_api, "_map_api has not yet been initialized. Please call the initialize() function first!"
        # minesim_graph = create_minesim_graph(map_name=self._map_api.map_name, semantic_map=self._map_api.semantic_map)
        if self._route_path_planner is None:
            self._route_path_planner = create_route_path_planner(
                map_name=self._map_api.map_name,
                semantic_map=self._map_api.semantic_map,
                minesim_graph=create_minesim_graph(map_name=self._map_api.map_name, semantic_map=self._map_api.semantic_map),
                is_initial_path_stitched=True,
            )

        # 搜索路径
        self._route_path_planner.get_ego_global_route_path(
            start_pose_statese2=StateSE2(
                x=initial_ego_state.rear_axle.x,
                y=initial_ego_state.rear_axle.y,
                heading=initial_ego_state.rear_axle.heading,
            ),
            goal_task=goal_task,
            scenario_name=self.scenario_name,
        )
        # 检查 refline_smooth 是否存在并且维度正确
        if self._route_path_planner.refline_smooth is None or self._route_path_planner.refline_smooth.shape[1] < 3:
            logging.error("#log# refline_smooth is not properly generated or lacks required columns.")
            raise ValueError("Error: refline_smooth data is missing or incomplete.")

        # 使用 refline_smooth 生成离散路径
        discrete_path = [StateSE2(x=waypoint[0], y=waypoint[1], heading=waypoint[2]) for waypoint in self._route_path_planner.refline_smooth]

        self._ego_path = create_path_from_se2(discrete_path)
        self._ego_path_linestring = path_to_linestring(discrete_path)

        # 计算最大速度限制;
        # ego_speed = initial_ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()
        # todo 原则上 可以根据参考路径曲率来计算 每个路径点的限速 ；[制作新地图path后再增加]
        # todo 根据path的综合曲率计算 waypoint点限速值；
        # speed_limit = route_plan[0].speed_limit_mps or self._policy.target_velocity
        # self._policy.target_velocity = speed_limit if speed_limit > ego_speed else ego_speed
        # 12.5 (45 Km/h)
        self._policy.target_velocity = 10.5

    def _get_expanded_ego_path(self, ego_state: EgoState, ego_idm_state: IDMAgentState) -> Polygon:
        """
        Returns the ego's expanded path as a Polygon.
        :return: A polygon representing the ego's path.

        #!将 ego 参考路径的waypoints 根据自车宽度扩展简单的扩展当前可达路径段作为 Polygon 返回。
        - 从 ego 车辆的参考路径中截取当前可达路径段。
        - 将路径转换为几何对象，并根据车辆宽度扩展，形成路径占用区域。
        - 返回 ego 车辆占用区域的多边形。
        - 注意： 因为矿车较宽，增加一个侧向宽度膨胀系数
        """
        assert self._ego_path, "_ego_path has not yet been initialized. Please call the initialize() function first!"
        ego_footprint = ego_state.car_footprint
        # 根据 自车参考路径 截取 当前ego位置及最远到达的位置的路径段---> start距离值， end距离值
        path_to_go = trim_path(
            path=self._ego_path,
            start=max(self._ego_path.get_start_progress(), min(ego_idm_state.progress, self._ego_path.get_end_progress())),
            end=max(
                self._ego_path.get_start_progress(),
                min(
                    ego_idm_state.progress + abs(self._policy.target_velocity) * self._planned_horizon,
                    self._ego_path.get_end_progress(),
                ),
            ),
        )

        # 将裁剪后的路径转换为 LineString，并基于车辆宽度进行 buffer 扩展，生成占用区域; #! expansion ratio
        # expanded_path = path_to_linestring(path=path_to_go).buffer((ego_footprint.width / 2), cap_style=CAP_STYLE.square)
        path_linestring: LineString = path_to_linestring(path=path_to_go)

        expanded_path = path_linestring.buffer(distance=(ego_footprint.width / 2 * self._truck_lateral_expansion_factor), cap_style=CAP_STYLE.square)

        # 合并扩展路径和车辆当前占用区域，返回最终的占用区域
        return unary_union([expanded_path, ego_state.car_footprint.geometry])

    @staticmethod
    def _get_leading_idm_agent(ego_state: EgoState, agent: SceneObject, relative_distance: float) -> IDMLeadAgentState:
        """返回表示另一个静态和动态代理的潜在 lead IDM 代理状态。
        Returns a lead IDM agent state that represents another static and dynamic agent.
        :param agent: A scene object.
        :param relative_distance: [m] The relative distance from the scene object to the ego.
        :return: A IDM lead agents state
        """
        if isinstance(agent, Agent):
            # Dynamic object
            longitudinal_velocity = agent.velocity.magnitude()
            # Wrap angle to [-pi, pi]
            relative_heading = principal_value(angle=(agent.center.heading - ego_state.center.heading))
            projected_velocity = transform(
                pose=StateSE2(longitudinal_velocity, 0, 0), transform_matrix=StateSE2(0, 0, relative_heading).as_matrix()
            ).x
        else:
            # Static object
            projected_velocity = 0.0

        return IDMLeadAgentState(progress=relative_distance, velocity=projected_velocity, length_rear=0.0)

    def _get_free_road_leading_idm_state(self, ego_state: EgoState, ego_idm_state: IDMAgentState) -> IDMLeadAgentState:
        """当没有leading代理时，返回lead IDM 代理状态。
        Returns a lead IDM agent state when there is no leading agent.
        :return: A IDM lead agents state.
        """
        assert self._ego_path, "_ego_path has not yet been initialized. Please call the initialize() function first!"
        projected_velocity = 0.0
        relative_distance = self._ego_path.get_end_progress() - ego_idm_state.progress
        length_rear = ego_state.car_footprint.length / 2
        return IDMLeadAgentState(progress=relative_distance, velocity=projected_velocity, length_rear=length_rear)

    def _get_leading_object(
        self, ego_idm_state: IDMAgentState, ego_state: EgoState, occupancy_map: OccupancyMap, unique_observations: UniqueObjects
    ) -> IDMLeadAgentState:
        """
        Get the most suitable leading object based on the occupancy map.
        :param ego_idm_state: The ego's IDM state at current iteration.
        :param ego_state: EgoState at current iteration.
        :param occupancy_map: OccupancyMap containing all objects in the scene.
        :param unique_observations: A mapping between the object token and the object itself.

        根据占用地图获取最合适的 leading 对象。
        :p aram ego_idm_state：自我在当前迭代时的 IDM 状态。
        :p aram ego_state：当前迭代的 EgoState。
        :p aram occupancy_map：包含场景中所有对象的 OccupancyMap。
        :p aram unique_observations：对象 Token 和对象本身之间的映射。
        """
        # 计算交互的 agents
        # intersecting_agents = occupancy_map.intersects(geometry=self._get_expanded_ego_path(ego_state, ego_idm_state))
        path_geometry = self._get_expanded_ego_path(ego_state=ego_state, ego_idm_state=ego_idm_state)
        intersecting_agents = occupancy_map.intersects(geometry=path_geometry)
        # Check if there are agents intersecting the ego's baseline
        if intersecting_agents.size > 0:
            # Extract closest object
            intersecting_agents.insert(geometry_id=self._ego_token, geometry=ego_state.car_footprint.geometry)
            nearest_id, nearest_agent_polygon, relative_distance = intersecting_agents.get_nearest_entry_to(self._ego_token)
            # An agent is the leading agent
            return self._get_leading_idm_agent(ego_state=ego_state, agent=unique_observations[nearest_id], relative_distance=relative_distance)

        else:
            # No leading agent
            return self._get_free_road_leading_idm_state(ego_state, ego_idm_state)

    def _construct_occupancy_map(self, ego_state: EgoState, observation: Observation) -> Tuple[OccupancyMap, UniqueObjects]:
        """
        Constructs an OccupancyMap from Observations.
        :param ego_state: Current EgoState
        :param observation: Observations of other agents and static objects in the scene.
        :return:
            - OccupancyMap.
            - A mapping between the object token and the object itself.
           记录self._occupancy_map_radius 范围内所有的 感知检测 agent 的 box 几何系列， 基于 from shapely.strtree import STRtree 高效查询；
        """
        if isinstance(observation, DetectionsTracks):
            unique_observations = {
                detection.track_token: detection
                for detection in observation.tracked_objects.tracked_objects
                if np.linalg.norm(ego_state.center.array - detection.center.array) < self._occupancy_map_radius
            }
            # unique_observations = {}
            # for detection in observation.tracked_objects.tracked_objects:
            #     if np.linalg.norm(ego_state.center.array - detection.center.array) < self._occupancy_map_radius:
            #         unique_observations[detection.track_token] = detection
            # a = 1

            return (
                STRTreeOccupancyMapFactory.get_from_boxes(scene_objects=list(unique_observations.values())),
                unique_observations,
            )
        else:
            raise ValueError(f"IDM planner only supports DetectionsTracks. Got {observation.detection_type()}")

    def _propagate(self, ego: IDMAgentState, lead_agent: IDMLeadAgentState, tspan: float) -> None:
        """
        Propagate agent forward according to the IDM policy.
        :param ego: The ego's IDM state.
        :param lead_agent: The agent leading this agent.
        :param tspan: [s] The interval of time to propagate for.
        """
        # TODO: Set target velocity to speed limit
        solution = self._policy.solve_forward_euler_idm_policy(
            agent=IDMAgentState(progress=0, velocity=ego.velocity), lead_agent=lead_agent, sampling_time=tspan
        )
        ego.progress += solution.progress
        ego.velocity = max(solution.velocity, 0)

    def _get_planned_trajectory(self, ego_state: EgoState, occupancy_map: OccupancyMap, unique_observations: UniqueObjects) -> InterpolatedTrajectory:
        """
        Plan a trajectory w.r.t. the occupancy map.
        :param ego_state: EgoState at current iteration.
        :param occupancy_map: OccupancyMap containing all objects in the scene.
        :param unique_observations: A mapping between the object token and the object itself.
        :return: A trajectory representing the predicted ego's position in future.

        根据占用地图规划轨迹。
        :p aram ego_state：当前迭代的 EgoState。
        :p aram occupancy_map：包含场景中所有对象的 OccupancyMap。
        :p aram unique_observations：对象 Token 和对象本身之间的映射。
        ：return： 代表预测的自我在未来位置的轨迹。
        """
        assert self._ego_path_linestring, "_ego_path_linestring has not yet been initialized. Please call the initialize() function first!"
        # Extract ego IDM state
        ego_progress = self._ego_path_linestring.project(Point(*ego_state.center.point.array))
        ego_idm_state = IDMAgentState(progress=ego_progress, velocity=ego_state.dynamic_car_state.center_velocity_2d.x)
        vehicle_parameters = ego_state.car_footprint.vehicle_parameters

        # Initialize planned trajectory with current state
        current_time_point = ego_state.time_point
        projected_ego_state = self._idm_state_to_ego_state(ego_idm_state, current_time_point, vehicle_parameters)
        planned_trajectory: List[EgoState] = [projected_ego_state]

        # Propagate planned trajectory for set number of samples
        for _ in range(self._planned_trajectory_samples):
            # Propagate IDM state w.r.t. selected leading agent
            leading_agent = self._get_leading_object(
                ego_idm_state=ego_idm_state, ego_state=ego_state, occupancy_map=occupancy_map, unique_observations=unique_observations
            )
            self._propagate(ego=ego_idm_state, lead_agent=leading_agent, tspan=self._planned_trajectory_sample_interval)

            # Convert IDM state back to EgoState
            current_time_point += TimePoint(int(self._planned_trajectory_sample_interval * 1e6))
            ego_state = self._idm_state_to_ego_state(idm_state=ego_idm_state, time_point=current_time_point, vehicle_parameters=vehicle_parameters)

            planned_trajectory.append(ego_state)

        return InterpolatedTrajectory(planned_trajectory)

    def _idm_state_to_ego_state(self, idm_state: IDMAgentState, time_point: TimePoint, vehicle_parameters: VehicleParameters) -> EgoState:
        """
        Convert IDMAgentState to EgoState
        :param idm_state: The IDMAgentState to be converted.
        :param time_point: The TimePoint corresponding to the state.
        :param vehicle_parameters: VehicleParameters of the ego.
        """
        assert self._ego_path, "_ego_path has not yet been initialized. Please call the initialize() function first!"

        new_ego_center = self._ego_path.get_state_at_progress(
            progress=max(self._ego_path.get_start_progress(), min(idm_state.progress, self._ego_path.get_end_progress()))
        )
        return EgoState.build_from_center(
            center=StateSE2(new_ego_center.x, new_ego_center.y, new_ego_center.heading),
            center_velocity_2d=StateVector2D(idm_state.velocity, 0),
            center_acceleration_2d=StateVector2D(0, 0),
            tire_steering_angle=0.0,
            time_point=time_point,
            vehicle_parameters=vehicle_parameters,
        )
