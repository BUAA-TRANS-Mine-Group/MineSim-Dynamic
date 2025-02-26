from collections import deque
from dataclasses import dataclass
import logging
from typing import Deque, Dict, List, Optional

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import CAP_STYLE
from shapely.ops import unary_union
from shapely.geometry import LineString

from devkit.common.actor_state.agent import Agent, PredictedTrajectory
from devkit.common.actor_state.oriented_box import OrientedBox
from devkit.common.actor_state.scene_object import SceneObjectMetadata
from devkit.common.actor_state.state_representation import ProgressStateSE2
from devkit.common.actor_state.state_representation import StateSE2
from devkit.common.actor_state.state_representation import StateVector2D
from devkit.common.actor_state.state_representation import TimePoint
from devkit.common.actor_state.tracked_objects_types import TrackedObjectType
from devkit.common.actor_state.waypoint import Waypoint
from devkit.common.actor_state.vehicle_parameters import get_vehicle_waypoint_speed_limit_data

from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import TrafficLightStatusType
from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphEdgeRefPathMapObject
from devkit.sim_engine.observation_manager.agent_update_policy.idm.idm_policy import IDMPolicy
from devkit.sim_engine.observation_manager.agent_update_policy.idm.idm_states import IDMAgentState, IDMLeadAgentState
from devkit.sim_engine.observation_manager.agent_update_policy.idm.utils import path_to_linestring
from devkit.sim_engine.path.interpolated_path import InterpolatedPath
from devkit.sim_engine.path.utils import trim_path
from devkit.sim_engine.path.utils import trim_path_up_to_progress

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IDMInitialState:
    """Initial state of IDMAgent."""

    metadata: SceneObjectMetadata
    tracked_object_type: TrackedObjectType
    box: OrientedBox
    velocity: StateVector2D  #  车辆几何中心的加速度，纵向和横向加速度分量。【 右手坐标系，x-纵向 向前为正，y-横向 向左为正】
    path_progress: float
    predictions: Optional[List[PredictedTrajectory]]


class IDMAgent:
    """IDM smart-agent."""

    def __init__(
        self,
        start_iteration: int,
        initial_state: IDMInitialState,
        path: InterpolatedPath,
        policy: IDMPolicy,
        # minimum_path_length: float = 200,
        # max_route_len: int = 5,
    ):
        """
        Constructor for IDMAgent.
        :param start_iteration: scenario iteration where agent first appeared
        :param initial_state: agent initial state
        :param policy: policy controlling the agent behavior
        # :param minimum_path_length: [m] The minimum path length ; max_waypoints_number=1000 ;200m
        # :param max_route_len: The max number of route elements to store
        """
        self._start_iteration = start_iteration  # scenario iteration where agent first appears
        self._initial_state = initial_state
        self._state = IDMAgentState(initial_state.path_progress, initial_state.velocity.x)
        self._path = path
        self._state.progress = 0
        self._policy = policy
        # self._minimum_path_length = minimum_path_length
        self._size = (initial_state.box.width, initial_state.box.length, initial_state.box.height)

        # This variable is used to trigger when the _full_agent_state needs to be recalculated  此变量用于在需要重新计算 _full_agent_state 时触发
        self._requires_state_update: bool = True
        self._full_agent_state: Optional[Agent] = None

        # 计算 agent 匹配路径的限速信息；
        max_lateral_accel, max_speed_limit, min_speed_limit = get_vehicle_waypoint_speed_limit_data(vehicle_width=self.agent.box.width)
        self._path.compute_speed_limits(max_lateral_accel=max_lateral_accel, max_speed=max_speed_limit, min_speed=min_speed_limit)

    def propagate(self, lead_agent: IDMLeadAgentState, tspan: float) -> None:
        """
        Propagate agent forward according to the IDM policy with dynamic speed limit lookup.

        说明：
        1. 前瞻距离计算：基于车辆当前速度与预瞄时间（headway_time）
        2. 实时查询路径限速：根据前瞻后的路径位置获取限速值
        3. 异常处理：限速不可用时回退到策略默认值
        """
        # 计算动态前瞻距离 = 速度 × 预瞄时间 + 固定缓冲
        lookahead_time = self._policy.headway_time  # 默认预瞄时间（例如2.5秒）
        # lookahead_distance = self._state.velocity * lookahead_time + 10.0  # 增加10m固定缓冲
        lookahead_distance = self._state.velocity * lookahead_time + self.agent.box.length  # 增加 vcehicle legth 固定缓冲

        # 计算前瞻路径进度（限制在路径范围内）
        lookahead_progress = self._clamp_progress(self._state.progress + lookahead_distance)

        try:
            # 实时查询限速值
            speed_limit = self._path.get_speed_limit_at_progress(lookahead_progress)
            # 应用限速（不低于最小速度）
            # self._policy.target_velocity = max(speed_limit, self._policy.min_speed)
            self._policy.target_velocity = speed_limit
        except RuntimeError as e:
            logger.warning(f"Speed limit lookup failed: {str(e)}, using default policy speed.")

        # IDM求解逻辑
        solution = self._policy.solve_forward_euler_idm_policy(
            agent=IDMAgentState(progress=0, velocity=self._state.velocity), lead_agent=lead_agent, sampling_time=tspan
        )
        self._state.progress += solution.progress
        self._state.velocity = max(solution.velocity, 0)
        self._requires_state_update = True

    @property
    def agent(self) -> Agent:
        """:return: the agent as a Agent object"""
        return self._get_agent_at_progress(progress=self._get_bounded_progress())

    @property
    def polygon(self) -> Polygon:
        """:return: the box polygon of the agent as a Agent object"""
        return self.agent.box.geometry

    @property
    def path(self) -> InterpolatedPath:
        """:return: the most matched route path of the agent object"""
        return self._path

    @property
    def projected_footprint(self) -> Polygon:
        """
        Returns the agent's projected footprint along it's planned path. The extended length is proportional
        to it's current velocity
        :return: The agent's projected footprint as a Polygon.
        """
        start_progress = self._clamp_progress(progress=self.progress - self.length / 2)
        end_progress = self._clamp_progress(progress=self.progress + self.length / 2 + self.velocity * self._policy.headway_time)
        projected_path = path_to_linestring(path=trim_path(path=self._path, start=start_progress, end=end_progress))
        return unary_union([projected_path.buffer(distance=(self.width * 0.55), cap_style=CAP_STYLE.flat), self.polygon])

    @property
    def width(self) -> float:
        """:return: [m] agent's width"""
        return float(self._initial_state.box.width)

    @property
    def length(self) -> float:
        """:return: [m] agent's length"""
        return float(self._initial_state.box.length)

    @property
    def progress(self) -> float:
        """:return: [m] agent's progress"""
        return self._state.progress  # type: ignore

    @property
    def velocity(self) -> float:
        """:return: [m/s] agent's velocity along the path"""
        return self._state.velocity  # type: ignore

    def to_se2(self) -> StateSE2:
        """
        :return: the agent as a StateSE2 object
        """
        return self._get_agent_at_progress(progress=self._get_bounded_progress()).box.center

    def is_active(self, iteration: int) -> bool:
        """
        Return if the agent should be active at a simulation iteration

        :param iteration: the current simulation iteration
        :return: true if active, false otherwise
        """
        return self._start_iteration <= iteration

    def has_valid_path(self) -> bool:
        """
        :return: true if agent has a valid path, false otherwise
        """
        return self._path is not None

    def _get_bounded_progress(self) -> float:
        """
        :return: [m] The agent's progress. The progress is clamped between the start and end progress of it's path
        """
        return self._clamp_progress(progress=self._state.progress)

    def get_path_to_go(self) -> List[ProgressStateSE2]:
        """
        :return: The agent's path trimmed to start at the agent's current progress
        """
        return trim_path_up_to_progress(self._path, self._get_bounded_progress())  # type: ignore

    def get_progress_to_go(self) -> float:
        """
        return: [m] the progress left until the end of the path
        """
        return self._path.get_end_progress() - self.progress  # type: ignore

    def get_agent_with_planned_trajectory(self, num_samples: int, sampling_time: float) -> Agent:
        """
        Samples the the agent's trajectory. The velocity is assumed to be constant over the sampled trajectory
        :param num_samples: number of elements to sample.
        :param sampling_time: [s] time interval of sequence to sample from.
        :return: the agent's trajectory as a list of Agent

        对代理的轨迹进行采样。假设速度在采样轨迹上是恒定的
        :p aram num_samples：要采样的元素 数number。
        :p aram sampling_time： [s] 要从中采样的序列的时间间隔。
        ：return： Agent 的轨迹作为 Agent 列表
        """
        # _progress=self._get_bounded_progress()
        # _agent = self._get_agent_at_progress(progress=_progress,num_samples=num_samples,sampling_time=sampling_time)
        # return _agent
        return self._get_agent_at_progress(progress=self._get_bounded_progress(), num_samples=num_samples, sampling_time=sampling_time)

    def _get_agent_at_progress(self, progress: float, num_samples: Optional[int] = None, sampling_time: Optional[float] = None) -> Agent:
        """
        Returns the agent as a box at a given progress
        :param progress: the arc length along the agent's path
        :return: the agent as a Agent object at the given progress

        NOTE： Agent 基于 IDM 更新状态，无以下几项：
            - angular_velocity: Optional[float] = None,
            - acceleration: Optional[StateVector2D] = None,
            - predictions: Optional[List[PredictedTrajectory]] = None,
            - past_trajectory: Optional[PredictedTrajectory] = None,
        """
        # Caching
        if not self._requires_state_update:
            return self._full_agent_state

        if self._path is not None:
            init_pose = self._path.get_state_at_progress(progress)
            box = OrientedBox.from_new_pose(box=self._initial_state.box, pose=StateSE2(init_pose.x, init_pose.y, init_pose.heading))
            future_trajectory = None

            if num_samples and sampling_time:
                progress_samples = [self._clamp_progress(progress + self.velocity * sampling_time * (step + 1)) for step in range(num_samples)]
                future_poses = [self._path.get_state_at_progress(progress) for progress in progress_samples]
                time_stamps = [TimePoint(int(1e6 * sampling_time * (step + 1))) for step in range(num_samples)]
                init_way_point = [Waypoint(time_point=TimePoint(0), oriented_box=box, velocity=self._velocity_to_global_frame(init_pose.heading))]
                waypoints = [
                    Waypoint(
                        time_point=time,
                        oriented_box=OrientedBox.from_new_pose(box=self._initial_state.box, pose=pose),
                        velocity=self._velocity_to_global_frame(pose.heading),
                    )
                    for time, pose in zip(time_stamps, future_poses)
                ]
                future_trajectory = PredictedTrajectory(probability=1.0, waypoints=init_way_point + waypoints)

            self._full_agent_state = Agent(
                metadata=self._initial_state.metadata,
                oriented_box=box,
                velocity=StateVector2D(x=self.velocity, y=0.0),
                tracked_object_type=self._initial_state.tracked_object_type,
                predictions=[future_trajectory] if future_trajectory is not None else [],
            )

        else:
            self._full_agent_state = Agent(
                metadata=self._initial_state.metadata,
                oriented_box=self._initial_state.box,
                velocity=self._initial_state.velocity,
                tracked_object_type=self._initial_state.tracked_object_type,
                predictions=self._initial_state.predictions,
            )
        self._requires_state_update = False
        return self._full_agent_state

    def _clamp_progress(self, progress: float) -> float:
        """
        Clamp the progress to be between the agent's path bounds
        :param progress: [m] the progress along the agent's path
        :return: [m] the progress clamped between the start and end progress of the agent's path
        """
        return max(self._path.get_start_progress(), (min(progress, self._path.get_end_progress())))  # type: ignore

    def _velocity_to_global_frame(self, heading: float) -> StateVector2D:
        """
        Transform agent's velocity along the path to global frame
        :param heading: [rad] The heading defining the transform to global frame.
        :return: The velocity vector in global frame.
        将 agent 沿着 path 的纵向速度，变换为全局坐标系的 X速度，Y速度；
        """
        return StateVector2D(x=self.velocity * np.cos(heading), y=self.velocity * np.sin(heading))
