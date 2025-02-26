from collections import defaultdict
from typing import Dict, List, Optional, Type

from devkit.common.actor_state.tracked_objects import TrackedObject
from devkit.common.actor_state.tracked_objects_types import TrackedObjectType
from devkit.sim_engine.history.simulation_history_buffer import SimulationHistoryBuffer
from devkit.sim_engine.observation_manager.abstract_observation import AbstractObservation
from devkit.sim_engine.observation_manager.agent_update_policy.idm.idm_agent_manager import IDMAgentManager
from devkit.sim_engine.observation_manager.agent_update_policy.idm.idm_agents_builder import build_idm_agents_on_map_rails
from devkit.sim_engine.observation_manager.observation_type import DetectionsTracks, Observation
from devkit.sim_engine.simulation_time_controller.simulation_iteration import SimulationIteration
from devkit.scenario_builder.abstract_scenario import AbstractScenario


class IDMAgents(AbstractObservation):
    """
    Simulate agents based on IDM policy.
    """

    def __init__(
        self,
        target_velocity: float,
        min_gap_to_lead_agent: float,
        headway_time: float,
        accel_max: float,
        decel_max: float,
        open_loop_detections_types: List[str],
        scenario: AbstractScenario,
        minimum_path_length: float = 20,
        planned_trajectory_samples: Optional[int] = None,
        planned_trajectory_sample_interval: Optional[float] = None,
        radius: float = 100,
    ):
        """
        Constructor for IDMAgents

        :param target_velocity: [m/s] Desired velocity in free traffic
        :param min_gap_to_lead_agent: [m] Minimum relative distance to lead vehicle
        :param headway_time: [s] Desired time headway. The minimum possible time to the vehicle in front
        :param accel_max: [m/s^2] maximum acceleration
        :param decel_max: [m/s^2] maximum deceleration (positive value)
        :param scenario: scenario
        :param open_loop_detections_types: The open-loop detection types to include.
        :param minimum_path_length: [m] The minimum path length to maintain.
        :param planned_trajectory_samples: number of elements to sample for the planned trajectory.
        :param planned_trajectory_sample_interval: [s] time interval of sequence to sample from.
        :param radius: [m] Only agents within this radius around the ego will be simulated.

        IDMAgents 的构造函数
        :param target_velocity: [m/s] 自由交通中的期望速度
        :param min_gap_too_lead_agent: [m] 与领头车辆的最小相对距离
        :param headway_time： [s] 期望的时间 headway。距离前方车辆的最短时间。
        :param accel_max: [m/s^2] 最大加速度
        :param decel_max: [m/s^2] 最大减速度（正值）
        :param scenario: 场景信息
        :param open_loop_detections_types： 开环检测类型。
        :param minimum_path_length: [m] 要保持的最小路径长度。
        :param planned_trajectory_samples：规划轨迹的采样数。
        :param planned_trajectory_sample_interval： [s] 采样序列的时间间隔。
        :param radius： [m] 只有在此半径范围内的 agent 才会被模拟。
        """
        self.current_iteration = 0

        self._target_velocity = target_velocity
        self._min_gap_to_lead_agent = min_gap_to_lead_agent
        self._headway_time = headway_time
        self._accel_max = accel_max
        self._decel_max = decel_max
        self._scenario = scenario
        self._open_loop_detections_types: List[TrackedObjectType] = []
        self._minimum_path_length = minimum_path_length
        self._planned_trajectory_samples = planned_trajectory_samples
        self._planned_trajectory_sample_interval = planned_trajectory_sample_interval
        self._radius = radius

        # Prepare IDM agent manager
        self._idm_agent_manager: Optional[IDMAgentManager] = None
        self._initialize_open_loop_detection_types(open_loop_detections_types)

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration = 0
        self._idm_agent_manager = None

    def _initialize_open_loop_detection_types(self, open_loop_detections: List[str]) -> None:
        """
        Initializes open-loop detections with the enum types from TrackedObjectType
        :param open_loop_detections: A list of open-loop detections types as strings
        :return: A list of open-loop detections types as strings as the corresponding TrackedObjectType
        """
        for _type in open_loop_detections:
            try:
                self._open_loop_detections_types.append(TrackedObjectType[_type])
            except KeyError:
                raise ValueError(f"The given detection type {_type} does not exist or is not supported!")

    def _get_idm_agent_manager(self) -> IDMAgentManager:
        """
        Create idm agent manager in case it does not already exists
        :return: IDMAgentManager
        """
        if not self._idm_agent_manager:
            agents, agent_occupancy = build_idm_agents_on_map_rails(
                target_velocity=self._target_velocity,
                min_gap_to_lead_agent=self._min_gap_to_lead_agent,
                headway_time=self._headway_time,
                accel_max=self._accel_max,
                decel_max=self._decel_max,
                minimum_path_length=self._minimum_path_length,
                scenario=self._scenario,
                open_loop_detections_types=self._open_loop_detections_types,
            )
            self._idm_agent_manager = IDMAgentManager(agents, agent_occupancy, self._scenario.map_api)

        return self._idm_agent_manager

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""
        # _idm_agent_manager = self._get_idm_agent_manager()
        # detections = _idm_agent_manager.get_active_agents(
        #     iteration=self.current_iteration, num_samples=self._planned_trajectory_samples, sampling_time=self._planned_trajectory_sample_interval
        # )
        detections = self._get_idm_agent_manager().get_active_agents(
            iteration=self.current_iteration,
            num_samples=self._planned_trajectory_samples,
            sampling_time=self._planned_trajectory_sample_interval,
        )

        if self._open_loop_detections_types:
            open_loop_detections = self._get_open_loop_track_objects(self.current_iteration)
            detections.tracked_objects.tracked_objects.extend(open_loop_detections)  # List
        return detections

    def update_observation(self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer) -> None:
        """Inherited, see superclass.

        - **功能**: 更新观测数据，以适应下一次仿真迭代。
            - 更新当前迭代的索引。
            - 计算当前迭代与下一迭代之间的时间跨度。
            - 获取当前交通信号灯的状态，并将其与对应的车道连接器关联起来。
            - 获取自车状态，并通过 `IDMAgentManager` 更新其他智能体的状态。
        - **作用**: 确保观测器的数据与仿真的时间步同步，并且根据新计算的轨迹和状态更新智能体的位置和速度。
        """
        self.current_iteration = next_iteration.index
        tspan = next_iteration.time_s - iteration.time_s

        # traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(self.current_iteration)
        # Extract traffic light data into Dict[traffic_light_status, lane_connector_ids]
        # traffic_light_status: Dict[TrafficLightStatusType, List[str]] = defaultdict(list)

        ego_state, _ = history.current_state
        _idm_agent_manager = self._get_idm_agent_manager()
        _idm_agent_manager.propagate_agents(
            ego_state=ego_state,
            tspan=tspan,
            iteration=self.current_iteration,
            open_loop_detections=self._get_open_loop_track_objects(self.current_iteration),
            radius=self._radius,
        )

    def _get_open_loop_track_objects(self, iteration: int) -> List[TrackedObject]:
        """
        Get open-loop tracked objects from scenario.
        :param iteration: The simulation iteration.
        :return: A list of TrackedObjects.
        """
        detections = self._scenario.get_tracked_objects_at_iteration(iteration)
        return detections.tracked_objects.get_tracked_objects_of_types(self._open_loop_detections_types)  # type: ignore
