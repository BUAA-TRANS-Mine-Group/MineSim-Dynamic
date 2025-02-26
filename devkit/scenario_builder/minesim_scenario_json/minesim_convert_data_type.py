from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Tuple
from devkit.common.actor_state.tracked_objects_types import TrackedObjectType
from devkit.common.actor_state.agent_state import AgentState
from devkit.common.actor_state.agent_temporal_state import AgentTemporalState
from devkit.common.actor_state.agent import Agent
from devkit.common.actor_state.scene_object import SceneObject
from devkit.common.actor_state.scene_object import SceneObjectMetadata
from devkit.common.actor_state.static_object import StaticObject
from devkit.common.actor_state.state_representation import TimePoint


@dataclass(frozen=True)
class SimTimePointOrigin:
    """MineSim scenario simulation time 0. NOT modify！"""

    sim_time_point_origin: TimePoint  # [micro seconds] time
    minesim_metadata_time_step: int  # 0 1 2 3 4
    dt: float  # e.g. 0.1 ,0.05


@unique
class MinesimScenarioType(Enum):
    INTERSECTION_MIXED = 1
    INTERSECTION_MIXED_ADD_STATIC_OBSTACLE = 2
    STATIC_OBSTACLE = 3  # static_obstacle
    STATIC_OBSTACLE_ADD_MIXED_VEHICLE = 4  # static_obstacle add mixed vehicle


class ScenarioFileBaseInfo:
    def __init__(
        self,
        log_file_load_path: str,
        log_file_name: str,
        scenario_name: str,
        location: str,
        scenario_type: MinesimScenarioType = MinesimScenarioType.INTERSECTION_MIXED,
        dt: float = None,
        data_root: str = None,
        map_root: str = None,
    ):
        self.log_file_load_path: str = log_file_load_path
        self.log_file_name: str = log_file_name
        self.scenario_name: str = scenario_name
        self.location: str = location  #! ["jiangxi_jiangtong","guangdong_dapai"]
        self.scenario_type = scenario_type
        self.dt: float = dt
        self.data_root: str = data_root
        self.map_root: str = map_root

    def __repr__(self) -> str:
        """
        自定义类的调试输出，主要展示 scenario_name 以及其他相关信息。
        """
        return f"<ScenarioFileBaseInfo='{self.scenario_name}'>"


class MineSimVehicleAgentState:
    def __init__(self, agent_state: AgentState, metadata_state: Dict):
        # 构建 MineSimVehicleAgentState 对象，封装 agent_state 和 metadata_state
        self.agent_state = agent_state
        # e.g. {'x': 1643.20972, 'y': 691.20917, 'yaw_rad': 4.45256, 'v_mps': 4.76679, 'yawrate_radps': -0.04748, 'acc_mpss': -0.39983}
        self.metadata_state = metadata_state


class MineSimScenarioVehicleTraj:
    def __init__(
        self,
        vehicle_id: int,
        vehicle_shape: Dict,
        vehicle_agent_states: List[MineSimVehicleAgentState],
        vehicle_time_s_list: List[float],
        vehicle_time_s_str_list: List[str],
        dt: float = 0.1,
    ):
        self.vehicle_id = vehicle_id
        self.vehicle_shape = vehicle_shape
        self.dt = dt
        self.vehicle_agent_states = vehicle_agent_states
        self.vehicle_time_s_list = vehicle_time_s_list
        self.vehicle_time_s_str_list = vehicle_time_s_str_list


class MineSimScenarioTrackedMetaData:
    def __init__(
        self,
        scenario_file_info: ScenarioFileBaseInfo,
        static_obstacles: Optional[List[StaticObject]] = None,
        dynamic_obstacle_vehicles: Optional[List[MineSimScenarioVehicleTraj]] = None,
    ):
        self.scenario_file_info = scenario_file_info
        self.scenario_type = self.scenario_file_info.scenario_type
        self.static_obstacles = static_obstacles
        self.dynamic_obstacle_vehicles = dynamic_obstacle_vehicles

        if static_obstacles is None and dynamic_obstacle_vehicles is None:
            raise ValueError("static_obstacles and dynamic_obstacle_vehicles cannot both be None.")

    def __repr__(self) -> str:
        """
        自定义类的调试输出，主要展示 scenario_name 以及其他相关信息。
        """
        return f"<MineSimScenarioTrackedMetaData='{self.scenario_file_info.scenario_name}'>"
