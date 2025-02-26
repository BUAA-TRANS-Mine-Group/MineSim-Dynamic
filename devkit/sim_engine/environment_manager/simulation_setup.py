from dataclasses import dataclass
from omegaconf import DictConfig
from hydra.utils import instantiate

from devkit.script.builders.utils.utils_type import is_target_type
from devkit.common.actor_state.vehicle_parameters import VehicleParameters
from devkit.common.actor_state.vehicle_parameters import get_mine_truck_parameters
from devkit.sim_engine.ego_simulation.ego_update_model.abstract_ego_state_update_model import AbstractEgoStateUpdateModel
from devkit.sim_engine.ego_simulation.abstract_controller import AbstractEgoController
from devkit.sim_engine.observation_manager.abstract_observation import AbstractObservation
from devkit.script.builders.observation_builder import build_observations
from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.sim_engine.simulation_time_controller.abstract_simulation_time_controller import AbstractSimulationTimeController
from devkit.sim_engine.ego_simulation.two_stage_controller import TwoStageController


@dataclass
class SimulationSetup:
    """Setup class for contructing a Simulation.
    用于存储和管理模拟器的初始化设置，如时间控制器、观察器、控制器等。

    #### 主要成员：
    - `time_controller`：这是一个 `AbstractSimulationTimeController` 类型的实例，用于管理模拟中的时间进度。
    - `observations`：这是一个 `AbstractObservation` 类型的实例，表示用于自动驾驶系统观察周围环境的数据，如感知结果。
    - `ego_controller`：这是一个 `AbstractEgoController` 类型的实例，负责控制自主车辆（Ego vehicle）。
    - `scenario`：一个 `ScenarioInfo` 类型的实例，描述了模拟的场景，包括地图、交通参与者等。

    #### all simulation component :
    - `scenario_info` : scenario manager , manage `scenario info`,  `planning problem info` and `HD map`.
    - `time_controller`：time manager, manage the time schedule in the simulation.
    - `ego_controller`： ego vehicle motion controller (manager)， control ego's motion track a trajectory.
    - `ego_update_model `: ego vehicle state update model (manager), simulate ego's state update in simulation.
    - `observations `: Perception Observation info.表示用于自动驾驶系统观察周围环境的数据，如感知结果
    - `ego_vehicle `: ego Vehicle Parameters.
    """

    def __init__(self, scenario: AbstractScenario = None, sim_config=None, cfg: DictConfig = None) -> None:
        """init all simulation component"""
        self.time_controller: AbstractSimulationTimeController = None  # [scenario manager]
        self.observations: AbstractObservation = None  # Perception Observation info
        self.ego_controller: AbstractEgoController = None  # Ego Controller
        self.scenario: AbstractScenario = scenario  # [scenario manager] scenario, simulation info and map

        self.ego_update_model: AbstractEgoStateUpdateModel = None  # Ego Vehicle Model (sub-module in Ego Controller)
        self.ego_vehicle: VehicleParameters = None

        # two configure mode:.PY or .YMAL
        if False:
            self._sim_config = sim_config
            self._setup_manager_by_sim_engine_conf()
        else:
            self.cfg = cfg
            self._setup_manager_by_nuplan_cfg()

    def __post_init__(self) -> None:
        """Post-initialization sanity checks.
        该方法在 dataclass 自动生成的 __init__ 方法之后调用，用于执行额外的初始化检查，
        确保传入的 time_controller、observations 和 ego_controller 均为正确的抽象类子类。
        """
        # Other checks
        assert isinstance(
            self.time_controller, AbstractSimulationTimeController
        ), "Error: simulation_time_controller must inherit from AbstractSimulationTimeController!"

    def _setup_manager_by_nuplan_cfg(self):
        """使用 MineSim/devkit/script/config ... YAML 配置参数.
        实例化仿真时所需的核心组件  simulation componet : (时间控制、观察器、自车控制器、场景等)
        """
        # Ego Controller # 实例化自车控制器
        self.ego_controller: AbstractEgoController = instantiate(self.cfg.ego_simulation, scenario=self.scenario)
        # !必须配置 ego vehicle
        self._setup_ego_vehicle_parameters_for_ego_controller()

        # Simulation Manager # 实例化仿真时间控制器  # note: simulation_time_controller named by nuplan
        self.time_controller: AbstractSimulationTimeController = instantiate(self.cfg.simulation_time_controller, scenario=self.scenario)
        # Perception # 构建观察器(感知/数据获取模块)
        self.observations: AbstractObservation = build_observations(self.cfg.observation_agent_update_policy, scenario=self.scenario)

    def _setup_ego_vehicle_parameters_for_ego_controller(self):
        if self.scenario.ego_vehicle_parameters is None:
            self.ego_vehicle = get_mine_truck_parameters(mine_name=self.scenario.scenario_file.location)
        if is_target_type(cfg=self.cfg.ego_simulation, target_type=TwoStageController):
            self.ego_controller._tracker.get_ego_vehicle_parameters(mine_name=self.scenario.scenario_file.location)

    def reset(self) -> None:
        """
        Reset all simulation controllers
        """
        self.observations.reset()
        self.ego_controller.reset()
        self.time_controller.reset()


# todo
# def validate_planner_setup(setup: SimulationSetup, planner: AbstractPlanner) -> None:
#     """
#     Validate planner and simulation setup
#     :param setup: Simulation setup
#     :param planner: Planner to be used
#     @raise ValueError in case simulation setup and planner are not a valid combination
#     """
#     # Validate the setup
#     type_observation_planner = planner.observation_type()
#     type_observation = setup.observations.observation_type()

#     if type_observation_planner != type_observation:
#         raise ValueError(
#             "Error: The planner did not receive the right observations:"
#             f"{type_observation} != {type_observation_planner} planner."
#             f"Planner {type(planner)}, MineObservation:{type(setup.observations)}"
#         )
