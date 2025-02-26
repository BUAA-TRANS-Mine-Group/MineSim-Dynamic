from typing import Optional

from devkit.common.trajectory.abstract_trajectory import AbstractTrajectory
from devkit.common.actor_state.ego_state import EgoState
from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.sim_engine.ego_simulation.ego_update_model.abstract_ego_state_update_model import AbstractEgoStateUpdateModel
from devkit.sim_engine.ego_simulation.ego_motion_controller.abstract_tracker import AbstractTracker
from devkit.sim_engine.ego_simulation.ego_motion_controller.abstract_tracker import AbstractTracker
from devkit.sim_engine.ego_simulation.abstract_controller import AbstractEgoController
from devkit.sim_engine.simulation_time_controller.simulation_iteration import SimulationIteration
from devkit.sim_engine.ego_simulation.ego_update_model.kinematic_bicycle_model import KinematicBicycleModel
from devkit.sim_engine.ego_simulation.ego_update_model.kinematic_bicycle_model_response_lag import KinematicBicycleModelResponseLag

# MineSim-Dynamic-Dev/devkit/sim_engine/ego_motion_controller


class TwoStageController(AbstractEgoController):
    """
    Implements a two stage tracking controller. The two stages comprises of:
        1. an AbstractTracker - This is to simulate a low level controller layer that is present in real AVs.
        2. an AbstractMotionModel - Describes how the AV evolves according to a physical model.
    实现一个两级跟踪控制器。这两个阶段包括：
        1. AbstractTracker——这是为了模拟真实自动驾驶汽车中的低级控制器层。
        2. 一个AbstractMotionModel——描述AV如何根据物理模型 进行状态更新。
    """

    def __init__(self, scenario: AbstractScenario, ego_motion_controller: AbstractTracker, ego_update_model: AbstractEgoStateUpdateModel):
        """
        Constructor for TwoStageController
        :param scenario: Scenario
        :param tracker(ego_motion_controller): The tracker used to compute control actions
        :param ego_update_model: The motion model to propagate the control actions
        """
        self._scenario = scenario
        self._tracker = ego_motion_controller
        self._ego_motion_model = ego_update_model

        # set to None to allow lazily loading of initial ego state
        self._current_state: Optional[EgoState] = None
        if type(ego_update_model) == KinematicBicycleModel:
            self._ego_motion_model._vehicle = self._scenario.ego_vehicle_parameters
            self._ego_motion_model._max_steering_angle = self._scenario._ego_vehicle_parameters.constraints["max_steering_angle"]
        elif type(ego_update_model) == KinematicBicycleModelResponseLag:
            self._ego_motion_model._vehicle = self._scenario.ego_vehicle_parameters
            self._ego_motion_model._max_steering_angle = self._scenario._ego_vehicle_parameters.constraints["max_steering_angle"]
            # self._ego_motion_model._accel_time_constant = None
            # self._ego_motion_model._steering_angle_time_constant = None
        elif type(ego_update_model) == KinematicBicycleModelResponseLag:
            ValueError("todo")
        else:
            ValueError("#log# ego_update_model is not supported.")

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._current_state = None

    def get_state(self) -> EgoState:
        """Inherited, see superclass."""
        if self._current_state is None:
            # todo bug
            self._current_state = self._scenario.initial_ego_state

        return self._current_state

    def update_state(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        ego_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> None:
        """Inherited, see superclass."""
        sampling_time = next_iteration.time_point - current_iteration.time_point

        # Compute the dynamic state to propagate the model
        dynamic_state = self._tracker.track_trajectory(current_iteration, next_iteration, ego_state, trajectory)

        # Propagate ego state using the motion model
        # 比如 KinematicBicycleModel
        self._current_state = self._ego_motion_model.propagate_state(state=ego_state, ideal_dynamic_state=dynamic_state, sampling_time=sampling_time)
