from __future__ import annotations

# Python library
import copy
import logging
from typing import Dict, List, Tuple, Optional, Type, Any

# Third-party library
from devkit.common.trajectory.abstract_trajectory import AbstractTrajectory
from devkit.sim_engine.callback.abstract_callback import AbstractCallback
from devkit.sim_engine.callback.multi_callback import MultiCallback
from devkit.sim_engine.observation_manager.observation_mine import MineObservation
from devkit.sim_engine.environment_manager.simulation_setup import SimulationSetup
from devkit.sim_engine.history.simulation_history import SimulationHistory
from devkit.sim_engine.history.simulation_history_buffer import SimulationHistoryBuffer
from devkit.sim_engine.history.simulation_history import SimulationHistorySample
from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.sim_engine.planning.planner.abstract_planner import PlannerInitialization
from devkit.sim_engine.planning.planner.abstract_planner import PlannerInput

logger = logging.getLogger(__name__)


class EnvironmentSimulation:
    """仿真环境读取及迭代过程,simulation."""

    def __init__(
        self,
        simulation_setup: SimulationSetup,
        callback: Optional[AbstractCallback] = None,
        simulation_history_buffer_duration: float = 0.1,
    ):
        """
        Create Simulation.
        :param simulation_setup: Configuration that describes the simulation.
        :param callback: A callback to be executed for this simulation setup
        :param simulation_history_buffer_duration: [s] Duration to pre-load scenario into the buffer.

        : param simulation_setup：描述模拟的配置。
        ：param callback：为模拟设置执行的回调
        ：param simulation_history_buffer_duration: [s] , 预加载场景到缓冲区的持续时间。
        """
        if simulation_history_buffer_duration < simulation_setup.scenario.database_interval:
            raise ValueError(
                f"simulation_history_buffer_duration {simulation_history_buffer_duration} has to be "
                f"larger than the scenario database_interval {simulation_setup.scenario.database_interval}"
            )
        self._setup = simulation_setup
        self._callback = MultiCallback([]) if callback is None else callback
        # Rolling window of past states
        # We add self._scenario_info.database_interval to the buffer duration here to ensure that the minimum
        # simulation_history_buffer_duration is satisfied
        # self._simulation_history_buffer_duration = simulation_history_buffer_duration + self._scenario.database_interval
        self._simulation_history_buffer_duration = simulation_history_buffer_duration + 0.1

        # Proxy
        self._time_controller = simulation_setup.time_controller
        self._ego_controller = simulation_setup.ego_controller
        self._observations = simulation_setup.observations
        self._scenario: AbstractScenario = simulation_setup.scenario

        # self._ego_vehicle = simulation_setup.ego_vehicle
        # self._ego_update_model = simulation_setup.ego_update_model

        # History where the steps of a simulation are stored
        self._history = SimulationHistory(planning_problem_goal_task=self._scenario.planning_problem_goal_task)

        # The + 1 here is to account for duration. For example, 20 steps at 0.1s starting at 0s will have a duration
        # of 1.9s. At 21 steps the duration will achieve the target 2s duration.
        self._history_buffer_size = int(self._simulation_history_buffer_duration / self._scenario.database_interval) + 1
        self._history_buffer: Optional[SimulationHistoryBuffer] = None

        # Flag that keeps track whether simulation is still running
        self._is_simulation_running = True

    def __reduce__(self) -> Tuple[Type[EnvironmentSimulation], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return self.__class__, (self._setup, self._callback, self._simulation_history_buffer_duration)

    def is_simulation_running(self) -> bool:
        """
        Check whether a simulation reached the end (goal box) or collision.
        :return True if simulation hasn't reached the end, otherwise false.
        """
        return not self._time_controller.reached_end() and self._is_simulation_running

    def reset(self) -> None:
        """Reset all internal states of simulation."""
        # Clear created log
        self._history.reset()

        # Reset all simulation internal members
        self._setup.reset()

        # Clear history buffer
        self._history_buffer = None

        # Restart simulation
        self._is_simulation_running = True

    def initialize(self) -> PlannerInitialization:
        """
        Initialize the simulation
         - Initialize Planner with goals and maps
        :return data needed for planner initialization.
        """
        self.reset()

        # Initialize history from scenario
        self._history_buffer = SimulationHistoryBuffer.initialize_from_scenario(
            buffer_size=self._history_buffer_size, scenario=self._scenario, observation_type=self._observations.observation_type()
        )

        # Initialize observations
        self._observations.initialize()

        # Add the current state into the history buffer
        # NOTE : 0.0  初始时候前2.0秒的历史是虚拟的:均使用0.0的值复制得到
        self._history_buffer.append(ego_state=self._ego_controller.get_state(), observation=self._observations.get_observation())

        # Return the planner initialization structure for this simulation
        return PlannerInitialization(
            initial_ego_state=self._scenario.initial_ego_state,
            planning_problem_goal_task=self._scenario.planning_problem_goal_task,
            map_api=self._scenario.map_api,
            scenario_name=self.scenario.scenario_name,
        )

    def get_planner_input(self) -> PlannerInput:
        """
        Construct inputs to the planner for the current iteration step
        :return Inputs to the planner.
        """
        if self._history_buffer is None:
            raise RuntimeError("Simulation was not initialized!")

        if not self.is_simulation_running():
            raise RuntimeError("Simulation is not running, stepping can not be performed!")

        # Extract current state
        iteration = self._time_controller.get_iteration()

        logger.debug(f"#log# Executing {iteration.index}!")
        return PlannerInput(iteration=iteration, history=self._history_buffer)

    def propagate(self, trajectory: AbstractTrajectory) -> None:
        """
        Propagate the simulation based on planner's trajectory and the inputs to the planner
        This function also decides whether simulation should still continue. This flag can be queried through
        reached_end() function
        :param trajectory: computed trajectory from planner.
        """
        if self._history_buffer is None:
            raise RuntimeError("Simulation was not initialized!")

        if not self.is_simulation_running():
            raise RuntimeError("Simulation is not running, simulation can not be propagated!")

        # Measurements
        iteration = self._time_controller.get_iteration()
        ego_state, observation = self._history_buffer.current_state

        # Add new sample to history
        logger.debug(f"#log# Adding to history: {iteration.index}")

        # ! 存储simulation 历史的信息 ； 记录仿真的每一步,不包括仿真初始的信息
        self._history.add_sample(SimulationHistorySample(iteration=iteration, ego_state=ego_state, trajectory=trajectory, observation=observation))

        # Propagate state to next iteration
        # todo 无回放 observations;
        next_iteration = self._time_controller.next_iteration()

        # Propagate state
        # !核心仿真部分
        if next_iteration:
            self._ego_controller.update_state(current_iteration=iteration, next_iteration=next_iteration, ego_state=ego_state, trajectory=trajectory)
            self._observations.update_observation(iteration=iteration, next_iteration=next_iteration, history=self._history_buffer)
        else:
            self._is_simulation_running = False

        # Append new state into history buffer
        # !存储场景中历史的信息 ； _history_buffer.durantion = 2.2 ， _history_buffer.size =22;可以配置历史缓存队列；
        self._history_buffer.append(ego_state=self._ego_controller.get_state(), observation=self._observations.get_observation())
        a = 1

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: used scenario in this simulation.
        """
        return self._scenario

    @property
    def setup(self) -> SimulationSetup:
        """
        :return: Setup for this simulation.
        """
        return self._setup

    @property
    def callback(self) -> AbstractCallback:
        """
        ! 执行该回调后,self._callback 回调的多个callback 都会被执行;
        e.g. devkit.sim_engine.callback.simulation_log_callback.SimulationLogCallback
        :return: Callback for this simulation.
        """
        return self._callback

    @property
    def history(self) -> SimulationHistory:
        """
        :return History from the simulation.
        """
        return self._history

    @property
    def history_buffer(self) -> SimulationHistoryBuffer:
        """
        :return SimulationHistoryBuffer from the simulation.
        """
        if self._history_buffer is None:
            raise RuntimeError("_history_buffer is None. Please initialize the buffer by calling Simulation.initialize()")
        return self._history_buffer



if __name__ == "__main__":
    import time
