from typing import List

from devkit.sim_engine.callback.abstract_callback import AbstractCallback
from devkit.sim_engine.history.simulation_history import SimulationHistory
from devkit.sim_engine.history.simulation_history import SimulationHistorySample
from devkit.sim_engine.planning.planner.abstract_planner import AbstractPlanner
from devkit.sim_engine.environment_manager.simulation_setup import SimulationSetup
from devkit.common.trajectory.abstract_trajectory import AbstractTrajectory


class MultiCallback(AbstractCallback):
    """
    This class simply calls many callbacks for simplified code.
    """

    def __init__(self, callbacks: List[AbstractCallback]):
        """
        Initialize with multiple callbacks.
        :param callbacks: all callbacks that will be called sequentially.
        """
        self._callbacks = callbacks

    @property
    def callbacks(self) -> List[AbstractCallback]:
        """
        Property to access callbacks.
        :return: list of callbacks this MultiCallback runs.
        """
        return self._callbacks

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_initialization_start(setup, planner)

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_initialization_end(setup, planner)

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_step_start(setup, planner)

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_step_end(setup, planner, sample)

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_planner_start(setup, planner)

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_planner_end(setup, planner, trajectory)

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_simulation_start(setup)

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_simulation_end(setup, planner, history)
