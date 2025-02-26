from devkit.sim_engine.callback.abstract_callback import AbstractCallback
from devkit.sim_engine.history.simulation_history import SimulationHistory
from devkit.sim_engine.history.simulation_history import SimulationHistorySample
from devkit.sim_engine.planning.planner.abstract_planner import AbstractPlanner
from devkit.sim_engine.environment_manager.simulation_setup import SimulationSetup
from devkit.common.trajectory.abstract_trajectory import AbstractTrajectory
from devkit.sim_engine.planning.simulation.visualization.abstract_visualization import AbstractVisualization


class VisualizationCallback(AbstractCallback):
    """Callback to render simulation data as the simulation runs."""

    def __init__(self, renderer: AbstractVisualization):
        """
        Constructor for VisualizationCallback.
        :param renderer: handler to create visualization.
        """
        self._visualization = renderer

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        In initialization start just render scenario
        """
        self._visualization.render_scenario(setup.scenario, True)

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """
        Render sample after a step
        """
        self._visualization.render_ego_state(sample.ego_state)
        self._visualization.render_observations(sample.observation)
        self._visualization.render_trajectory(sample.trajectory.get_sampled_trajectory())
        self._visualization.render(sample.iteration)

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """
        On reached_end just call step_end
        """
        self.on_step_end(setup, planner, history.data[-1])
