from typing import Type

from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.sim_engine.history.simulation_history_buffer import SimulationHistoryBuffer
from devkit.sim_engine.observation_manager.abstract_observation import AbstractObservation
from devkit.sim_engine.observation_manager.observation_type import DetectionsTracks, Observation
from devkit.sim_engine.simulation_time_controller.simulation_iteration import SimulationIteration


class BoxTracksObservation(AbstractObservation):
    """
    Replay detections from samples.
    """

    def __init__(self, scenario: AbstractScenario):
        """
        :param scenario: The scenario
        """
        self.scenario = scenario
        self.current_iteration = 0

    def reset(self) -> None: 
        """Inherited, see superclass."""
        self.current_iteration = 0

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""
        return self.scenario.get_tracked_objects_at_iteration(self.current_iteration)

    def update_observation(self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer) -> None:
        """Inherited, see superclass."""
        self.current_iteration = next_iteration.index
