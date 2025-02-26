import abc

from devkit.common.actor_state.ego_state import EgoState
from devkit.common.trajectory.abstract_trajectory import AbstractTrajectory
from devkit.sim_engine.simulation_time_controller.simulation_iteration import SimulationIteration


class AbstractEgoController(abc.ABC):
    """
    Interface for generic ego controllers.
    """

    @abc.abstractmethod
    def get_state(self) -> EgoState:
        """
        Returns the current ego state.
        :return: The current ego state.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset the observation (all internal states should be reseted, if any).
        """
        pass

    @abc.abstractmethod
    def update_state(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        ego_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> None:
        """
        Update ego's state from current iteration to next iteration.

        :param current_iteration: The current simulation iteration.
        :param next_iteration: The desired next simulation iteration.
        :param ego_state: The current ego state.
        :param trajectory: The output trajectory of a planner.
        """
        pass
