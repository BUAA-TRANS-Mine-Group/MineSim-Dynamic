import abc

from devkit.common.actor_state.dynamic_car_state import DynamicCarState
from devkit.common.actor_state.ego_state import EgoState
from devkit.sim_engine.simulation_time_controller.simulation_iteration import SimulationIteration
from devkit.common.trajectory.abstract_trajectory import AbstractTrajectory


class AbstractTracker(abc.ABC):
    """
    Interface for a generic tracker.
    """

    @abc.abstractmethod
    def track_trajectory(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        initial_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> DynamicCarState:
        """
        Return an ego state with updated dynamics according to the controller commands.
        :param current_iteration: The current simulation iteration.
        :param next_iteration: The desired next simulation iteration.
        :param initial_state: The current simulation iteration.
        :param trajectory: The reference trajectory to track.
        :return: The ego state to be propagated
        """
        pass

    @abc.abstractmethod
    def get_ego_vehicle_parameters(self, mine_name: str):
        pass
