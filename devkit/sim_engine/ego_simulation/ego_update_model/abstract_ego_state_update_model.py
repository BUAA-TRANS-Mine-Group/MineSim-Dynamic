import abc

from devkit.common.actor_state.dynamic_car_state import DynamicCarState
from devkit.common.actor_state.ego_state import EgoState, EgoStateDot
from devkit.common.actor_state.state_representation import TimePoint


class AbstractEgoStateUpdateModel(abc.ABC):
    """
    Interface for generic ego state updater.
    """

    @abc.abstractmethod
    def get_state_dot(self, state: EgoState) -> EgoStateDot:
        """
        Compute x_dot = f(x) for the motion model.

        :param state: The state for which to compute motion model.
        :return: The state derivative as an EgoState.
        """
        pass

    @abc.abstractmethod
    def propagate_state(self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint) -> EgoState:
        """
        Propagate the state according to the motion model(ego update model).

        :param state: The initial state to propagate.
        :param sampling_time: The time duration to propagate for.
        :param ideal_dynamic_state: The desired dynamic state for propagation.
        :return: The propagated state.
        """
        pass
