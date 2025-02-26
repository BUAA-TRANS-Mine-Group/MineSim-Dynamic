import numpy as np

from devkit.common.actor_state.dynamic_car_state import DynamicCarState
from devkit.common.actor_state.ego_state import EgoState
from devkit.common.actor_state.state_representation import StateVector2D, TimePoint
from devkit.common.actor_state.vehicle_parameters import VehicleParameters
from devkit.sim_engine.ego_simulation.ego_update_model.kinematic_bicycle_model import KinematicBicycleModel


class KinematicBicycleModelRespLagRoadSlope(KinematicBicycleModel):
    """
    Bicycle Model with Response Lag and Road Slope (KBM-wRLwRS)

    "models": [
            "KBM",  # Kinematic Bicycle Model (KBM)
            "KBM-wRL",  # Bicycle Model with Response Lag (KBM-wRL)
            "KBM-wRLwRS",  # Bicycle Model with Response Lag and Road Slope (KBM-wRLwRS)
        ],
    """

    pass
    # TODO
