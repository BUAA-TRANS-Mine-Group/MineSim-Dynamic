from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

from devkit.common.actor_state.tracked_objects import TrackedObjects


@dataclass
class Observation(ABC):
    """
    Abstract observation container.
    """

    iteration_step: int  # add in MineSim:0,1,2

    @classmethod
    def detection_type(cls) -> str:
        """
        Returns detection type of the observation.
        """
        return cls.__name__

    @property
    def timestep_s_str(self) -> str:
        """add in MineSim: e.g. "0.0","0.1" """
        return f"{self.iteration_step/10:.1f}"


@dataclass
class DetectionsTracks(Observation):
    """
    Output of the perception system, i.e. tracks.
    """

    tracked_objects: TrackedObjects
    is_real_data: bool = True  # False,虚拟仿真的，例如使用 IDM agent； True， 实际运行测量得到的数据
