from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from devkit.common.actor_state.oriented_box import OrientedBox
from devkit.common.actor_state.state_representation import StateSE2
from devkit.common.actor_state.tracked_objects_types import TrackedObjectType


@dataclass(frozen=True)
class SceneObjectMetadata:
    """
    Metadata for every object; add Minesim data.

    devkit e.g.
    SceneObjectMetadata(timestamp_us=1626469642601231, token='9307f96fdabf578c', track_id=5, track_token='69053340173457e4', category_name='vehicle')

    - Note 1:
        timestamp_us: int # Timestamp of this object in micro seconds
        track_id_minesim: int #! add object type in MineSim , e.g. "1" "5" "parkedVehicle-101"
        token: Optional[str] = None # Unique token in a whole dataset; #! set None in MineSim
        track_id: Optional[int] = None  # Human understandable id of the object  #! set None in MineSim
        track_token: Optional[str] = None # Token of the object which is temporally consistent #! set “object-id” in MineSim
        category_name: Optional[str] = None  # Human readable category name #! add object type in MineSim
        - e.g.PickupSuv_BX5;
        - vehicle_shape={'vehicle_type': 'PickupSuv_BX5', 'length': 4.49, 'width': 1.877, 'height': 1.675, 'locationPoint2Head': 2.245, 'locationPoint2Rear': 2.245}

    - Note 2:
        `@dataclass(frozen=True)`: 它是不可变的（immutable)
        `__post_init__()` 是 `dataclass` 提供的特殊方法，在对象实例化后会自动执行。
        `object.__setattr__(self, 'track_token', value)` 允许绕过 `frozen=True` 限制，在 `__post_init__` 方法内安全地设置字段值。

    """

    timestamp_us: int
    track_id_minesim: int
    token: Optional[str] = None
    track_id: Optional[int] = None
    track_token: Optional[str] = None
    category_name: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "track_token", f"object-{self.track_id_minesim}")

    @property
    def timestamp_s(self) -> float:
        """
        :return: timestamp in seconds
        """
        return self.timestamp_us * 1e-6


class SceneObject:
    """Class describing SceneObjects, i.e. objects present in a planning scene"""

    def __init__(self, tracked_object_type: TrackedObjectType, oriented_box: OrientedBox, metadata: SceneObjectMetadata):
        """
        Representation of an Agent in the scene.
        :param tracked_object_type: Type of the current static object
        :param oriented_box: Geometrical representation of the static object
        :param metadata: High-level information about the object
        """
        self._metadata = metadata
        self.instance_token = None
        self._tracked_object_type = tracked_object_type

        self._box: OrientedBox = oriented_box

    @property
    def metadata(self) -> SceneObjectMetadata:
        """
        Getter for object metadata
        :return: Object's metadata
        """
        return self._metadata

    @property
    def token(self) -> str:
        """
        Getter for object unique token, different for same object in different samples
        :return: The unique token
        """
        return self._metadata.token

    @property
    def track_token(self) -> Optional[str]:
        """
        Getter for object unique token tracked across samples, same for same objects in different samples
        :return: The unique track token
        """
        return self._metadata.track_token

    @property
    def tracked_object_type(self) -> TrackedObjectType:
        """
        Getter for object classification type
        :return: The object classification type
        """
        return self._tracked_object_type

    @property
    def box(self) -> OrientedBox:
        """
        Getter for object OrientedBox
        :return: The object oriented box
        """
        return self._box

    @property
    def center(self) -> StateSE2:
        """
        Getter for object center pose
        :return: The center pose
        """
        return self.box.center

    @classmethod
    def make_random(cls, token: str, object_type: TrackedObjectType) -> SceneObject:
        """
        Instantiates a random SceneObject.
        :param token: Unique token
        :param object_type: Classification type
        :return: SceneObject instance.
        """
        center = random.sample(range(50), 2)
        heading = np.random.uniform(-np.pi, np.pi)
        size = random.sample(range(1, 50), 3)
        track_id = random.sample(range(1, 10), 1)[0]
        timestamp_us = random.sample(range(1, 10), 1)[0]

        return SceneObject(
            metadata=SceneObjectMetadata(token=token, track_id=track_id, track_token=token, timestamp_us=timestamp_us),
            tracked_object_type=object_type,
            oriented_box=OrientedBox(StateSE2(*center, heading), size[0], size[1], size[2]),
        )

    @classmethod
    def from_raw_params(
        cls,
        token: str,
        track_token: str,
        timestamp_us: int,
        track_id: int,
        center: StateSE2,
        size: Tuple[float, float, float],
    ) -> SceneObject:
        """
        Instantiates a generic SceneObject.
        :param token: The token of the object.
        :param track_token: The track token of the object.
        :param timestamp_us: [us] timestamp for the object.
        :param track_id: Human readable track id.
        :param center: Center pose.
        :param size: Size of the geometrical box (width, length, height).
        :return: SceneObject instance.
        """
        box = OrientedBox(center, width=size[0], length=size[1], height=size[2])
        return SceneObject(
            metadata=SceneObjectMetadata(token=token, track_token=track_token, timestamp_us=timestamp_us, track_id=track_id),
            tracked_object_type=TrackedObjectType.GENERIC_OBJECT,
            oriented_box=box,
        )
