from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Dict

import numpy as np
import numpy.typing as npt

Transform = npt.NDArray[np.float32]  # 4x4 homogeneous transformation matrix


class MineSimMapLayer(Enum):
    """Enum of classification types for MineSimMapLayerType. 枚举成员定义
    member = value: int, fullname:str
    e.g. MineSimMapLayer.POLYGON.value # 0
    e.g. MineSimMapLayer.POLYGON.name # "POLYGON"
    e.g. MineSimMapLayer.POLYGON.fullname # "polygon"
    """

    POLYGON = 0, "polygon"
    NODE = 1, "node"
    NODE_BLOCK = 2, "node_block"
    ROAD_BLOCK = 3, "road_block"
    REFERENCE_PATH = 4, "reference_path"
    BORDERLINE = 5, "borderline"
    ROAD_SEGMENT = 6, "road"
    INTERSECTION = 7, "intersection"
    LOADING_AREA = 8, "loading_area"
    UNLOADING_AREA = 9, "unloading_area"
    DUBINS_POSE = 10, "dubins_pose"

    def __int__(self) -> int:
        """用于将枚举成员转换为整数 Convert an element to int
        :return: int
        """
        return self.value

    def __new__(cls, value: int, name: str) -> MineSimMapLayer:
        """自定义枚举成员的创建方式。 Create new element
        :param value: its value
        :param name: its name
        """
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __eq__(self, other: object) -> bool:
        """自定义等值判断方法。检查两个枚举实例是否具有相同的 `name` 和 `value`。 Equality checking"""
        try:
            return self.name == other.name and self.value == other.value
        except AttributeError:
            return NotImplemented

    def __hash__(self) -> int:
        """Hash;生成唯一的哈希值，以便将枚举对象用作字典的键或存储在集合中。"""
        return hash((self.name, self.value))


class GeometricLayer(IntEnum):
    """geometric layers"""

    POLYGON = 0  # polygon
    NODE = 1
    NODE_BLOCK = 2

    @classmethod
    def deserialize(cls, layer: str) -> GeometricLayer:
        """Deserialize the type when loading from a string.
        从字符串加载时反序列化类型。
        """
        return GeometricLayer.__members__[layer]


minesim_map_layer_names = {
    "polygon": MineSimMapLayer.POLYGON,
    "node": MineSimMapLayer.NODE,
    "node_block": MineSimMapLayer.NODE_BLOCK,
    "road_block": MineSimMapLayer.ROAD_BLOCK,
    "reference_path": MineSimMapLayer.REFERENCE_PATH,
    "borderline": MineSimMapLayer.BORDERLINE,
    "road": MineSimMapLayer.ROAD_SEGMENT,
    "intersection": MineSimMapLayer.INTERSECTION,
    "loading_area": MineSimMapLayer.LOADING_AREA,
    "unloading_area": MineSimMapLayer.UNLOADING_AREA,
    "dubins_pose": MineSimMapLayer.DUBINS_POSE,
}


class SemanticMapLayer(IntEnum):
    """
    Enum for SemanticMapLayers.
    """

    ROAD_SEGMENT = 0  # segment
    INTERSECTION = 1
    LOADING_AREA = 2
    UNLOADING_AREA = 3
    ROAD_BLOCK = 4
    REFERENCE_PATH = 5  # 矿区没有车道，使用PATH替代；
    BORDERLINE = 6
    DUBINS_POSE = 7
    # self.geometric_layers = ["polygon", "node", "node_block"]
    # self.other_layers = ["dubins_pose", "road_block"]
    # self.non_geometric_line_layers = ["reference_path", "borderline"]
    # self.non_geometric_polygon_layers = ["road", "intersection", "loading_area", "unloading_area"]
    # NOTE 原nuplan的语义图层 ==========
    # LANE = 0
    # INTERSECTION = 1
    # STOP_LINE = 2
    # TURN_STOP = 3
    # CROSSWALK = 4
    # DRIVABLE_AREA = 5
    # YIELD = 6
    # TRAFFIC_LIGHT = 7
    # STOP_SIGN = 8
    # EXTENDED_PUDO = 9
    # SPEED_BUMP = 10
    # LANE_CONNECTOR = 11
    # BASELINE_PATHS = 12
    # BOUNDARIES = 13
    # WALKWAYS = 14
    # CARPARK_AREA = 15
    # PUDO = 16
    # ROADBLOCK = 17
    # ROADBLOCK_CONNECTOR = 18
    # PRECEDENCE_AREA = 19

    @classmethod
    def deserialize(cls, layer: str) -> SemanticMapLayer:
        """Deserialize the type when loading from a string.
        从字符串加载时反序列化类型。
        """
        return SemanticMapLayer.__members__[layer]


class ReferencePathType(Enum):
    """Enum of classification types for ReferencePathType. 枚举成员定义
    member = value: int, name:str
    """

    BASE_PATH = 0, "base_path"
    CONNECTOR_PATH = 1, "connector_path"

    def __int__(self) -> int:
        """用于将枚举成员转换为整数 Convert an element to int
        :return: int
        """
        return self.value

    def __new__(cls, value: int, name: str) -> MineSimMapLayer:
        """自定义枚举成员的创建方式。 Create new element
        :param value: its value
        :param name: its name
        """
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __eq__(self, other: object) -> bool:
        """自定义等值判断方法。检查两个枚举实例是否具有相同的 `name` 和 `value`。 Equality checking"""
        try:
            return self.name == other.name and self.value == other.value
        except AttributeError:
            return NotImplemented

    def __hash__(self) -> int:
        """Hash;生成唯一的哈希值，以便将枚举对象用作字典的键或存储在集合中。"""
        return hash((self.name, self.value))


class BorderlineType(Enum):
    """Enum of classification types. 枚举成员定义

    member = value: int, name:str
    """

    EDGE = 0, "edge"
    INNER = 1, "inner"

    def __int__(self) -> int:
        """用于将枚举成员转换为整数 Convert an element to int"""
        return self.value

    def __new__(cls, value: int, name: str) -> MineSimMapLayer:
        """自定义枚举成员的创建方式。 Create new element
        :param value: its value
        :param name: its name
        """
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __eq__(self, other: object) -> bool:
        """自定义等值判断方法。检查两个枚举实例是否具有相同的 `name` 和 `value`。 Equality checking"""
        try:
            return self.name == other.name and self.value == other.value
        except AttributeError:
            return NotImplemented

    def __hash__(self) -> int:
        """Hash;生成唯一的哈希值，以便将枚举对象用作字典的键或存储在集合中。"""
        return hash((self.name, self.value))


class DubinsPoseType(Enum):
    """Enum of classification types  枚举成员定义

    member = value: int, name:str
    """

    MERGE = 0, "merge"
    SPLIT = 1, "split"
    NORMAL = 2, "normal"

    def __int__(self) -> int:
        """用于将枚举成员转换为整数 Convert an element to int"""
        return self.value

    def __new__(cls, value: int, name: str) -> MineSimMapLayer:
        """自定义枚举成员的创建方式。 Create new element
        :param value: its value
        :param name: its name
        """
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __eq__(self, other: object) -> bool:
        """自定义等值判断方法。检查两个枚举实例是否具有相同的 `name` 和 `value`。 Equality checking"""
        try:
            return self.name == other.name and self.value == other.value
        except AttributeError:
            return NotImplemented

    def __hash__(self) -> int:
        """Hash;生成唯一的哈希值，以便将枚举对象用作字典的键或存储在集合中。"""
        return hash((self.name, self.value))


@dataclass
class RasterLayer:
    """
    Wrapper dataclass of a layer of the rasterized map.
    对应矿区的 BitMap : MineSim-Dynamic-Dev/devkit/sim_engine/map_manager/minesim_map_data/minesim_bitmap_png_loader.py
    """

    data: npt.NDArray[np.uint8]  # raster image as numpy array
    precision: np.float64  # [m] precision of map 精度
    transform: Transform  # transform from physical to pixel coordinates


class StopLineType(IntEnum):
    """
    Enum for StopLineType.
    # TODO 设计矿区 虚拟停车线；
    """

    PED_CROSSING = 0
    STOP_SIGN = 1
    TRAFFIC_LIGHT = 2
    TURN_STOP = 3
    YIELD = 4
    UNKNOWN = 5


class TrafficLightStatusType(IntEnum):
    """
    Enum for TrafficLightStatusType.
    # TODO 有些矿区 存在信号灯 标识；后续需要再考虑扩展
    """

    GREEN = 0
    YELLOW = 1
    RED = 2
    UNKNOWN = 3

    def serialize(self) -> str:
        """Serialize the type when saving."""
        return self.name

    @classmethod
    def deserialize(cls, key: str) -> TrafficLightStatusType:
        """Deserialize the type when loading from a string."""
        return TrafficLightStatusType.__members__[key]


# @dataclass
# class VectorMap:
#     """将 SemanticMapLayers 映射到关联的 VectorLayer 的数据类。
#     Dataclass mapping SemanticMapLayers to associated VectorLayer.
#     """

#     layers: Dict[SemanticMapLayer, VectorLayer]


# @dataclass
# class RasterMap:
#     """
#     Dataclass mapping SemanticMapLayers to associated RasterLayer.
#     """

#     layers: Dict[SemanticMapLayer, RasterLayer]
