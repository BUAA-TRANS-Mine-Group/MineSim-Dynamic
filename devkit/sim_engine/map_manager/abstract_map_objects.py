from __future__ import annotations

from functools import lru_cache
from functools import cached_property
import abc
import logging
from typing import Dict, List, Optional, Tuple, Union

from shapely.geometry import LineString, Point, Polygon

from devkit.common.actor_state.state_representation import Point2D
from devkit.common.actor_state.state_representation import StateSE2
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import DubinsPoseType
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import MineSimMapLayer
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import ReferencePathType
from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapJsonLoader

logger = logging.getLogger(__name__)


class AbstractMapObject(abc.ABC):
    """
    Base interface representation of all map objects.
    """

    def __init__(self, object_token: str):
        """
        Constructor of the base lane type.
        :param object_token: unique identifier of the map object.
        """
        self.token = object_token


class PolylineMapObject(AbstractMapObject):
    """
    A class to represent any map object that can be represented as a polyline.
    """

    def __init__(self, polyline_token: str):
        super().__init__(object_token=polyline_token)

    @property
    @abc.abstractmethod
    def linestring(self) -> LineString:
        """
        Returns the polyline as a Linestring.
        :return: The polyline as a Linestring.
        """
        pass

    @property
    @abc.abstractmethod
    def length(self) -> float:
        """
        Returns the length of the polyline [m].
        :return: the length of the polyline.
        """
        pass

    @property
    @abc.abstractmethod
    def discrete_path(self) -> List[StateSE2]:
        """
        Gets a discretized representation of the polyline.
        :return: a list of StateSE2.
        """
        pass

    @abc.abstractmethod
    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        """
        Returns the arc length along the polyline where the given point is the closest.
        :param point: [m] x, y coordinates in global frame.
        :return: [m] arc length along the polyline.
        """
        pass

    @abc.abstractmethod
    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        """
        Returns the pose along the polyline where the given point is the closest.
        :param point: [m] x, y coordinates in global frame.
        :return: nearest pose along the polyline as StateSE2.
        """
        pass

    @abc.abstractmethod
    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        """
        Return curvature at an arc length along the polyline.
        :param arc_length: [m] arc length along the polyline. It has to be 0<= arc_length <=length.
        :return: [1/m] curvature along a polyline.
        """
        pass

    def get_nearest_curvature_from_position(self, point: Point2D) -> float:
        """
        Returns the curvature along the polyline where the given point is the closest.
        :param point: [m] x, y coordinates in global frame.
        :return: [1/m] curvature along a polyline.
        """
        return self.get_curvature_at_arc_length(self.get_nearest_arc_length_from_position(point))


class PolygonMapObject(AbstractMapObject):
    """一个类，用于表示可表示为多边形的任何 Map 对象。
    A class to represent any map object that can be represented as a polygon.
    """

    @property
    @abc.abstractmethod
    def polygon(self) -> Polygon:
        """
        Returns the surface of the map object as a Polygon.
        :return: The map object as a Polygon.
        """
        pass

    @property
    @abc.abstractmethod
    def type(self) -> MineSimMapLayer:
        pass

    @property
    @abc.abstractmethod
    def link_dubinspose_tokens(self) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def link_referencepath_tokens(self) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def link_borderline_tokens(self) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def link_node_tokens(self) -> List[str]:
        pass

    def contains_point(self, point: Point2D) -> bool:
        """检查指定的点是否是地图对象多边形的一部分。
        Checks if the specified point is part of the map object polygon.
        :return: True if the point is within the polygon.
        """
        return bool(self.polygon.contains(Point(point.x, point.y)))


class DubinsPoseMapObject(AbstractMapObject):
    """Dubins Pose Map Object"""

    def __init__(self, dubinspose_token: str, semantic_map: MineSimSemanticMapJsonLoader):
        super().__init__(object_token=dubinspose_token)
        self.pose_id = semantic_map.getind(layer_name=MineSimMapLayer.DUBINS_POSE.fullname, token=dubinspose_token)
        self.dubins_pose_meta_data: Dict = semantic_map.dubins_pose[self.pose_id]

    @property
    def Pose_StateSE2(self) -> StateSE2:
        """Returns Pose_StateSE2"""
        return StateSE2(x=self.dubins_pose_meta_data["x"], y=self.dubins_pose_meta_data["y"], heading=self.dubins_pose_meta_data["yaw"])

    @cached_property
    def type(self) -> DubinsPoseType:
        if self.polygon_meta_data.type == "merge":
            element_type = DubinsPoseType.MERGE
        elif self.polygon_meta_data.type == "split":
            element_type = DubinsPoseType.SPLIT
        elif self.polygon_meta_data.type == "normal":
            element_type = DubinsPoseType.NORMAL
        else:
            logger.error(f"#log# dubins_pose token={self.dubins_pose_meta_data.token},type={self.dubins_pose_meta_data.type} is error!!!")
        return element_type

    @property
    def link_polygon_token(self) -> str:
        return self.dubins_pose_meta_data["link_polygon_token"]
