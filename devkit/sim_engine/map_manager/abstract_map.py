from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from devkit.common.actor_state.state_representation import Point2D

from devkit.sim_engine.map_manager.abstract_map_objects import AbstractMapObject
from devkit.sim_engine.map_manager.abstract_map_objects import PolygonMapObject
from devkit.sim_engine.map_manager.abstract_map_objects import DubinsPoseMapObject
from devkit.sim_engine.map_manager.abstract_map_objects import PolylineMapObject
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import GeometricLayer
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import SemanticMapLayer
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import ReferencePathType
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import BorderlineType
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import DubinsPoseType
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import RasterLayer

from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphEdgeRefPathMapObject
from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphTopologyNodeMapObject
from devkit.sim_engine.map_manager.minesim_map.reference_path_base import ReferencePathBase
from devkit.sim_engine.map_manager.minesim_map.reference_path_connector import ReferencePathConnector

from devkit.sim_engine.map_manager.minesim_map_data.minesim_bitmap_png_loader import MineSimBitMapPngLoader
from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapJsonLoader
from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapExplorer

# 图搜索对象元素
# GraphSearchEdgeObject = Union[GraphEdgeRefPathMapObject, ReferencePathBase, ReferencePathConnector]
# GraphSearchNodeObject = Union[GraphTopologyNodeMapObject]
# GraphSearchObject = Union[GraphEdgeRefPathMapObject, ReferencePathBase, ReferencePathConnector, GraphTopologyNodeMapObject]
# TODO 完善地图对象元素
MapObject = Union[PolygonMapObject, DubinsPoseMapObject, GraphTopologyNodeMapObject, PolylineMapObject, GraphEdgeRefPathMapObject]


class AbstractMap(abc.ABC):
    """
    Interface for generic scenarios Map API.
    """

    @abc.abstractmethod
    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        """返回可用的地图对象类型。
        Returns the available map objects types.
        :return: A list of SemanticMapLayers.
        """
        pass

    @abc.abstractmethod
    def get_raster_map(self) -> MineSimBitMapPngLoader:
        """
        Gets raster maps specified.
        """
        pass

    @property
    @abc.abstractmethod
    def map_name(self) -> str:
        """
        :return: name of the location where the map is.
        """
        pass

    @abc.abstractmethod
    def initialize_all_layers(self) -> None:
        """
        Load all layers to vector map
        """
        pass

    # @abc.abstractmethod
    # def get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[MapObject]:
    #     """ 返回包含给定点 x、y 的语义层上的所有映射对象。

    #     Returns all map objects on a semantic layer that contains the given point x, y.
    #     :param point: [m] x, y coordinates in global frame.
    #     :param layer: A semantic layer to query.
    #     :return: list of map objects.
    #     """
    #     pass

    # @abc.abstractmethod
    # def get_one_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Optional[MapObject]:
    #     """返回包含给定点 x、y 的语义层上的一个 map 对象。
    #     Returns one map objects on a semantic layer that contains the given point x, y.
    #     :param point: [m] x, y coordinates in global frame.
    #     :param layer: A semantic layer to query.
    #     :return: one map object if there is one map object else None if no map objects.
    #     @raise AssertionError if more than one object is found
    #     """
    #     pass

    # @abc.abstractmethod
    # def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
    #     """检查给定的点 x、y 是否位于语义层内。
    #     Checks if the given point x, y lies within a semantic layer.
    #     :param point: [m] x, y coordinates in global frame.
    #     :param layer: A semantic layer to query.
    #     :return: True if [x, y] is in a layer, False if it is not.
    #     @raise ValueError if layer does not exist
    #     """
    #     pass

    # @abc.abstractmethod
    # def get_proximal_map_objects(self, point: Point2D, radius: float, layers: List[SemanticMapLayer]) -> Dict[SemanticMapLayer, List[MapObject]]:
    #     """提取围绕点 x、y 的给定半径内的贴图对象。
    #     Extract map objects within the given radius around the point x, y.
    #     :param point: [m] x, y coordinates in global frame.
    #     :param radius [m] floating number about vector map query range.
    #     :param layers: desired layers to check.
    #     :return: A dictionary mapping SemanticMapLayers to lists of map objects.
    #     """
    #     pass

    # @abc.abstractmethod
    # def get_map_object(self, object_token: str, layer: SemanticMapLayer) -> Optional[MapObject]:
    #     """获取具有给定对象 ID 的 map 对象。
    #     Gets the map object with the given object id.
    #     :param object_token: desired unique id of a map object that should be extracted.
    #     :param layer: A semantic layer to query.
    #     :return: a map object if object corresponding to object_token exists else None.
    #     """
    #     pass

    # @abc.abstractmethod
    # def get_distance_to_nearest_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Tuple[Optional[str], Optional[float]]:
    #     """获取到最近的所需表面的距离（以米为单位）;该距离是从点到表面上最近位置的 L1 范数。
    #     Gets the distance (in meters) to the nearest desired surface; that distance is the L1 norm from the point to the closest location on the surface.
    #     :param point: [m] x, y coordinates in global frame.
    #     :param layer: A semantic layer to query.
    #     :return: The surface ID and the distance to the surface if there is one. If there isn't, then -1 and np.NaN will
    #         be returned for the surface ID and distance to the surface respectively.
    #     """
    #     pass

    # @abc.abstractmethod
    # def get_distance_to_nearest_raster_layer(self, point: Point2D, layer: SemanticMapLayer) -> float:
    #     """获取到最近的栅格图层的距离（以米为单位）;该距离是从点到表面上最近位置的 L1 范数。
    #     Gets the distance (in meters) to the nearest raster layer; that distance is the L1 norm from the point to the closest location on the surface.
    #     :param point: [m] x, y coordinates in global frame.
    #     :param layer: A semantic layer to query.
    #     :return: he distance to the surface if available, else None if the associated spatial map query failed.
    #     @raise ValueError if layer does not exist
    #     """
    #     pass

    # @abc.abstractmethod
    # def get_distances_matrix_to_nearest_map_object(self, points: List[Point2D], layer: SemanticMapLayer) -> Optional[npt.NDArray[np.float64]]:
    #     """返回点列表与其最近的所需表面之间的距离矩阵（以米为单位）。该距离是从点到表面上最近位置的 L1 范数。
    #     Returns the distance matrix (in meters) between a list of points and their nearest desired surface.that distance is the L1 norm from the point to the closest location on the surface.
    #     :param points: [m] A list of x, y coordinates in global frame.
    #     :param layer: A semantic layer to query.
    #     :return: An array of shortest distance from each point to the nearest desired surface.
    #     """
    #     pass
