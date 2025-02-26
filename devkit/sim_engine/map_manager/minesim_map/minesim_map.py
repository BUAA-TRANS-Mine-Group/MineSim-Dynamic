from collections import defaultdict
from functools import lru_cache
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, cast

import geopandas
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely.geometry as geom

from devkit.common.actor_state.state_representation import Point2D
from devkit.sim_engine.map_manager.abstract_map import AbstractMap
from devkit.sim_engine.map_manager.abstract_map import MapObject
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import GeometricLayer
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import SemanticMapLayer
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import ReferencePathType
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import BorderlineType
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import DubinsPoseType
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import RasterLayer

from devkit.sim_engine.map_manager.minesim_map_data.minesim_bitmap_png_loader import MineSimBitMapPngLoader
from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapJsonLoader
from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapExplorer
from devkit.sim_engine.map_manager.abstract_map_objects import DubinsPoseMapObject
from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphEdgeRefPathMapObject
from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphTopologyNodeMapObject
from devkit.sim_engine.map_manager.minesim_map.reference_path_base import ReferencePathBase
from devkit.sim_engine.map_manager.minesim_map.reference_path_connector import ReferencePathConnector

logger = logging.getLogger(__name__)  # 获取一个与当前模块同名的日志记录器


class MineSimMap(AbstractMap):
    """
    MineSimMap implementation of Map API.
    """

    def __init__(self, map_name: str, bitmap_png_loader: MineSimBitMapPngLoader, semanticmap_json_loader: MineSimSemanticMapJsonLoader) -> None:
        """Initializes the MineSim map class."""
        self._map_name = map_name
        self.raster_bit_map = bitmap_png_loader
        self.semantic_map = semanticmap_json_loader
        self._map_objects: Dict[SemanticMapLayer, Dict[str, MapObject]] = defaultdict(dict)
        # TODO
        # self._map_object_getter: Dict[SemanticMapLayer, Callable[[str], MapObject]] = {
        #     SemanticMapLayer.ROAD_SEGMENT: self._get_lane,
        #     SemanticMapLayer.INTERSECTION: self._get_lane_connector,
        #     SemanticMapLayer.LOADING_AREA: self._get_roadblock,
        #     SemanticMapLayer.UNLOADING_AREA: self._get_roadblock_connector,
        #     SemanticMapLayer.ROAD_BLOCK: self._get_stop_line,
        #     SemanticMapLayer.REFERENCE_PATH: self._get_crosswalk,
        #     SemanticMapLayer.BORDERLINE: self._get_intersection,
        #     SemanticMapLayer.DUBINS_POSE: self._get_walkway,
        # }

        self._vector_layer_mapping = {
            SemanticMapLayer.ROAD_SEGMENT: "road",
            SemanticMapLayer.INTERSECTION: "intersection",
            SemanticMapLayer.LOADING_AREA: "loading_area",
            SemanticMapLayer.UNLOADING_AREA: "unloading_area",
            SemanticMapLayer.ROAD_BLOCK: "road_block",
            SemanticMapLayer.REFERENCE_PATH: "reference_path",
            SemanticMapLayer.BORDERLINE: "borderline",
            SemanticMapLayer.DUBINS_POSE: "dubins_pose",
        }
        self.all_graph_edge_refpath: List[GraphEdgeRefPathMapObject] = None
        self.all_graph_node_dubins: List[GraphTopologyNodeMapObject] = None

    def __reduce__(self) -> Tuple[Type["MineSimMap"], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        This object is reconstructed by pickle to avoid serializing potentially large state/caches.
        :return: Object type and constructor arguments to be used.
        """
        return self.__class__, (self._map_name, self.raster_bit_map, self.semantic_map)

    @property
    def map_name(self) -> str:
        """Inherited, see superclass."""
        return self._map_name

    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        """Inherited, see superclass."""
        return list(self._map_object_getter.keys())

    def get_available_raster_layers(self) -> List[SemanticMapLayer]:
        """Inherited, see superclass."""
        return list(self._raster_layer_mapping.keys())

    def get_raster_map(self) -> MineSimBitMapPngLoader:
        """Inherited, see superclass."""

        return self.raster_bit_map

    def load_bitmap_using_utm_local_range(
        self, utm_local_range: Tuple[float, float, float, float] = (0.0, 0.0, 0.1, 0.1), x_margin: float = 10, y_margin: float = 10
    ) -> np.ndarray:
        self.raster_bit_map.load_bitmap_using_utm_local_range(utm_local_range=utm_local_range, x_margin=x_margin, y_margin=y_margin)

    def get_all_graph_node_dubins(self) -> List[GraphTopologyNodeMapObject]:
        pass

    def initialize_all_layers(self) -> None:
        pass

    # def get_proximal_map_objects(self, point: Point2D, radius: float, layers: List[SemanticMapLayer]) -> Dict[SemanticMapLayer, List[MapObject]]:
    #     """Inherited, see superclass."""
    #     x_min, x_max = point.x - radius, point.x + radius
    #     y_min, y_max = point.y - radius, point.y + radius
    #     patch = geom.box(x_min, y_min, x_max, y_max)

    #     supported_layers = self.get_available_map_objects()
    #     unsupported_layers = [layer for layer in layers if layer not in supported_layers]

    #     assert len(unsupported_layers) == 0, f"Object representation for layer(s): {unsupported_layers} is unavailable"

    #     object_map: Dict[SemanticMapLayer, List[MapObject]] = defaultdict(list)

    #     for layer in layers:
    #         object_map[layer] = self._get_proximity_map_object(patch, layer)

    #     return object_map

    # def get_distance_to_nearest_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Tuple[Optional[str], Optional[float]]:
    #     """Inherited from superclass."""
    #     surfaces = self._get_vector_map_layer(layer)

    #     if surfaces is not None:
    #         surfaces["distance_to_point"] = surfaces.apply(lambda row: geom.Point(point.x, point.y).distance(row.geometry), axis=1)
    #         surfaces = surfaces.sort_values(by="distance_to_point")

    #         # A single surface might be made up of multiple polygons (due to an old practice of annotating a long
    #         # surface with multiple polygons; going forward there are plans by the mapping team to update the maps such
    #         # that one surface is covered by at most one polygon), thus we simply pick whichever polygon is closest to
    #         # the point.
    #         nearest_surface = surfaces.iloc[0]
    #         nearest_surface_id = nearest_surface.fid
    #         nearest_surface_distance = nearest_surface.distance_to_point
    #     else:
    #         nearest_surface_id = None
    #         nearest_surface_distance = None

    #     return nearest_surface_id, nearest_surface_distance

    # def get_distance_to_nearest_raster_layer(self, point: Point2D, layer: SemanticMapLayer) -> float:
    #     """Inherited from superclass"""
    #     raise NotImplementedError

    # def get_distances_matrix_to_nearest_map_object(self, points: List[Point2D], layer: SemanticMapLayer) -> Optional[npt.NDArray[np.float64]]:
    #     """
    #     Returns the distance matrix (in meters) between a list of points and their nearest desired surface.
    #         That distance is the L1 norm from the point to the closest location on the surface.
    #     :param points: [m] A list of x, y coordinates in global frame.
    #     :param layer: A semantic layer to query.
    #     :return: An array of shortest distance from each point to the nearest desired surface.
    #     """
    #     surfaces = self._get_vector_map_layer(layer)

    #     if surfaces is not None:
    #         # Construct geo series
    #         corner_points = geopandas.GeoSeries([geom.Point(point.x, point.y) for point in points])

    #         # Distance
    #         distances = surfaces.geometry.apply(lambda g: corner_points.distance(g))

    #         # Distance to the nearest surface
    #         distances = np.asarray(distances.min())
    #         return cast(npt.NDArray[np.float64], distances)
    #     else:
    #         return None

    #     """
    #     Load all layers to vector map
    #     :param: None
    #     :return: None
    #     """
    #     for layer_name in self._vector_layer_mapping.values():
    #         self._load_vector_map_layer(layer_name)
    #     for layer_name in self._raster_layer_mapping.values():
    #         self._load_vector_map_layer(layer_name)
    #     self._load_vector_map_layer(self._LANE_CONNECTOR_POLYGON_LAYER)

    # def _semantic_vector_layer_map(self, layer: SemanticMapLayer) -> str:
    #     """
    #     Mapping from SemanticMapLayer int to MapsDB internal representation of vector layers.
    #     :param layer: The querired semantic map layer.
    #     :return: A internal layer name as a string.
    #     @raise ValueError if the requested layer does not exist for MapsDBMap
    #     """
    #     try:
    #         return self._vector_layer_mapping[layer]
    #     except KeyError:
    #         raise ValueError("Unknown layer: {}".format(layer.name))

    # def _get_vector_map_layer(self, layer: SemanticMapLayer) -> VectorLayer:
    #     """Inherited, see superclass."""
    #     layer_id = self._semantic_vector_layer_map(layer)
    #     return self._load_vector_map_layer(layer_id)

    # def _get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[MapObject]:
    #     """
    #     Gets a list of lanes where its polygon overlaps the queried point.
    #     :param point: [m] x, y coordinates in global frame.
    #     :return: a list of lanes. An empty list if no lanes were found.
    #     """
    #     if layer == SemanticMapLayer.LANE_CONNECTOR:
    #         return self._get_all_lane_connectors(point)
    #     else:
    #         layer_df = self._get_vector_map_layer(layer)
    #         ids = layer_df.loc[layer_df.contains(geom.Point(point.x, point.y))]["fid"].tolist()

    #         return [self.get_map_object(map_object_id, layer) for map_object_id in ids]

    # def _get_proximity_map_object(self, patch: geom.Polygon, layer: SemanticMapLayer) -> List[MapObject]:
    #     """
    #     Gets nearby lanes within the given patch.
    #     :param patch: The area to be checked.
    #     :param layer: desired layer to check.
    #     :return: A list of map objects.
    #     """
    #     layer_df = self._get_vector_map_layer(layer)
    #     map_object_ids = layer_df[layer_df["geometry"].intersects(patch)]["fid"]

    #     return [self.get_map_object(map_object_id, layer) for map_object_id in map_object_ids]
