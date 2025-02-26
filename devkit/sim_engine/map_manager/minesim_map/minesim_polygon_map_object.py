from functools import cached_property
import logging
from typing import Dict, List

from shapely.geometry import Polygon

from devkit.sim_engine.map_manager.abstract_map_objects import PolygonMapObject
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import MineSimMapLayer
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import minesim_map_layer_names
from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapJsonLoader

logger = logging.getLogger(__name__)


class MineSimPolygonMetaDataType:
    """include:
    semantic_map.polygon[3]['token']
    semantic_map.polygon[3]['type']
    semantic_map.polygon[3]['link_node_tokens']
    semantic_map.polygon[3]['link_borderline_tokens']
    semantic_map.polygon[3]['link_dubinspose_tokens']
    semantic_map.polygon[3]['link_referencepath_tokens']
    """

    def __init__(self, token, type, link_node_tokens, link_borderline_tokens, link_dubinspose_tokens, link_referencepath_tokens):
        # e.g.
        # self.token = "polygon-3"
        # self.type = "intersection"
        # self.link_node_tokens = ['node-42', 'node-43', 'node-44', 'node-45', 'node-46', 'node-47', 'node-48', 'node-49', 'node-50', 'node-51', 'node-52', 'node-53', 'node-54', 'node-55']
        # self.link_borderline_tokens = ["borderline-16", "borderline-44", "borderline-73"]
        # self.link_dubinspose_tokens = ["dubinspose-269", "dubinspose-270", "dubinspose-271", "dubinspose-292"]
        # self.link_referencepath_tokens = ["path-72", "path-73", "path-74", "path-75", "path-76", "path-77"]
        self.token = token
        self.type = type
        self.link_node_tokens = link_node_tokens
        self.link_borderline_tokens = link_borderline_tokens
        self.link_dubinspose_tokens = link_dubinspose_tokens
        self.link_referencepath_tokens = link_referencepath_tokens

    def format(self):
        return {
            "token": self.token,
            "type": self.type,
            "link_node_tokens": self.link_node_tokens,
            "link_borderline_tokens": self.link_borderline_tokens,
            "link_dubinspose_tokens": self.link_dubinspose_tokens,
            "link_referencepath_tokens": self.link_referencepath_tokens,
        }


class MineSimPolygonMapObject(PolygonMapObject):
    """
    MineSimMap implementation of Polygon Map Object.
    """

    def __init__(self, generic_polygon_area_token: str, generic_polygon_area_data: dict, semantic_map: MineSimSemanticMapJsonLoader):
        """
        Constructor of generic polygon map layer.
        :param generic_polygon_area_token: Generic polygon area token.
        :param generic_polygon_area: Generic polygon area.
        """
        super().__init__(object_token=generic_polygon_area_token)
        self.polygon_id = semantic_map.getind(layer_name=MineSimMapLayer.POLYGON.fullname, token=generic_polygon_area_token)
        self.polygon_meta_data: MineSimPolygonMetaDataType = self._get_polygon_meta_data(
            generic_polygon_area_data=semantic_map.polygon[self.polygon_id]
        )
        self.polygon_nodes: List[Dict] = self._get_polygon_nodes(semantic_map=semantic_map)

    # @cached_property 是一个装饰器，用于将类方法转换为只读属性，并缓存其计算结果。
    # 缓存特性：
    # 方法首次调用时，返回值会被计算并缓存。
    # 后续访问该属性时，直接返回缓存值，而不会重新计算。
    # 避免重复计算，提高性能，特别适合计算代价较高的属性。
    # 功能：在第一次访问属性时计算值，并缓存计算结果。
    # 用途：适合需要延迟计算但只需计算一次的属性。
    @cached_property
    def polygon(self) -> Polygon:
        """Inherited from superclass."""
        link_coords = [(node["x"], node["y"]) for node in self.polygon_nodes]
        return Polygon(link_coords)

    @cached_property
    def type(self) -> MineSimMapLayer:
        """Inherited from superclass."""
        if self.polygon_meta_data.type == "intersection":
            element_type = MineSimMapLayer.INTERSECTION
        elif self.polygon_meta_data.type == "road":
            element_type = MineSimMapLayer.ROAD_SEGMENT
        elif self.polygon_meta_data.type == "load":
            element_type = MineSimMapLayer.LOADING_AREA
        elif self.polygon_meta_data.type == "unload":
            element_type = MineSimMapLayer.UNLOADING_AREA
        else:
            logger.error(f"#log# polygon token={self.polygon_meta_data.token},type={self.polygon_meta_data.type} is error!!!")
        return element_type

    @property
    def link_dubinspose_tokens(self) -> List[str]:
        """Inherited from superclass."""
        return self.polygon_meta_data.link_dubinspose_tokens

    @property
    def link_referencepath_tokens(self) -> List[str]:
        return self.polygon_meta_data.link_referencepath_tokens

    @property
    def link_borderline_tokens(self) -> List[str]:
        """Inherited from superclass."""
        return self.polygon_meta_data.link_borderline_tokens

    @property
    def link_node_tokens(self) -> List[str]:
        """Inherited from superclass."""
        return self.polygon_meta_data.link_node_tokens

    def _get_polygon_meta_data(self, generic_polygon_area_data: dict) -> MineSimPolygonMetaDataType:
        return MineSimPolygonMetaDataType(
            token=generic_polygon_area_data["token"],
            type=generic_polygon_area_data["type"],
            link_node_tokens=generic_polygon_area_data["link_node_tokens"],
            link_borderline_tokens=generic_polygon_area_data["link_borderline_tokens"],
            link_dubinspose_tokens=generic_polygon_area_data["link_dubinspose_tokens"],
            link_referencepath_tokens=generic_polygon_area_data["link_referencepath_tokens"],
        )

    def _get_polygon_nodes(self, semantic_map: MineSimSemanticMapJsonLoader) -> List[Dict]:
        _polygon_nodes = []
        for node_token in self.link_node_tokens():
            node_id = semantic_map.getind(layer_name=MineSimMapLayer.NODE.fullname, token=node_token)
            _polygon_nodes.append(semantic_map.node[node_id])

        return _polygon_nodes
