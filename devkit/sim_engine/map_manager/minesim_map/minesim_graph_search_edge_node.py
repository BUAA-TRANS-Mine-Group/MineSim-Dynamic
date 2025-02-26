from __future__ import annotations

from functools import lru_cache
from functools import cached_property
import abc
import logging
from typing import Dict, List, Optional, Tuple, Union

from shapely.geometry import LineString, Point, Polygon

from devkit.sim_engine.map_manager.abstract_map_objects import AbstractMapObject
from devkit.sim_engine.map_manager.abstract_map_objects import DubinsPoseMapObject

from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import MineSimMapLayer
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import ReferencePathType
from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapJsonLoader
from devkit.sim_engine.map_manager.abstract_map_objects import PolylineMapObject

logger = logging.getLogger(__name__)


class GraphTopologyNodeMapObject(DubinsPoseMapObject):
    """
    NOTE: Directed Graph Representation: Nodes and Edges; (图搜索算法)图的表示:节点与边;

    用于连接参考路径的 DubinsPose, 有这三种类型：
    DubinsPoseType.MERGE = 0
    DubinsPoseType.SPLIT = 1
    DubinsPoseType.NORMAL = 2 # 用于在road中连接两条 base path
    """

    def __init__(self, dubinspose_token: str, semantic_map: MineSimSemanticMapJsonLoader):
        super().__init__(dubinspose_token=dubinspose_token, semantic_map=semantic_map)
        self.parent_node_tokens, self.child_node_tokens, self.incoming_path_tokens, self.outgoing_path_tokens = self._search_parent_and_child_nodes(
            semantic_map=semantic_map
        )

    def _search_parent_and_child_nodes(self, semantic_map: MineSimSemanticMapJsonLoader):
        """Searches for parent and child nodes of the current node within the map."""
        try:
            link_polygon_id = semantic_map.getind(layer_name=MineSimMapLayer.POLYGON.fullname, token=self.link_polygon_token)
        except KeyError:
            logging.error(f"#log# Error: {self.token}: link_polygon_token '{self.link_polygon_token}' not found in semantic map.")
            return [], [], [], []

        parent_node_tokens: List[str] = []
        child_node_tokens: List[str] = []
        incoming_path_tokens: List[str] = []
        outgoing_path_tokens: List[str] = []

        for link_referencepath_token in semantic_map.polygon[link_polygon_id]["link_referencepath_tokens"]:
            path_id = semantic_map.getind(layer_name=MineSimMapLayer.REFERENCE_PATH.fullname, token=link_referencepath_token)
            link_dubinspose_tokens = semantic_map.reference_path[path_id]["link_dubinspose_tokens"]
            if link_dubinspose_tokens[0] == self.token:
                child_node_tokens.append(link_dubinspose_tokens[-1])
                outgoing_path_tokens.append(link_referencepath_token)
            elif link_dubinspose_tokens[-1] == self.token:
                parent_node_tokens.append(link_dubinspose_tokens[0])
                incoming_path_tokens.append(link_referencepath_token)
        pass
        # Log an error if no parent or child nodes were found
        if not parent_node_tokens and not child_node_tokens:
            logging.error(
                f"#log# Error: No valid parent/child nodes found for reference_path {link_referencepath_token}. "
                f"link_dubinspose_tokens={link_dubinspose_tokens}"
            )

        return parent_node_tokens, child_node_tokens, incoming_path_tokens, outgoing_path_tokens


@lru_cache(maxsize=1000)
def get_graph_topology_node_dunbins(dubinspose_token: str, semantic_map: MineSimSemanticMapJsonLoader) -> GraphTopologyNodeMapObject:
    """
    NOTE:  随机的加载 ALL GraphTopologyNodeMapObject， 大量计算防止重复；
    """
    return GraphTopologyNodeMapObject(dubinspose_token=dubinspose_token, semantic_map=semantic_map)


# Search Graph : Line + Node
class GraphEdgeRefPathMapObject(AbstractMapObject):
    """
    参考路径 作为图搜索算法的  Graph: Edge
    一个类，用于表示地图 graph connectivity ; MineSime Map 通过参考路径来构建路网的关联关系
    A class to represent the map graph connectivity.
    """

    def __init__(self, path_token: str, semantic_map: MineSimSemanticMapJsonLoader):
        super().__init__(object_token=path_token)

        self.path_id = semantic_map.getind(layer_name=MineSimMapLayer.REFERENCE_PATH.fullname, token=path_token)
        self.refpath_meta_data: dict = semantic_map.reference_path[self.path_id]
        self.polygon_nodes: List[Dict] = semantic_map.reference_path[self.path_id]["link_polygon_tokens"]
        self.start_topology_node: GraphTopologyNodeMapObject = get_graph_topology_node_dunbins(
            dubinspose_token=semantic_map.reference_path[self.path_id]["link_dubinspose_tokens"][0], semantic_map=semantic_map
        )
        self.end_topology_node: GraphTopologyNodeMapObject = get_graph_topology_node_dunbins(
            dubinspose_token=semantic_map.reference_path[self.path_id]["link_dubinspose_tokens"][-1], semantic_map=semantic_map
        )
        self._path_polyline: PolylineMapObject = None  # 懒加载 Lazy Loading

    #     @cached_property
    # def baseline_path(self) -> PolylineMapObject:
    #     """Inherited from superclass."""
    #     return NuPlanPolylineMapObject(get_row_with_value(self._baseline_paths_df, "lane_connector_fid", self.id))

    @cached_property
    def type(self) -> ReferencePathType:
        _type = self.refpath_meta_data["type"]
        if _type == "connector_path":
            element_type = ReferencePathType.CONNECTOR_PATH
        elif _type == "base_path":
            element_type = ReferencePathType.BASE_PATH

        else:
            logger.error(f"#log# ReferencePath token={self.token},type={_type} is error!!!")
        return element_type

    @property
    def incoming_path_tokens(self) -> List[Dict]:
        return self.refpath_meta_data["incoming_tokens"]

    @property
    def outgoing_path_tokens(self) -> List[Dict]:
        return self.refpath_meta_data["outgoing_tokens"]

    @property
    def refpath_waypoints(self) -> List[List]:
        return self.refpath_meta_data["waypoints"]

    @property
    def is_start_blocked(self) -> bool:
        return self.refpath_meta_data["is_start_blocked"]

    @property
    def is_end_blocked(self) -> bool:
        return self.refpath_meta_data["is_end_blocked"]

    @property
    def waypoint_sampling_interval_meter(self) -> bool:
        return self.refpath_meta_data["waypoint_sampling_interval_meter"]

    @property
    def link_polygon_tokens(self) -> bool:
        return self.refpath_meta_data["link_polygon_tokens"]

    @property
    def link_dubinspose_tokens(self) -> bool:
        return self.refpath_meta_data["link_dubinspose_tokens"]

    @property
    def path_polyline(self) -> PolylineMapObject:
        """懒加载 Lazy Loading
        - 使用 `@property` + `if self._xxx is None:` +  `_load_xxx()` 方式来延迟加载;
        - _xxx 默认 None, 在首次访问时，调用 `_load_xxx()` 方法进行计算并存储结果。
        - 仅在首次访问时加载数据，提高性能。
        """
        if self._path_polyline is None:
            self._path_polyline = self._load_path_polyline()
        return self._path_polyline

    def _load_path_polyline(self) -> PolylineMapObject:
        # TODO
        pass
        return None

    @property
    @abc.abstractmethod
    def speed_limit_mps(self) -> Optional[float]:
        """
        Getter function for obtaining the speed limit of the lane.
        :return: [m/s] Speed limit.
        """
        pass
