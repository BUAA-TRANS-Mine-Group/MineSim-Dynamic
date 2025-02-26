from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Tuple, Type

from devkit.sim_engine.map_manager.abstract_map_factory import AbstractMapFactory
from devkit.sim_engine.map_manager.minesim_map.minesim_map import MineSimMap
from devkit.sim_engine.map_manager.minesim_map_data.minesim_bitmap_png_loader import MineSimBitMapPngLoader
from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapJsonLoader
from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphTopologyNodeMapObject
from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphEdgeRefPathMapObject

from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import SemanticMapLayer
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import ReferencePathType
from devkit.sim_engine.map_manager.minesim_map.reference_path_base import ReferencePathBase
from devkit.sim_engine.map_manager.minesim_map.reference_path_connector import ReferencePathConnector

import logging

logger = logging.getLogger(__name__)


class MineSimMapFactory(AbstractMapFactory):
    """
    Factory creating maps from an IMapsDB interface.
    # TODO 以后再修改，MineSimMapFactory 用于加载和管理 ALL 2个区域的地图.
    """

    def __init__(self, bitmap_png_loader: MineSimBitMapPngLoader, semanticmap_json_loader: MineSimSemanticMapJsonLoader):
        """
        :param maps_db: An IMapsDB instance e.g. GPKGMapsDB.
        """
        self._bitmap_png_loader = bitmap_png_loader
        self._semanticmap_json_loader = semanticmap_json_loader

    def build_map_from_name(self, map_name: str) -> MineSimMap:
        """
        Builds a map interface given a map name.
        Examples of names: 'sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood'
        :param map_name: Name of the map.
        :return: The constructed map interface.
        """
        return MineSimMap(map_name=map_name, bitmap_png_loader=self._bitmap_png_loader, semanticmap_json_loader=self._semanticmap_json_loader)


# 加载 PNG mask 次数
@lru_cache(maxsize=2)
def get_mine_maps_mask_png(map_root: str, location: str) -> MineSimBitMapPngLoader:
    return MineSimBitMapPngLoader(map_root=map_root, location=location, bitmap_type="bitmap_mask", is_transform_gray=True)


# 加载JSON 次数；
@lru_cache(maxsize=2)
def get_mine_maps_semantic_json(map_root: str, location: str) -> MineSimSemanticMapJsonLoader:
    """MineSimSemanticMapJsonLoader
    @lru_cache(maxsize=32)：
    功能：这是一个装饰器，用来缓存函数的返回值。
    lru_cache 是 Python 标准库 functools 提供的一个装饰器，用于实现最近最少使用（Least Recently Used, LRU）缓存。
    maxsize=32 表示缓存的最大条目数为 32。如果缓存已满且有新条目被添加，最近最少使用的条目将被删除。
    使用缓存的好处是，对于同一组输入参数，函数只会执行一次，后续调用将直接从缓存中获取结果，从而提高性能。
    """
    return MineSimSemanticMapJsonLoader(map_root=map_root, location=location)


@lru_cache(maxsize=2)
def get_maps_api(map_root: str, map_name: str) -> MineSimMap:
    """
    NOTE:  devkit/scenario_builder/minesim_scenario_json/minesim_dynamic_scenario.py 加载地图的核心函数
    Get a MineSimMap object corresponding to a particular set of parameters.
    :param map_root: The root folder for the map data.
    :param map_name: The map name to load. map_names=["jiangxi_jiangtong","guangdong_dapai"]
    :return: The loaded MineSimMap object.
    """
    bitmap_png_loader = get_mine_maps_mask_png(map_root=map_root, location=map_name)
    semanticmap_json_loader = get_mine_maps_semantic_json(map_root=map_root, location=map_name)

    return MineSimMap(map_name=map_name, bitmap_png_loader=bitmap_png_loader, semanticmap_json_loader=semanticmap_json_loader)


@lru_cache(maxsize=2)
def get_graph_edges_and_nodes(
    map_name: str, semantic_map: MineSimSemanticMapJsonLoader
) -> Tuple[Dict[str, GraphEdgeRefPathMapObject], Dict[str, GraphTopologyNodeMapObject]]:
    """
    Lazily loads all graph edge refpaths and nodes for a given map.

    :param map_name: Name of the map.
    :param semantic_map: Instance of MineSimSemanticMapJsonLoader containing map data.
    :return: A tuple containing:
        - Dictionary mapping refpath tokens to  GraphEdgeRefPathMapObject instances.
        - Dictionary mapping node tokens to GraphTopologyNodeMapObject instances.
    """
    logging.info(f"#log# Loading all graph edge refpaths for map: {map_name}")

    all_graph_edge_refpath: Dict[GraphEdgeRefPathMapObject] = {}
    all_graph_node_dubinspose: Dict[str, GraphTopologyNodeMapObject] = {}

    for ref_path in semantic_map.reference_path:
        token = ref_path.get("token")
        ref_path_type = ref_path.get("type")

        # Ensure type exists and is valid
        if not token or not ref_path_type:
            logging.warning(f"#log# Missing 'token' or 'type' in reference path: {ref_path}")
            continue

        if ref_path_type == ReferencePathType.BASE_PATH.fullname:
            graph_edge_refpath = ReferencePathBase(path_token=token, semantic_map=semantic_map)
        elif ref_path_type == ReferencePathType.CONNECTOR_PATH.fullname:
            graph_edge_refpath = ReferencePathConnector(path_token=token, semantic_map=semantic_map)
        else:
            logging.error(f"#log# Invalid ReferencePathType for path_token={token}, type={ref_path_type}")
            continue

        # Add unique topology nodes to dictionary
        if graph_edge_refpath.start_topology_node.token not in all_graph_node_dubinspose:
            all_graph_node_dubinspose[graph_edge_refpath.start_topology_node.token] = graph_edge_refpath.start_topology_node

        if graph_edge_refpath.end_topology_node.token not in all_graph_node_dubinspose:
            all_graph_node_dubinspose[graph_edge_refpath.end_topology_node.token] = graph_edge_refpath.end_topology_node

        if graph_edge_refpath.token not in all_graph_node_dubinspose:
            all_graph_edge_refpath[graph_edge_refpath.token] = graph_edge_refpath

    logging.info(f"#log# Loaded {len(all_graph_edge_refpath)} graph edges and {len(all_graph_node_dubinspose)} nodes for map: {map_name}")
    return all_graph_edge_refpath, all_graph_node_dubinspose


if __name__ == "__main__":
    # 测试代码 MineSim-Dynamic/devkit/sim_engine/map_manager/minesim_map/test/test_minesim_map_factory.py
    pass
