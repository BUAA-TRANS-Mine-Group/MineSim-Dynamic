from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphTopologyNodeMapObject
from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphEdgeRefPathMapObject
from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapJsonLoader
from devkit.sim_engine.map_manager.minesim_map.minesim_map_factory import get_graph_edges_and_nodes


class MineSimGraph:
    """
    A class for representing a directed graph consisting of nodes and edges.
    Designed for pathfinding and graph-based operations in the MineSim project.
    """

    def __init__(self, map_name: str):
        """
        Initializes an empty directed graph with nodes and edges.
        """
        self.map_name = map_name
        self.nodes: Dict[str, GraphTopologyNodeMapObject] = {}  # Store node objects with their tokens as keys
        # 记录 start_dunbins_token end_dunbins_token 索引的 GraphEdgeRefPathMapObject
        self.edges: Dict[str, Dict[str, GraphEdgeRefPathMapObject]] = {}  # Directed adjacency list storing edges with tokens

        self.meta_node_objects: Dict[str, GraphTopologyNodeMapObject] = None
        self.meta_edge_objects: Dict[str, GraphEdgeRefPathMapObject] = None

    def add_node(self, node: GraphTopologyNodeMapObject):
        """
        Adds a node to the graph.

        :param node: The node object to be added.
        """
        if node.token not in self.nodes:
            self.nodes[node.token] = node
            self.edges[node.token] = {}  # Initialize adjacency list for the node 初始化节点的邻接 dict
            # logger.info(f"#log# Node {node.token} added to the graph.")
        else:
            pass
            # logger.warning(f"#log# Node {node.token} already exists.")

    def add_edge(self, edge: GraphEdgeRefPathMapObject):
        """
        Adds a directed edge to the graph.

        :param edge: The edge object containing start and end nodes.
        """
        start_token = edge.start_topology_node.token
        end_token = edge.end_topology_node.token

        if start_token in self.nodes and end_token in self.nodes:
            self.edges[start_token][end_token] = edge
            # logger.info(f"#log# Edge added from {start_token} to {end_token}.")
        else:
            logger.error(f"#log# Check path={edge.token}, Cannot add edge, one or both nodes {start_token}, {end_token} do not exist.")

    def get_outgoing_node_token_neighbors(self, node_token: str) -> List[str]:
        """
        Retrieves the neighbors (outgoing nodes) of a given node.

        :param node_token: Token of the node whose neighbors are required.
        :return: List of neighbor node tokens.
        """
        # TODO
        if node_token in self.edges:
            return [neighbor[0] for neighbor in self.edges[node_token]]
        return []

    def get_outgoing_path_token_neighbors(self, path_token: str) -> List[str]:
        if path_token in self.meta_edge_objects:
            return self.meta_edge_objects[path_token].outgoing_path_tokens
        return []

    def build_graph(self, edge_objects: Dict[str, GraphEdgeRefPathMapObject], node_objects: Dict[str, GraphTopologyNodeMapObject]):
        """
        Builds the graph by adding nodes and edges from provided data.

        :param edge_objects: List of graph edges.
        :param node_objects: Dictionary of graph nodes.
        """
        self.meta_node_objects = node_objects
        self.meta_edge_objects = edge_objects
        logger.info(f"#log# map={self.map_name}: Building graph with provided nodes and edges...")

        # Add nodes to the graph
        for node_token, node in node_objects.items():
            self.add_node(node)

        # Add edges based on reference paths
        for refpath_token, edge in edge_objects.items():
            self.add_edge(edge)

    def display_graph(self):
        """
        Displays the current graph structure (nodes and edges).
        """
        # logger.info("Displaying graph structure:")
        for node, neighbors in self.edges.items():
            neighbor_tokens = [n[0] for n in neighbors]
            print(f"#log# Node {node} -> {neighbor_tokens}")

    def offline_check_graph(self):
        # TODO
        pass


# Example usage
@lru_cache(maxsize=2)
def create_minesim_graph(map_name: str, semantic_map: MineSimSemanticMapJsonLoader) -> MineSimGraph:
    """
    Creates a graph from the semantic map.

    :param map_name: The name of the map.
    :param semantic_map: The semantic map loader instance.
    :return: A MineSimGraph instance.
    """
    all_graph_edge_refpath, all_graph_node_dubinspose = get_graph_edges_and_nodes(map_name, semantic_map)

    graph = MineSimGraph(map_name=map_name)
    graph.build_graph(edge_objects=all_graph_edge_refpath, node_objects=all_graph_node_dubinspose)
    # graph.display_graph()

    return graph


# Sample function call
# mine_map_api = MineSimSemanticMapJsonLoader("path_to_map.json")
# graph = create_graph(map_name="example_map", semantic_map=mine_map_api.semantic_map)
