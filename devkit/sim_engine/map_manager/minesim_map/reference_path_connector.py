import abc
from functools import cached_property
from typing import List, Optional


from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphEdgeRefPathMapObject
from devkit.sim_engine.map_manager.abstract_map_objects import PolylineMapObject


class ReferencePathConnector(GraphEdgeRefPathMapObject):
    """
    Class representing Reference Path Base
    """

    @cached_property
    def speed_limit_mps(self) -> Optional[float]:
        """
        Getter function for obtaining the speed limit of the lane.
        :return: [m/s] Speed limit.
        """
        # TODO
        return 16.7  # 16.7 (60 Km/h)

    def get_intersection_polygon_borderlines(self) -> List[PolylineMapObject]:
        # TODO 待增加； 有无必要？
        pass
