import abc
from functools import cached_property
from typing import List, Optional, Tuple

from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphEdgeRefPathMapObject
from devkit.sim_engine.map_manager.abstract_map_objects import PolylineMapObject


class ReferencePathBase(GraphEdgeRefPathMapObject):
    """
    Class representing Reference Path Base
    """

    @cached_property
    def left_boundary(self) -> PolylineMapObject:
        """
        Getter function for obtaining the left boundary of the lane.
        :return: Left boundary of the lane.
        """
        # TODO 待完成；先修改 map.json 1.7
        pass

    @cached_property
    def right_boundary(self) -> PolylineMapObject:
        """
        Getter function for obtaining the right boundary of the lane.
        :return: Right boundary of the lane.
        """
        # TODO 待完成；先修改 map.json 1.7
        pass

    @cached_property
    def speed_limit_mps(self) -> Optional[float]:
        """
        Getter function for obtaining the speed limit of the lane.
        :return: [m/s] Speed limit.
        """
        # TODO
        return 16.7  # 16.7 (60 Km/h)

    @cached_property
    def opposite_lane_path_token(self) -> GraphEdgeRefPathMapObject:
        """对向车道的参考路径"""
        # TODO 待完成；# semantic_map: MineSimSemanticMapJsonLoader):
        pass

    # def get_width_left_right(self, point: Point2D, using_broken_ine: bool = False) -> Tuple[float, float]:
    #     """
    #     Gets distance to left and right sides of the lane from point.
    #     :param point: Point in global frame.
    #     :return: The distance to left and right sides of the lane. If the query is invalid, inf is returned.
    #         If point is outside the GraphEdgeRefPathMapObject and cannot be projected onto the GraphEdgeRefPathMapObject and
    #         include_outside is True then the distance to the edge on the nearest end is returned.

    #     获取从点到 Path 左侧和右侧的距离。
    #     NOTE： 若没有中间隔离带，返回Road 左右边界。
    #     :p aram point：全局帧中的点。
    #     :p aram using_broken_ine: road_centerline
    #         - True,   使用的道路中心虚线;
    #         - False， 不使用的道路中心虚线
    #     ：return： 到车道左侧和右侧的距离。如果查询无效，则返回 inf。
    #     """
    #     # TODO  获取从点到 Path 左侧和右侧的距离。# 先修改 map.json 1.7
    #     pass

    # def get_road_centerline(self, point: Point2D, include_outside: bool = False) -> Tuple[float, float]:
    #     # TODO minesim 地图增加虚拟的 road_centerline ; 修改 map.json 1.8 @ LI Zheng

    #     pass
