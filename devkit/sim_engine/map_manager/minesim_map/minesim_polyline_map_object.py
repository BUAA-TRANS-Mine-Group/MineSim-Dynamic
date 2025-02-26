import logging
import math
from functools import cached_property
from typing import Dict, List, cast

from pandas.core.series import Series
from shapely.geometry import LineString, Point

from devkit.common.actor_state.state_representation import Point2D, StateSE2
from devkit.sim_engine.map_manager.abstract_map_objects import PolylineMapObject
from devkit.sim_engine.map_manager.minesim_map.utils import estimate_curvature_along_path, extract_discrete_polyline
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import MineSimMapLayer, SemanticMapLayer
from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapJsonLoader

logger = logging.getLogger(__name__)


def _get_heading(pt1: Point, pt2: Point) -> float:
    """
    Computes the angle two points makes to the x-axis.
    :param pt1: origin point.
    :param pt2: end point.
    :return: [rad] resulting angle.
    """
    x_diff = pt2.x - pt1.x
    y_diff = pt2.y - pt1.y
    return math.atan2(y_diff, x_diff)


class MineSimPolylineRefPathMapObject(PolylineMapObject):
    """
    MineSimMap implementation of Polyline Map Object.
    """

    def __init__(
        self,
        path_token: str,
        polyline_type: SemanticMapLayer,
        semantic_map: MineSimSemanticMapJsonLoader,
        distance_for_curvature_estimation: float = 2.0,
        distance_for_heading_estimation: float = 0.2,
    ):
        """
        Constructor of polyline map layer.
        :param polyline: a pandas series representing the polyline.
        :param distance_for_curvature_estimation: [m] distance of the split between 3-points curvature estimation.
        :param distance_for_heading_estimation: [m] distance between two points on the polyline to calculate the relative heading.

        折线地图图层的构造函数。
        :p aram polyline：表示多段线的 pandas 系列。
        :p aram distance_for_curvature_estimation：[m] 3 点曲率估计之间的分割距离。
        :p aram distance_for_heading_estimation： [m] 多段线上两点之间的距离，用于计算相对航向。
        """
        super().__init__(polyline_token=path_token)
        self.polyline_type: SemanticMapLayer = polyline_type
        self.element_id = semantic_map.getind(layer_name=MineSimMapLayer.REFERENCE_PATH.fullname, token=self.token)
        # !参考路径航路点 waypoints[x坐标, y坐标, yaw航向, height海拔高度, slope坡度]，单位[m, m, rad, m, degree]； yaw航向[0,2pi]最好转化为标准的[-pi,pi]
        self.waypoints_meta_data = semantic_map.reference_path[self.element_id]["waypoints"]
        # 提取 xyz 值，忽略 heading 和 slope
        xyz_points = [(wp[0], wp[1], wp[3]) for wp in self.waypoints_meta_data]
        # 使用 xyz 初始化 LineString
        self._polyline = LineString(xyz_points)

        assert self._polyline.length > 0.0, "#log# The length of the polyline has to be greater than 0!"

        self._distance_for_curvature_estimation = distance_for_curvature_estimation
        self._distance_for_heading_estimation = distance_for_heading_estimation

    @property
    def linestring(self) -> LineString:
        """Inherited from superclass."""
        return self._polyline

    @property
    def length(self) -> float:
        """Inherited from superclass."""
        return float(self._polyline.length)

    @cached_property
    def discrete_path(self) -> List[StateSE2]:
        """Inherited from superclass."""
        return cast(List[StateSE2], extract_discrete_polyline(self._polyline))

    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        """Inherited from superclass."""
        return self._polyline.project(Point(point.x, point.y))  # type: ignore

    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        """Inherited from superclass."""
        arc_length = self.get_nearest_arc_length_from_position(point)
        state1 = self._polyline.interpolate(arc_length)
        state2 = self._polyline.interpolate(arc_length + self._distance_for_heading_estimation)

        if state1 == state2:
            # Handle the case where the queried position (state1) is at the end of the baseline path
            state2 = self._polyline.interpolate(arc_length - self._distance_for_heading_estimation)
            heading = _get_heading(state2, state1)
        else:
            heading = _get_heading(state1, state2)

        return StateSE2(state1.x, state1.y, heading)

    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        """Inherited from superclass."""
        curvature = estimate_curvature_along_path(self._polyline, arc_length, self._distance_for_curvature_estimation)

        return float(curvature)

    def _get_polyline_meta_data(self, semantic_map: MineSimSemanticMapJsonLoader) -> Dict:
        pass
        if self.polyline_type == SemanticMapLayer.BORDERLINE:
            element_id = semantic_map.getind(layer_name=MineSimMapLayer.REFERENCE_PATH.fullname, token=self.token)
            points_meta_data = semantic_map.reference_path[element_id]["waypoints"]
        elif self.polyline_type == SemanticMapLayer.REFERENCE_PATH:
            element_id = semantic_map.getind(layer_name=MineSimMapLayer.BORDERLINE.fullname, token=self.token)
            points_meta_data = semantic_map.borderline[element_id]["borderpoints"]
        else:
            logger.error(f"#log# {self.token}: _get_polyline_meta_data type  is error!!!")

        return element_id, points_meta_data


class MineSimPolylineBorderlineMapObject(PolylineMapObject):
    """
    MineSimMap implementation of Polyline Map Object.
    # TODO 修改地图：增加版本 1.7，map.json 中增离线处理，增加 有方向的 borderline (只针对 edge 类型)
        - borderpoints; start_polygon_edge_node_tokens([node-1,node-2]); end_polygon_edge_node_tokens[node-3,node-4];
        - start_polygon_token; end_polygon_token;
    """

    def __init__(
        self,
        borderline_token: str,
        polyline_type: SemanticMapLayer,
        semantic_map: MineSimSemanticMapJsonLoader,
        distance_for_curvature_estimation: float = 2.0,
        distance_for_heading_estimation: float = 0.5,
    ):
        """
        Constructor of polyline map layer.
        :param polyline: a pandas series representing the polyline.
        :param distance_for_curvature_estimation: [m] distance of the split between 3-points curvature estimation.
        :param distance_for_heading_estimation: [m] distance between two points on the polyline to calculate the relative heading.

        折线地图图层的构造函数。
        :p aram polyline：表示多段线的 pandas 系列。
        :p aram distance_for_curvature_estimation：[m] 3 点曲率估计之间的分割距离。
        :p aram distance_for_heading_estimation： [m] 多段线上两点之间的距离，用于计算相对航向。
        """
        super().__init__(polyline_token=borderline_token)
        self.polyline_type: SemanticMapLayer = polyline_type

        self.element_id = semantic_map.getind(layer_name=MineSimMapLayer.BORDERLINE.fullname, token=self.token)
        self.borderpoints_meta_data = semantic_map.borderline[self.element_id]["borderpoints"]

        # TODO 设计有方向的边界线；
        # self._polyline: LineString = _get_directed_polyline_meta_data()
        self._polyline = LineString(self.borderpoints_meta_data)

        assert self._polyline.length > 0.0, "#log# The length of the polyline has to be greater than 0!"

        self._distance_for_curvature_estimation = distance_for_curvature_estimation
        self._distance_for_heading_estimation = distance_for_heading_estimation

    @property
    def linestring(self) -> LineString:
        """Inherited from superclass."""
        return self._polyline

    @property
    def length(self) -> float:
        """Inherited from superclass."""
        return float(self._polyline.length)

    @cached_property
    def discrete_path(self) -> List[StateSE2]:
        """Inherited from superclass."""
        return cast(List[StateSE2], extract_discrete_polyline(self._polyline))

    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        """Inherited from superclass."""
        return self._polyline.project(Point(point.x, point.y))  # type: ignore

    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        """Inherited from superclass."""
        arc_length = self.get_nearest_arc_length_from_position(point)
        state1 = self._polyline.interpolate(arc_length)
        state2 = self._polyline.interpolate(arc_length + self._distance_for_heading_estimation)

        if state1 == state2:
            # Handle the case where the queried position (state1) is at the end of the baseline path
            state2 = self._polyline.interpolate(arc_length - self._distance_for_heading_estimation)
            heading = _get_heading(state2, state1)
        else:
            heading = _get_heading(state1, state2)

        return StateSE2(state1.x, state1.y, heading)

    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        """Inherited from superclass."""
        curvature = estimate_curvature_along_path(self._polyline, arc_length, self._distance_for_curvature_estimation)

        return float(curvature)
