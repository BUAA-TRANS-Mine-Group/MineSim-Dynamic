from typing import List, Union
import math

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
from shapely.geometry import Polygon

from devkit.common.actor_state.oriented_box import Dimension, OrientedBox
from devkit.common.actor_state.state_representation import Point2D, StateSE2
from devkit.common.actor_state.vehicle_parameters import get_mine_truck_parameters
from devkit.common.geometry.transform import translate_laterally, translate_longitudinally


def lateral_distance(reference: StateSE2, other: Point2D) -> float:
    """
    Lateral distance from a point to a reference pose
    :param reference: the reference pose
    :param other: the query point
    :return: the lateral distance
    """
    return float(-np.sin(reference.heading) * (other.x - reference.x) + np.cos(reference.heading) * (other.y - reference.y))


def longitudinal_distance(reference: StateSE2, other: Point2D) -> float:
    """
    Longitudinal distance from a point to a reference pose
    :param reference: the reference pose
    :param other: the query point
    :return: the longitudinal distance
    """
    return float(np.cos(reference.heading) * (other.x - reference.x) + np.sin(reference.heading) * (other.y - reference.y))


def signed_lateral_distance(ego_state: StateSE2, other: Polygon) -> float:
    """
    Computes the minimal lateral distance of ego from another polygon
    [Chinese] 计算ego到另一个多边形的最小横向距离
    :param ego_state: the state of ego
    :param other: the query polygon
    :return: the signed lateral distance
    """
    ego_half_width = get_mine_truck_parameters().half_width
    ego_left = translate_laterally(ego_state, ego_half_width)
    ego_right = translate_laterally(ego_state, -ego_half_width)

    vertices = list(zip(*other.exterior.coords.xy))
    distance_left = max(min(lateral_distance(ego_left, Point2D(*vertex)) for vertex in vertices), 0)
    distance_right = max(min(-lateral_distance(ego_right, Point2D(*vertex)) for vertex in vertices), 0)
    return distance_left if distance_left > distance_right else -distance_right


def signed_longitudinal_distance(ego_state: StateSE2, other: Polygon) -> float:
    """
    Computes the minimal longitudinal distance of ego from another polygon
     [Chinese] 计算ego到另一个多边形的最小longitudinal距离
    :param ego_state: the state of ego
    :param other: the query polygon
    :return: the signed lateral distance
    """
    ego_half_length = get_mine_truck_parameters().half_length
    ego_front = translate_longitudinally(ego_state, ego_half_length)
    ego_back = translate_longitudinally(ego_state, -ego_half_length)

    vertices = list(zip(*other.exterior.coords.xy))
    distance_front = max(min(longitudinal_distance(ego_front, Point2D(*vertex)) for vertex in vertices), 0)
    distance_back = max(min(-longitudinal_distance(ego_back, Point2D(*vertex)) for vertex in vertices), 0)
    return distance_front if distance_front > distance_back else -distance_back


def compute_distance(lhs: StateSE2, rhs: StateSE2) -> float:
    """
    Compute the euclidean distance between two points
    :param lhs: first point
    :param rhs: second point
    :return distance between two points
    """
    return float(np.hypot(lhs.x - rhs.x, lhs.y - rhs.y))


def compute_lateral_displacements(poses: List[StateSE2]) -> List[float]:
    """
    Computes the lateral displacements (y_t - y_t-1) from a list of poses

    :param poses: list of N poses to compute displacements from
    :return: list of N-1 lateral displacements
    """
    return [poses[idx].y - poses[idx - 1].y for idx in range(1, len(poses))]


def principal_value(angle: Union[float, int, npt.NDArray[np.float64]], min_: float = -np.pi) -> Union[float, npt.NDArray[np.float64]]:
    """
    Wrap heading angle in to specified domain (multiples of 2 pi alias),
    ensuring that the angle is between min_ and min_ + 2 pi. This function raises an error if the angle is infinite
    :param angle: rad
    :param min_: minimum domain for angle (rad)
    :return angle wrapped to [min_, min_ + 2 pi).

    principal ,主值：在数学中，特别是复分析中。
    输入的航向角归一化到指定的范围内（以 2 pi 的倍数为别名），确保角度被限制在 [min_, min_ + 2 pi) 的区间内。如果角度值是无穷大，则会抛出错误。

    :param angle: 输入的角度，单位为弧度，支持单个浮点数、整数或 numpy 数组。
    :param min_: 角度的最小值域，单位为弧度，默认为 -π。
    :return: 返回归一化后的角度，确保其在 [min_, min_ + 2 pi) 的范围内。
             输出类型与输入相同，输入为单个数值则返回单个数值，输入为数组则返回数组。

    示例:
    >>> principal_value(4 * np.pi)  # 输出 -π
    >>> principal_value(np.array([0, 2 * np.pi]))  # 输出 [0, 0]
    """
    assert np.all(np.isfinite(angle)), "angle is not finite"

    lhs = (angle - min_) % (2 * np.pi) + min_

    return lhs


def l2_euclidean_corners_distance(box1: OrientedBox, box2: OrientedBox) -> float:
    """
    Computes the L2 norm [m] of the euclidean distance between the corners of an OrientedBox in two configurations.
    :param box1: The first box configuration.
    :param box2: The second box configuration.
    :return: [m] The norm of the euclidean distance.
    """
    distances = [np.linalg.norm(box1_corner.array - box2_corner.array) for box1_corner, box2_corner in zip(box1.all_corners(), box2.all_corners())]
    return float(np.linalg.norm(distances))


def se2_box_distances(query: StateSE2, targets: list[StateSE2], box_size: Dimension, consider_flipped: bool = True) -> List[float]:
    """
    Computes the minimal distance [m] from a query to a list of targets. The distance is computed using the norm of the
    euclidean distances between the corners of a box spawned using the pose as center and given dimensions.
    The query box is also rotated by 180deg and the minimum of the two distances is used.
    :param query: The query pose.
    :param targets: The targets to compute the distance.
    :param box_size: The size of the box to be constructed.
    :param consider_flipped: Whether to also check for the same query pose, but rotated by 180 degrees.
    :return: A list of distances [m] from query to targets
    """
    query_box = OrientedBox(query, box_size.length, box_size.width, box_size.height)
    backwards_query_box = OrientedBox.from_new_pose(query_box, StateSE2(query.x, query.y, query.heading + np.pi))
    target_boxes = [OrientedBox(target, box_size.length, box_size.width, box_size.height) for target in targets]
    if consider_flipped:
        return [
            min(
                l2_euclidean_corners_distance(query_box, target_box),
                l2_euclidean_corners_distance(backwards_query_box, target_box),
            )
            for target_box in target_boxes
        ]
    else:
        return [l2_euclidean_corners_distance(query_box, target_box) for target_box in target_boxes]


class AngularInterpolator:
    """Creates an angular linear interpolator."""

    def __init__(self, states: npt.NDArray[np.float64], angular_states: npt.NDArray[np.float64]):
        """
        :param states: x values for interpolation
        :param angular_states: y values for interpolation
        """
        _angular_states = np.unwrap(angular_states, axis=0)

        self.interpolator = interp1d(states, _angular_states, axis=0)

    def interpolate(self, sampled_state: Union[float, List[float]]) -> npt.NDArray[np.float64]:
        """
        Interpolates a single state
        :param sampled_state: The state at which to perform interpolation
        :return: The value of the state interpolating linearly at the given state
        """
        return principal_value(self.interpolator(sampled_state))  # type: ignore


def angle_diff(x: float, y: float, period: float) -> float:
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity for assessing angle difference.
    :return: Signed smallest between-angle difference in range (-pi, pi).
    """
    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > math.pi:
        diff = diff - (2 * math.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff
