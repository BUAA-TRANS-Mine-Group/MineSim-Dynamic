from collections import deque
import logging
from typing import Dict, List, Optional, Tuple
from typing import Any, Tuple, Type

import numpy as np
from queue import PriorityQueue

from devkit.common.actor_state.state_representation import StateSE2
from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapJsonLoader
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import MineSimMapLayer
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import ReferencePathType

logger = logging.getLogger(__name__)  # 获取一个与当前模块同名的日志记录器


def calculate_start_refpath_info(start_pose_statese2: StateSE2, semantic_map: MineSimSemanticMapJsonLoader) -> Tuple[str, int]:
    # 1. 查询 polygon
    polygon_token = semantic_map.get_polygon_token_using_node(x=start_pose_statese2.x, y=start_pose_statese2.y)
    polygon_id = semantic_map.getind(layer_name=MineSimMapLayer.POLYGON.fullname, token=polygon_token)
    if MineSimMapLayer.INTERSECTION.fullname == semantic_map.polygon[polygon_id]["type"]:
        logging.error(f"#log# polygon ={polygon_token}, type={ MineSimMapLayer.INTERSECTION.fullmame},not support, TODO!!!")
        # TODO 2.1. 查询 polygon 属于 intersection # ReferencePathConnector
    elif (
        MineSimMapLayer.LOADING_AREA.fullname == semantic_map.polygon[polygon_id]["type"]
        or MineSimMapLayer.UNLOADING_AREA.fullname == semantic_map.polygon[polygon_id]["type"]
    ):
        # TODO 待检查 2.2.查询 polygon属于 road unloading_area loading_are
        candidate_referencepath_tokens = semantic_map.polygon[polygon_id]["link_referencepath_tokens"]
    elif MineSimMapLayer.ROAD_SEGMENT.fullname == semantic_map.polygon[polygon_id]["type"]:
        candidate_referencepath_tokens = semantic_map.polygon[polygon_id]["link_referencepath_tokens"]

    # 去掉 conector path
    candidate_basepath_tokens = []
    for path_token in candidate_referencepath_tokens:
        path_id = semantic_map.getind(layer_name=MineSimMapLayer.REFERENCE_PATH.fullname, token=path_token)
        if ReferencePathType.BASE_PATH.fullname == semantic_map.reference_path[path_id]["type"]:
            candidate_basepath_tokens.append(path_token)

    if candidate_basepath_tokens:  # non []
        return get_most_matched_base_path(
            pose_statese2=start_pose_statese2, semantic_map=semantic_map, candidate_basepath_tokens=candidate_basepath_tokens
        )
    else:
        return ()


def calculate_end_refpath_info(pose_statese2: StateSE2, semantic_map: MineSimSemanticMapJsonLoader) -> Tuple[str, int]:
    # TODO 待完善
    return calculate_start_refpath_info(start_pose_statese2=pose_statese2, semantic_map=semantic_map)


def get_most_matched_base_path(
    pose_statese2: StateSE2, semantic_map: MineSimSemanticMapJsonLoader, candidate_basepath_tokens: List[str]
) -> Tuple[str, int]:
    """非 intersection polygon 内部，得到最佳匹配的 path.
    - 计算每条候选路径的最近路径点、距离以及航向角差值。
    - 优先选择距离较近且航向角差值较小的路径。

    :param pose_statese2: 当前自车位姿 (x, y, heading)
    :param semantic_map: 地图数据加载器
    :param candidate_basepath_tokens: 候选 base 类型 path 路径的 token 列表
    :return: 最匹配的路径 token 和最近路径点的 ID
    """
    candidate_paths = []
    # Step 1: 遍历所有候选路径，计算路径点的属性
    for path_token in candidate_basepath_tokens:
        path_id = semantic_map.getind(layer_name=MineSimMapLayer.REFERENCE_PATH.fullname, token=path_token)
        waypoints = np.array(semantic_map.reference_path[path_id]["waypoints"])

        # 匹配最近路径点和距离
        nearest_waypoint_id, nearest_distance = find_nearest_waypoint_by_downsampling(
            waypoints=waypoints, point_x=pose_statese2.x, point_y=pose_statese2.y, downsampling_rate=5
        )

        # 计算航向角差值 # 路径点的航向角  # 当前位姿的航向角
        yaw_diff = abs(calc_yaw_diff_two_waypoints(angle1=waypoints[nearest_waypoint_id][2], angle2=pose_statese2.heading))

        # 保存路径信息
        candidate_paths.append(
            {
                "path_token": path_token,
                "nearest_waypoint_id": nearest_waypoint_id,
                "nearest_distance": nearest_distance,
                "yaw_diff": yaw_diff,
            }
        )

    # Step 2: 按照优先级排序
    # 优先按 航向角差值排序，其次按 最近距离排序
    candidate_paths.sort(key=lambda path: (path["yaw_diff"], path["nearest_distance"]))

    # Step 3: 根据匹配条件选择路径
    if candidate_paths:
        best_path = candidate_paths[0]  # 选择排序后的第一个路径
        return best_path["path_token"], best_path["nearest_waypoint_id"]
    else:
        # 如果没有有效路径，抛出异常或返回默认值
        raise ValueError("No valid base path found for the given position.")


def get_most_matched_connector_path(
    pose_statese2: StateSE2, semantic_map: MineSimSemanticMapJsonLoader, candidate_connector_path_tokens: List[str]
) -> Tuple[str, int]:
    """
    从候选路径中选出与当前位置最匹配的路径。

    - 首先计算每条路径的最近路径点及距离。
    - 按照路径点距离排序，优先选择绝对距离小于6米且夹角最小的路径。
    - 如果没有小于6米的路径，判断是否存在距离小于10米的路径，仍选择夹角最小的路径。
    - 如果还没有符合的路径，选择所有候选路径中夹角最小的路径。

    :param pose_statese2: 当前自车位姿 (x, y, heading)
    :param semantic_map: 地图数据加载器
    :param candidate_connector_path_tokens: 候选路径的 token 列表
    :return: 最匹配的路径 token 和最近路径点的 ID
    """
    candidate_paths = []

    # Step 1: 遍历所有候选路径，计算每条路径的最近路径点和属性
    for path_token in candidate_connector_path_tokens:
        path_id = semantic_map.getind(layer_name=MineSimMapLayer.REFERENCE_PATH.fullname, token=path_token)

        # 匹配最近点
        waypoints = np.array(semantic_map.reference_path[path_id]["waypoints"])
        nearest_waypoint_id, nearest_distance = find_nearest_waypoint_by_downsampling(
            waypoints=waypoints,
            point_x=pose_statese2.x,
            point_y=pose_statese2.y,
            downsampling_rate=5,
        )

        # 计算航向角差值 # 参考路径点的航向角  # 当前 agent 车的航向角
        yaw_diff = calc_yaw_diff_two_waypoints(angle1=waypoints[nearest_waypoint_id][2], angle2=pose_statese2.heading)

        # 保存路径信息
        candidate_paths.append(
            {
                "path_token": path_token,
                "path_id": path_id,
                "nearest_waypoint_id": nearest_waypoint_id,
                "nearest_distance": nearest_distance,
                "yaw_diff": abs(yaw_diff),
            }
        )

    # Step 2: 按距离和航向角差值排序
    candidate_paths.sort(key=lambda path: (path["nearest_distance"], path["yaw_diff"]))

    # Step 3: 根据距离选择最匹配的路径
    path_within_6m = [path for path in candidate_paths if path["nearest_distance"] < 6.0]
    path_within_10m = [path for path in candidate_paths if path["nearest_distance"] < 10.0]

    if path_within_6m:
        # 如果有路径的最近点距离小于6米，选择其中夹角最小的路径
        best_path = min(path_within_6m, key=lambda path: path["yaw_diff"])
    elif path_within_10m:
        # 如果没有小于6米的路径，选择距离小于10米且夹角最小的路径
        best_path = min(path_within_10m, key=lambda path: path["yaw_diff"])
    else:
        # 如果没有小于10米的路径，选择所有路径中夹角最小的路径
        best_path = min(candidate_paths, key=lambda path: path["yaw_diff"])

    # 返回最匹配的路径 token 和最近路径点 ID
    return best_path["path_token"], best_path["nearest_waypoint_id"]


def find_nearest_waypoint_by_downsampling(waypoints: np.array, point_x: float, point_y: float, downsampling_rate: int = 5) -> Tuple[int, float]:
    """寻找最近点 in waypoints.

    :param waypoints: 形如 [[x1, y1], [x2, y2], ...] 的numpy数组，表示路径点
    :param point_x: 查询点的x坐标
    :param point_y: 查询点的y坐标
    :param downsampling_rate: 下采样率，默认为5 ;每5个路径点抽取一个点进行搜索，以加快计算速度

    :return:
        - 最近的 waypoint 在原始 waypoints 中的 index, 0 is start
        - 最近的 waypoint 到查询点的距离
    """
    if len(waypoints) == 0:
        raise ValueError("Waypoints list cannot be empty.")

    # 进行路径点的下采样
    if len(waypoints) < downsampling_rate:
        downsampled_indices = np.arange(len(waypoints))
    else:
        downsampled_indices = np.arange(0, len(waypoints), downsampling_rate)

    waypoints_downsampling = waypoints[downsampled_indices]

    # 计算每个下采样点到目标点的欧几里得距离
    distances = np.sqrt((waypoints_downsampling[:, 0] - point_x) ** 2 + (waypoints_downsampling[:, 1] - point_y) ** 2)

    # 找到最小距离的索引
    id_nearest_downsampled = np.argmin(distances)
    nearest_distance = distances[id_nearest_downsampled]

    # 获取该最近点在原始 waypoints 中的 index
    id_nearest_original = downsampled_indices[id_nearest_downsampled]

    return id_nearest_original, nearest_distance


def calc_yaw_diff_two_waypoints(angle1: float, angle2: float):
    """计算两个路径点之间的夹角,结果在[-pi,pi]范围内,"""
    return (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi
