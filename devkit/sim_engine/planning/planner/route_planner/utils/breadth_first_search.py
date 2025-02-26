from collections import deque
import copy
from functools import lru_cache
import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely

from devkit.sim_engine.planning.planner.route_planner.utils.minesim_graph import MineSimGraph
from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphEdgeRefPathMapObject
from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphTopologyNodeMapObject
from devkit.sim_engine.planning.planner.route_planner.utils.graph_search_utils import calculate_end_refpath_info

from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTasks
from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTask
from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTaskFinalPose

logger = logging.getLogger(__name__)


class BreadthFirstSearch:
    """
    图搜索算法: 广度优先搜索 (BFS)
    执行迭代广度优先搜索的类。
    该类对 lane-level 图形搜索进行操作。目标条件指定为是否可以在目标 BOX 处找到车道。

    A class that performs iterative breadth-first search.
    The class operates on lane-level graph search.
    The goal condition is specified to be if the lane can be found at the target BOX.
    """

    def __init__(
        self,
        minesim_graph: MineSimGraph,
    ):
        """
        Constructor for the BreadthFirstSearch class.

        :param minesim_graph: An instance of the MineSimGraph
        :param start_refpath_token: The starting reference path token
        :param start_refpath_waypoint_id: Optional starting waypoint index
        """
        self.minesim_graph = minesim_graph

        # ego route path search
        self.start_refpath_token: str = None
        self.start_refpath_waypoint_id: int = None
        self.end_refpath_token: str = None
        self.end_refpath_waypoint_id: int = None

    def search_route_path(
        self, goal_task: PlanningProblemGoalTasks, start_refpath_token: str, start_refpath_waypoint_id: int = None, target_depth: int = 5
    ) -> Tuple[List[str], bool]:
        """
        Performs breadth-first search to find a route to the target reference path.

        :param goal_task: The target goal condition.
        :param target_depth: Maximum allowed search depth.
        :return:
            - A list of reference path tokens representing the route.
            - A boolean indicating whether the goal was successfully found.
        """
        if self.start_refpath_token is None:
            self.start_refpath_token = start_refpath_token
            self.start_refpath_waypoint_id = start_refpath_waypoint_id

        # TODO 解析目标任务
        # if isinstance(goal_task, PlanningProblemGoalTaskFinalPose):
        #     self.end_refpath_token, self.end_refpath_waypoint_id = calculate_end_refpath_info(pose_statese2=goal_task.final_pose,semantic_map=semantic_map)

        # Initialize BFS queue with (current_path, path_so_far, depth)
        refpath_token_queue = deque([(self.start_refpath_token, [], 0)])
        visited = set()

        while refpath_token_queue:
            current_path_token, path_tokens, depth = refpath_token_queue.popleft()

            # Avoid revisiting nodes # 避免重重复访问节点
            if current_path_token in visited:
                continue
            visited.add(current_path_token)

            # Add current edge to the path
            path_tokens.append(current_path_token)

            #! Check goal condition, if the goal is reached
            if self._is_goal_reached(current_path_token=current_path_token, goal_task=goal_task):
                logger.info(f"#log# successfully search route path_tokens: {path_tokens}")
                return path_tokens, True

            #! Check depth limit
            if depth < target_depth:
                # Expand neighbors if depth limit is not exceeded
                outgoing_path_tokens = self.minesim_graph.get_outgoing_path_token_neighbors(path_token=current_path_token)
                for outgoing_path_token in outgoing_path_tokens:
                    refpath_token_queue.append((outgoing_path_token, path_tokens.copy(), depth + 1))

        logger.warning("#log# BFS failed to find a valid path within the target depth.")
        return path_tokens, False  # Return the longest path explored if goal not reached

    def search_feasible_route_path_for_other_agent(
        self,
        start_refpath_token: str,
        start_refpath_waypoint_id: int = 0,  # 默认值设为 0 避免 None 错误
        target_depth: int = 5,
        max_waypoints_number: int = 1000,
    ) -> Tuple[bool, List[str], int]:
        """
        搜索 agent 终点前方可行的路径，满足足够长度。
        结束条件：
        - 达到地图边界；
        - 搜索深度达标；
        - 累计 waypoints 数量超过阈值（max_waypoints_number）。

        Args:
            start_refpath_waypoint_id: 起始路径点的索引，默认为 0。
            max_waypoints_number: 最大允许的 waypoints 数量（约 200 米，每个点间隔 0.2 米）。
        Returns:
            Tuple[是否成功, 路径token列表, 终止路径点索引]
        """
        # Initialize BFS queue with (current_path token, path_so_far token list, depth，cumulative_waypoints_num)
        refpath_token_queue = deque([(start_refpath_token, [], 0, 0)])
        visited = set()
        end_path_waypoint_id = 0  # 初始化默认值

        while refpath_token_queue:
            current_path_token, path_tokens, depth, cumulative_waypoints_num = refpath_token_queue.popleft()

            # Avoid revisiting nodes # 避免重重复访问节点
            if current_path_token in visited:
                continue
            visited.add(current_path_token)
            # Add current edge to the path
            path_tokens.append(current_path_token)

            #! Check end condition
            is_search_successful, current_end_id, new_cumulative = self._check_search_end_condition_for_agent(
                current_path_token=current_path_token,
                start_refpath_token=start_refpath_token,
                start_refpath_waypoint_id=start_refpath_waypoint_id,
                cumulative_waypoints_num=cumulative_waypoints_num,
                max_waypoints_number=max_waypoints_number,
            )
            cumulative_waypoints_num = new_cumulative
            end_path_waypoint_id = current_end_id  # 更新终止索引

            if is_search_successful:
                return True, path_tokens, end_path_waypoint_id

            #! Check depth limit  
            if depth < target_depth:
                # Expand neighbors if depth limit is not exceeded 扩展邻居节点
                outgoing_path_tokens = copy.deepcopy(self.minesim_graph.get_outgoing_path_token_neighbors(path_token=current_path_token))
                if outgoing_path_tokens:
                    random.shuffle(outgoing_path_tokens)
                for outgoing_path_token in outgoing_path_tokens:
                    refpath_token_queue.append((outgoing_path_token, path_tokens.copy(), depth + 1, cumulative_waypoints_num))

        return True, path_tokens, end_path_waypoint_id  # 达到地图边界；不算失败状态 or 搜索深度达标；

    def _check_search_end_condition_for_agent(
        self,
        current_path_token: str,
        start_refpath_token: str,
        start_refpath_waypoint_id: int,
        cumulative_waypoints_num: int,
        max_waypoints_number: int,
    ) -> tuple[bool, int, int]:
        """路径终止条件的检查：agent 往前搜索得到：满足足够长度—— 累计达到 1000 个 waypoints;
        Returns:
            tuple[bool, int,int]: is_search_successful? , current path max index; new cumulative_waypoints_num
        """
        # 计算当前路径的可用 waypoints 数量
        edge = self.minesim_graph.meta_edge_objects[current_path_token]
        total_waypoints = len(edge.refpath_waypoints)
        if current_path_token == start_refpath_token and cumulative_waypoints_num == 0:
            usable_waypoints = total_waypoints - start_refpath_waypoint_id
            if start_refpath_waypoint_id >= total_waypoints:
                raise ValueError("起始路径点索引越界！")
        else:
            usable_waypoints = total_waypoints
        new_cumulative = cumulative_waypoints_num + usable_waypoints
        excess = new_cumulative - max_waypoints_number

        if excess >= 0:
            # 计算实际可用的终止索引
            end_index = total_waypoints - excess
            end_index = max(end_index, 0)  # 确保非负
            return True, end_index, new_cumulative
        else:
            return False, total_waypoints-1, new_cumulative

    def _is_goal_reached(self, current_path_token: str, goal_task: PlanningProblemGoalTask) -> bool:
        """
        Checks whether the goal task has been reached.

        :param current_token: The current graph edge token
        :param goal_task: The goal condition task
        :return: True if goal is reached, False otherwise
        """
        # 判断 current_path_token的 waypoint 是否穿过 polygon ; waypoint可以5个点 降采样;点间隔0.2m;
        # 这两种情况的路径点足够长:
        # !case 1 refpath_waypoints仅头部一些点在 goal_task.goal_range_polygon 里面; 没有穿过去.
        # !case 2 refpath_waypoints穿过 goal_task.goal_range_polygon
        return self._has_waypoints_reached_goal_area(current_path_token=current_path_token, goal_task=goal_task)

    def _has_waypoints_reached_goal_area(self, current_path_token: str, goal_task: PlanningProblemGoalTask) -> bool:
        """
        检查沿当前参考路径的任何路径点是否穿过或结束于目标多边形内部。
        Checks whether any waypoints along the current reference path pass through or end inside the goal polygon.

        :param current_path_token: The token representing the current graph edge.
        :param goal_task: The goal condition task containing the target goal polygon.
        :return: True if goal is reached (case 1 or case 2), False otherwise (case 3 or case 4).
        """
        # Retrieve waypoints for the current reference path
        refpath_waypoints = np.array(self.minesim_graph.meta_edge_objects[current_path_token].refpath_waypoints)

        # Downsample waypoints to reduce computation, taking every 5th point
        downsampled_waypoints = refpath_waypoints[::5] if len(refpath_waypoints) > 10 else refpath_waypoints

        # Convert the downsampled waypoints to shapely Point objects for intersection checks
        downsampled_waypoint_points = [shapely.geometry.Point(wp[0], wp[1]) for wp in downsampled_waypoints]

        # Check for waypoint presence within the goal polygon
        # !case 1 refpath_waypoints仅头部一些点在 goal_task.goal_range_polygon 里面; 没有穿过去.
        # !case 2 refpath_waypoints穿过 goal_task.goal_range_polygon
        # case 3 refpath_waypoints仅尾部一些点在 goal_task.goal_range_polygon 里面; 没有穿过去.
        # case 4 refpath_waypoints ALL的点都不在 goal_task.goal_range_polygon 里面
        # Case 5: All points are inside the polygon
        polygon = goal_task.goal_range_polygon
        inside_points = [point for point in downsampled_waypoint_points if polygon.contains(point)]

        if not inside_points:
            # Case 4: No waypoints are inside the polygon
            return False

        # Determine if the path crosses through the polygon
        first_inside = polygon.contains(downsampled_waypoint_points[0])
        last_inside = polygon.contains(downsampled_waypoint_points[-1])

        if first_inside and last_inside:
            # Case 5: All points are inside the polygon
            return False

        if last_inside:
            # Case 3: Only the end waypoints are inside, not crossed
            return False

        if first_inside:
            # Case 1: Only the start waypoints are inside, not crossed
            return True

        # Case 2: Path crosses the polygon from outside to inside and back to outside
        return True


@lru_cache(maxsize=2)
def create_route_path_breadth_first_search(map_name: str, minesim_graph: MineSimGraph) -> MineSimGraph:
    """1个矿区的 BreadthFirstSearch 创建一次即可;"""
    return BreadthFirstSearch(minesim_graph=minesim_graph)
