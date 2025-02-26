from copy import deepcopy
from functools import lru_cache
import logging
import sys
import os
from typing import Dict, List, Tuple, Optional, Union, Any

# Third-party library
from matplotlib.patches import FancyArrow
from shapely.geometry import Point, Polygon
import numpy as np
import matplotlib.pyplot as plt

# Local library
from devkit.common.actor_state.state_representation import StateSE2
from devkit.common.coordinate_system.frenet import State
from devkit.common.geometry.cubic_spline import CubicSpline2D

from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTasks
from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapJsonLoader
from devkit.sim_engine.planning.planner.route_planner.utils.breadth_first_search import BreadthFirstSearch
from devkit.sim_engine.planning.planner.route_planner.utils.breadth_first_search import create_route_path_breadth_first_search
from devkit.sim_engine.planning.planner.route_planner.utils.minesim_graph import MineSimGraph
from devkit.sim_engine.planning.planner.route_planner.utils.graph_search_utils import calculate_start_refpath_info
from devkit.sim_engine.planning.planner.route_planner.utils.graph_search_utils import get_most_matched_base_path
from devkit.sim_engine.planning.planner.route_planner.utils.graph_search_utils import get_most_matched_connector_path
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import MineSimMapLayer
from devkit.sim_engine.map_manager.minesim_map_data.maps_datatypes import ReferencePathType

logger = logging.getLogger(__name__)


class GlobalRoutePathPlanner:
    """RoutePlanner 实现当前场景的路径寻优."""

    def __init__(
        self,
        map_name: str,
        semantic_map: MineSimSemanticMapJsonLoader,
        minesim_graph: MineSimGraph,
        is_initial_path_stitched: bool = True,  # EGO 的起始路径是否需要拼接
    ):
        self.map_name = map_name
        self.semantic_map = semantic_map
        self.minesim_graph = minesim_graph
        self.mine_search_bfs: BreadthFirstSearch = None
        self.is_initial_path_stitched = is_initial_path_stitched

        # =========== ego_route_path INFO ===========
        self.ego_route_path_tokens: List[str] = None
        self.ego_route_search_success: bool = False
        self.ego_route_path_waypoints: List[List[float]] = None
        self.ego_initial_stitched_route_waypoints: List[List[float]] = None  #
        self.refline: np.array = None  # ! waypoint[x,y,heading, height,slope];unit[m, m, rad, m, degree];heading [0,2pi]; [-pi,pi]  is best
        self.cubic_spline: CubicSpline2D = None  # 三次样条 组合的参考路径多段线.
        self.refline_smooth: np.array = None  # ![x,y,ref_yaw, ref_rk] ;heading=ref_yaw

    def get_ego_global_route_path(
        self,
        start_pose_statese2: StateSE2,
        goal_task: PlanningProblemGoalTasks,
        scenario_name: str,
        vehicle_length: float = 9.0,
        vehicle_width: float = 4.0,
    ) -> bool:
        """ego车辆的 全局路径寻优,route.
        获取HD Map中ego车到达目标区域的参考路径: Breadth First Search + 直接拼接方法

        - 注1:该参考路径仅表示了对于道路通行规则的指示
            数据处理来源:1)在地图中手动标记dubinspose;2)生成dubins curve;3)离散拼接.
        - 注2:显然该参考路径存在以下缺陷--
            1)在实际道路中使用该参考路径跟车行驶时不符合曲率约束;2)onsite_mine仿真中存在与边界发生碰撞的风险.
        - 注3:onsite_mine自动驾驶算法设计鼓励参与者设计符合道路场景及被控车辆特性的实时轨迹规划算法.

        :param vehicle_length: 车辆长度 ,,默认使用 devkit/common/actor_state/vehicle_parameters.py XG90车型参数： "length": 9.0,"width": 4.0,
        :param vehicle_width: 车辆宽度. 用于拼接路径
        输出:ego车到达目标区域的参考路径.
        """
        logger.info(f"#log# Search ego route path...")
        start_refpath_token, start_refpath_waypoint_id = calculate_start_refpath_info(
            start_pose_statese2=start_pose_statese2, semantic_map=self.semantic_map
        )

        if not self.mine_search_bfs:
            self.mine_search_bfs = create_route_path_breadth_first_search(map_name=self.map_name, minesim_graph=self.minesim_graph)

        path_tokens, is_search_successful = self.mine_search_bfs.search_route_path(
            goal_task=goal_task,
            start_refpath_token=start_refpath_token,
            start_refpath_waypoint_id=start_refpath_waypoint_id,
            target_depth=5,
        )
        if is_search_successful:
            self.ego_route_search_success: bool = True
            self.ego_route_path_tokens: List[str] = path_tokens

            self.ego_route_path_waypoints = self._get_discrete_waypoints_connected_route_path(
                path_tokens=self.ego_route_path_tokens, start_refpath_waypoint_id=start_refpath_waypoint_id, goal_task=goal_task
            )

            # 只用于IDM等依赖参考路径的简单规划器
            if self.is_initial_path_stitched:
                # 增加航向角考虑: 生成路径时不仅考虑 x, y，还可增加航向角度，使平滑转弯更加合理。
                # 动态调整过渡点数量: 根据偏移距离动态调整路径采样点，提高适应性。
                self.ego_initial_stitched_route_waypoints = GlobalRoutePathPlanner.stitch_initial_route_waypoints(
                    vehicle_token="ego",
                    start_pose_statese2=start_pose_statese2,
                    route_path_waypoints=self.ego_route_path_waypoints,
                    vehicle_length=vehicle_length,
                    vehicle_width=vehicle_width,
                    sampling_interval=6,
                    ref_path_waypoints_interval=self.semantic_map.reference_path[2]["waypoint_sampling_interval_meter"],
                )
                self.refline = np.array(self.ego_initial_stitched_route_waypoints)
                self.cubic_spline, self.refline_smooth = self.generate_frenet_frame(
                    centerline_pts=self.refline, sampling_interval=20, resampling_distance=0.2
                )

            else:
                # 其它规划器，需要做轨迹规划；
                self.refline = np.array(self.ego_route_path_waypoints)
                self.cubic_spline, self.refline_smooth = self.generate_frenet_frame(
                    centerline_pts=self.refline, sampling_interval=20, resampling_distance=0.2
                )
            logger.info(f"#log# Successfully searched ego route path: ego_route_path_tokens={self.ego_route_path_tokens}.")
        else:
            # TODO
            logger.warning("#log# Failed to search ego route path. Using other route instead!")

        # ! DEBUG plot
        if True:
            self._debug_plot_discrete_path_waypoints(start_pose_statese2=start_pose_statese2, scenario_name=scenario_name)
        return self.ego_route_search_success

    def get_agent_feasible_global_route_path(self, start_pose_statese2: StateSE2) -> bool:
        """
        - 其它车辆的 全局路径寻优,route
        - 用于其它车辆 进行 reactive-agent 交互式仿真;
        - 在 agent 最终 pose 基础上，随机匹配一条可行的参考路径；
        """
        # step 1： 定位polygon
        polygon_token = self.semantic_map.get_polygon_token_using_node(x=start_pose_statese2.x, y=start_pose_statese2.y)
        polygon_id = self.semantic_map.getind(layer_name=MineSimMapLayer.POLYGON.fullname, token=polygon_token)

        # step 2： 匹配开始路径 及 token
        if MineSimMapLayer.INTERSECTION.fullname == self.semantic_map.polygon[polygon_id]["type"]:
            # IF agent 在路口内部结束，距离那个 ROAD polygon 近，去那个polygon；
            candidate_connector_path_tokens = self.semantic_map.polygon[polygon_id]["link_referencepath_tokens"]
            start_refpath_token, start_refpath_waypoint_id = get_most_matched_connector_path(
                pose_statese2=start_pose_statese2, semantic_map=self.semantic_map, candidate_connector_path_tokens=candidate_connector_path_tokens
            )
        elif (
            MineSimMapLayer.LOADING_AREA.fullname == self.semantic_map.polygon[polygon_id]["type"]
            or MineSimMapLayer.UNLOADING_AREA.fullname == self.semantic_map.polygon[polygon_id]["type"]
            or MineSimMapLayer.ROAD_SEGMENT.fullname == self.semantic_map.polygon[polygon_id]["type"]
        ):
            candidate_referencepath_tokens = self.semantic_map.polygon[polygon_id]["link_referencepath_tokens"]
            # 去掉 conector path
            candidate_basepath_tokens = []
            for path_token in candidate_referencepath_tokens:
                path_id = self.semantic_map.getind(layer_name=MineSimMapLayer.REFERENCE_PATH.fullname, token=path_token)
                if ReferencePathType.BASE_PATH.fullname == self.semantic_map.reference_path[path_id]["type"]:
                    candidate_basepath_tokens.append(path_token)
            if candidate_basepath_tokens:  # non []
                start_refpath_token, start_refpath_waypoint_id = get_most_matched_base_path(
                    pose_statese2=start_pose_statese2, semantic_map=self.semantic_map, candidate_basepath_tokens=candidate_basepath_tokens
                )
            else:
                start_refpath_token, start_refpath_waypoint_id = None, None
        pass

        # step 3： 搜索可行的参考路径直到截止条件 search feasible long route path
        if start_refpath_token:
            return self.search_feasible_long_route_path_for_agent(
                start_refpath_token=start_refpath_token, start_refpath_waypoint_id=start_refpath_waypoint_id
            )
        else:
            return []

    def search_feasible_long_route_path_for_agent(self, start_refpath_token: str, start_refpath_waypoint_id: int) -> List[List[float]]:
        """搜索得到 agent 可行的参考路径点序列"""
        if not self.mine_search_bfs:
            self.mine_search_bfs = create_route_path_breadth_first_search(map_name=self.map_name, minesim_graph=self.minesim_graph)

        is_search_successful, path_tokens, end_path_waypoint_id = self.mine_search_bfs.search_feasible_route_path_for_other_agent(
            start_refpath_token=start_refpath_token, start_refpath_waypoint_id=start_refpath_waypoint_id, target_depth=3, max_waypoints_number=1000
        )

        # 拼接 route_waypoints
        all_waypoints = []
        for index, path_token in enumerate(path_tokens):
            refpath_waypoints = self.minesim_graph.meta_edge_objects[path_token].refpath_waypoints
            if index == 0:
                # **第一段路径**：依据 `start_refpath_waypoint_id` 截断路径，仅保留起始点后的部分
                refpath_waypoints = refpath_waypoints[start_refpath_waypoint_id:]
            elif index == len(path_tokens) - 1:
                # **最后一段路径**：依据 `end_path_waypoint_id` 截断路径，仅保留终点之前的部分
                refpath_waypoints = refpath_waypoints[: end_path_waypoint_id + 1]
            else:
                # **中间路径段**：保留整个路径段
                pass

            # 拼接路径段
            all_waypoints.extend(refpath_waypoints)
        return all_waypoints

    @staticmethod
    def stitch_initial_route_waypoints(
        vehicle_token: str,
        start_pose_statese2: StateSE2,
        route_path_waypoints: List[List[float]],
        vehicle_length: float = 9.0,
        vehicle_width: float = 4.0,
        ref_path_waypoints_interval: float = 0.2,
        sampling_interval: float = 5,
    ) -> np.array:
        """
        生成与参考路径拼接的平滑新路径，考虑车辆宽度、航向角，并动态调整过渡点数量。
        #! 注意： 目前只适合 IDM planner 在simulation中提取可以跟踪行驶的路径

        - 车长，车宽作为输入参数；
        - route_path_waypoints: 输入的 waypoints 离散间隔距离为0.2m;
        - 动态调整过渡点数量，提高适应性；
            - vehicle_token == "ego"
                - 偏离距离小于 0.6*车宽时，返回原始路径；
                - 偏离距离在 0.6*车宽到 3.0*车宽之间，向前拼接一段距离，自动计算拼接距离；
                - 偏离距离大于 3.0*车宽，返回错误提示；
            - vehicle_token != "ego"
                - 始终拼接
        - 生成路径时考虑航向角，实现平滑转弯；
        - 返回 waypoints[x,y,heading] 或 waypoints[x,y,heading, curvature]

        :param start_pose_statese2: ego车当前位姿信息 (x, y, heading)
        :param route_path_waypoints: 参考路径点集合 [[x, y, heading], ...] 或 [[x, y, heading, ...], ...]
        :param vehicle_length: 车辆长度 ,默认使用 devkit/common/actor_state/vehicle_parameters.py XG90车型参数： "length": 9.0,"width": 4.0,
        :param vehicle_width: 车辆宽度
        :return: 修正后的路径点，格式为[[x, y, heading, curvature], ...]
        """
        if not route_path_waypoints or len(route_path_waypoints) < 2:
            logging.error("Error: Invalid waypoints input.")
            return route_path_waypoints

        # 提取ego路径的前3列，避免不必要的数据处理
        ref_path = np.array(route_path_waypoints)[:, :3]
        # Step 1: 计算自车位置到参考路径的最短横向距离
        ego_position = np.array([start_pose_statese2.x, start_pose_statese2.y])
        distances = np.linalg.norm(ref_path[:, :2] - ego_position, axis=1)
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]

        # Step 2: 处理偏离情况
        if vehicle_token == "ego":
            if min_distance < 0.6 * vehicle_width:
                logging.info(f"#log# vehicle_token={vehicle_token},偏离较小，无需修正路径")
                return np.array(route_path_waypoints)  # 偏离较小，直接返回原始路径
            elif min_distance > 3.0 * vehicle_width:
                logging.warning(f"#log# vehicle_token={vehicle_token},偏离过大(3.0 * vehicle_width)，无法进行路径拼接，检查初始化位置!")
        else:
            if min_distance < 0.1 * vehicle_width:
                logging.info(f"#log# vehicle_token={vehicle_token},偏离较小，无需修正路径")
                return np.array(route_path_waypoints)  # 偏离较小，直接返回原始路径

        # Step 3: 计算最大拼接距离，考虑前方路径剩余长度 # 最大拼接距离不超过25.0米 或者前方剩下的路径长度
        max_stitch_distance = min(25.0, (len(ref_path) - min_distance_index) * ref_path_waypoints_interval)
        stitch_distance = min(min_distance * 3.0, vehicle_length * 2.0, max_stitch_distance)  # !尽可能长一点，平滑

        # Step 4: 选取前方参考路径拼接区间；
        num_transition_points = max(int(stitch_distance / ref_path_waypoints_interval), 5)  # 过渡点数基于0.2米间隔; # 计算需要的拼接点数量，最少5个点
        stitch_index_ahead = min(min_distance_index + num_transition_points, len(ref_path) - 1)  # 确保拼接索引不超出路径范围

        # Step 5: 生成拼接过渡区间的点:
        # 插值出 num_transition_points多个点替换掉前方 参考路径的系列点；
        waypoints_x = np.linspace(start_pose_statese2.x, ref_path[stitch_index_ahead][0], num_transition_points)
        waypoints_y = np.linspace(start_pose_statese2.y, ref_path[stitch_index_ahead][1], num_transition_points)
        centerline_pts = np.column_stack((waypoints_x, waypoints_y))
        # 使用三次样条插值生成平滑曲线;        # 保留第一个和最后一个点，对中间的点进行降采样
        pts_down_sampling = np.vstack([centerline_pts[0], centerline_pts[1:-1:sampling_interval], centerline_pts[-1]])
        cubic_spline = CubicSpline2D(pts_down_sampling[:, 0], pts_down_sampling[:, 1])

        # Step 6: 重新采样以获取平滑的轨迹
        # 按照resampling_distance m间隔对参考路径重新进行离散采样
        s_values = np.arange(0, cubic_spline.s_list[-1], ref_path_waypoints_interval)
        stitched_waypoints_xy = np.array([cubic_spline.calc_position(s) for s in s_values])
        stitched_waypoints_yaw = np.array([cubic_spline.calc_yaw(s) for s in s_values])
        # 范围[-PI,PI),负值表示顺时针方向的角度,正值表示逆时针方向的角度.
        # 避免索引越界，移除末尾多余数据# 去掉最后 2 个元素
        stitched_waypoints_xy = stitched_waypoints_xy[:-2]
        stitched_waypoints_yaw = stitched_waypoints_yaw[:-2]
        stitched_waypoints = np.column_stack((stitched_waypoints_xy, stitched_waypoints_yaw))

        #  Step 7: 拼接路径，确保避免索引错误
        # 仅提取 route_path_waypoints 的前三列 (x, y, heading)，跳过无用的其他列
        remaining_waypoints = ref_path[stitch_index_ahead + 1 :, :3] if stitch_index_ahead + 1 < len(ref_path) else []
        new_waypoints = np.vstack((stitched_waypoints, remaining_waypoints))

        logging.info(f"缝合了初始 route path: vehicle_token={vehicle_token},偏离距离: {min_distance:.2f} m, 拼接长度: {stitch_distance:.2f} m")
        return new_waypoints

    def _get_discrete_waypoints_connected_route_path(
        self, path_tokens: List[str], start_refpath_waypoint_id: int, goal_task: PlanningProblemGoalTasks
    ) -> List[List[float]]:
        """
        拼接多条参考路径；
        - 对于中间的路径段 path，去掉首尾各一个点；
        - 第一条和最后一条 path 太长的问题，需要截断：
            - 第一条path太长,根据start_refpath_waypoint_id往前取几个路径点截断；
            - 最后一条path太长,根据goal_task.goal_range_polygon最后框内的点往后取几个路径点来截断；

        Returns:
            List[List[float]]: waypoint[utm_x.utm_y, heading, elevation, slope], unit[m, m, rad, m,]
        """
        full_path_waypoints = []

        for i, path_token in enumerate(path_tokens):
            waypoints = deepcopy(self.minesim_graph.meta_edge_objects[path_token].refpath_meta_data["waypoints"])

            if i == 0:  # 第一条路径: 根据 start_refpath_waypoint_id 截断路径，保留起点后的部分
                start_cut_waypoint_id = max(start_refpath_waypoint_id - 12, 0)
                waypoints = waypoints[start_cut_waypoint_id:]

            if i == len(path_tokens) - 1:  # 最后一条路径: 通过 goal_task.goal_range_polygon 截断路径，保留范围内的点
                goal_polygon = goal_task.goal_range_polygon
                valid_indices = [idx for idx, point in enumerate(waypoints) if goal_polygon.contains(Point(point[0], point[1]))]
                if valid_indices:
                    last_valid_index = min(max(valid_indices) + 10, len(waypoints) - 1)
                    waypoints = waypoints[: last_valid_index + 1]

            # 中间路径处理，去掉首尾点
            if i == 0 or i == len(path_tokens) - 1:
                full_path_waypoints += waypoints
            else:
                # 对于中间路径，去掉首尾各一个点，确保路径连续性
                if len(waypoints) > 3:
                    full_path_waypoints += waypoints[1:-1]
                else:
                    full_path_waypoints += waypoints  # 如果路径段太短，则保留所有点

        return full_path_waypoints

    def generate_frenet_frame(
        self, centerline_pts: np.ndarray, sampling_interval: int = 3, resampling_distance: float = 0.1
    ) -> Tuple[CubicSpline2D, np.array]:
        """
        依据参考路径中心线(稀疏的)点序列进行参数化曲线建模（三次样条曲线）,然后重新离散化采样参考路径中心线各值.
        - 用于 FOP 规划器框架 转化参考线为 frenet 坐标系

        Args:
            centerline_pts (np.ndarray): 全局规划的参考路径 中心线, 稀疏的点序列[x, y, yaw , width]*i.
            sampling_interval (int): 采样间隔，默认每3个点采样一次，保留首尾。
            resampling_distance (float): 重新采样距离，默认0.1m。

        Returns:
            cubic_spline (CubicSpline2D): cubic spline 三次样条曲线 组合的参考路径多段线.
            (ref_xy, ref_yaw, ref_rk(np.ndarray): 全局规划的参考路径 中心线,0.1m 等间隔采样[x, y, yaw , curve]*N. ref_rk左正右负;
        """
        # 保留第一个和最后一个点，对中间的点进行降采样
        pts_down_sampling = np.vstack([centerline_pts[0], centerline_pts[1:-1:sampling_interval], centerline_pts[-1]])
        cubic_spline = CubicSpline2D(pts_down_sampling[:, 0], pts_down_sampling[:, 1])
        # 按照resampling_distance m间隔对参考路径重新进行离散采样
        s = np.arange(0, cubic_spline.s_list[-1], resampling_distance)
        ref_xy = [cubic_spline.calc_position(i_s) for i_s in s]
        # 返回从 x 轴到点 (x, y) 的角度,角度范围[-PI,PI),负值表示顺时针方向的角度,正值表示逆时针方向的角度.
        ref_yaw = [cubic_spline.calc_yaw(i_s) for i_s in s]
        ref_rk = [cubic_spline.calc_curvature(i_s) for i_s in s]
        refline_smooth = np.column_stack((ref_xy, ref_yaw, ref_rk))  # todo 海拔高度\坡度应该作插值

        return cubic_spline, refline_smooth

    def calc_curvature_speed_limit(self):
        """计算参考路径曲率限速
        #todo 计算参考路径曲率限速
        """
        pass

    def _debug_plot_discrete_path_waypoints(self, start_pose_statese2: StateSE2, scenario_name: str):
        """
        用于类内部，临时 debug 查看、绘图、保存各种 route path 的绘图对比结果，保存在当前文件目录下的 cache_figures 文件夹内部。

        - 绘制路径:
            - 原始的 route_path_waypoints（蓝色）
            - stitched_route_waypoints（绿色）【可选】
            - refline（红色）【可选】
            - refline_smooth（橙色）【可选】

        - 保存路径: 当前文件目录的 cache_figures 文件夹下，文件命名包含时间戳以防冲突。

        :param start_pose_statese2: ego车的初始位姿，包含 x, y, heading 信息。
        :return: None
        """
        # 确保缓存文件夹存在
        output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "cache_figures")
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))
        plt.title("Route Path Debug Visualization")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")

        # 绘制原始参考路径
        if self.ego_route_path_waypoints is not None:
            waypoints = np.array(self.ego_route_path_waypoints)
            # plt.plot(waypoints[:, 0], waypoints[:, 1], "b-", label="Original Route Path")
            plt.scatter(waypoints[:, 0], waypoints[:, 1], c="b", s=10, label="Route Points")

        # 绘制拼接后的路径
        if self.ego_initial_stitched_route_waypoints is not None:
            stitched_waypoints = np.array(self.ego_initial_stitched_route_waypoints)
            plt.plot(stitched_waypoints[:, 0], stitched_waypoints[:, 1], "g--", label="Stitched Route Path")
            plt.scatter(stitched_waypoints[:, 0], stitched_waypoints[:, 1], c="g", s=10, label="Stitched Points")

        # 绘制参考线
        # if self.refline is not None:
        #     plt.plot(self.refline[:, 0], self.refline[:, 1], "r-", label="Reference Line")

        # 绘制平滑后的参考线
        if self.refline_smooth is not None:
            plt.plot(self.refline_smooth[:, 0], self.refline_smooth[:, 1], "orange", label="CubicSpline2D Smoothed Reference Line")

        # 绘制 ego 初始位姿
        ego_x, ego_y, ego_heading = start_pose_statese2.x, start_pose_statese2.y, start_pose_statese2.heading
        plt.scatter(ego_x, ego_y, c="purple", s=30, marker="o", label="Ego Initial Position")
        plt.annotate("Ego Start", (ego_x, ego_y), textcoords="offset points", xytext=(10, 10), ha="center", fontsize=10, color="purple")

        # 添加 ego 初始位姿的航向箭头
        arrow_length = 5.0  # 航向箭头长度
        ego_arrow = FancyArrow(
            ego_x, ego_y, arrow_length * np.cos(ego_heading), arrow_length * np.sin(ego_heading), color="purple", width=0.5, label="Ego Heading"
        )
        plt.gca().add_patch(ego_arrow)

        plt.legend()
        plt.grid()

        # 生成文件名，包含时间戳以防冲突
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        figure_filename = os.path.join(output_dir, f"{scenario_name}_route_path_debug_{timestamp}.svg")

        plt.savefig(figure_filename, dpi=600)
        logger.info(f"#log# Debug plot saved to: {figure_filename}")


@lru_cache(maxsize=2)
def create_route_path_planner(
    map_name: str,
    semantic_map: MineSimSemanticMapJsonLoader,
    minesim_graph: MineSimGraph,
    is_initial_path_stitched: bool,  # EGO 的起始路径是否需要拼接, = True
) -> MineSimGraph:
    """
    1个矿区的搜索图 创建一次即可： 可以提供给 ego agents 用于路径搜索

    :return: A GlobalRoutePathPlanner instance.
    """
    route_path_planner = GlobalRoutePathPlanner(
        map_name=map_name,
        semantic_map=semantic_map,
        minesim_graph=minesim_graph,
        is_initial_path_stitched=is_initial_path_stitched,
    )

    return route_path_planner
