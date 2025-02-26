from queue import PriorityQueue
import sys
import os
from typing import Dict, List, Tuple, Optional, Union, Any

# Third-party library
from shapely.geometry import Point, Polygon
import numpy as np

# Local library
from devkit.common.coordinate_system.frenet import State
from devkit.common.geometry.cubic_spline import CubicSpline2D
from devkit.sim_engine.planning.planner.route_planner.onsite_mine.path_search_tree import PathSearchTree


class GlobalRoutePlanner:
    """调用 RoutePlanner 的多个接口实现当前场景的路径寻优."""

    def __init__(self, observation):
        # global planning
        self.route_candidates = list()  # 路网中多条候选路径
        self.refline = None  # ![x,y,ref_yaw, height,slope]
        self.frenet_traj = None
        self.cubic_spline = None  # 三次样条 组合的参考路径多段线.
        self.refline_smooth = None  # ![x,y,ref_yaw, ref_rk]

        # planning problem
        self.ego_start_state = State()
        self.ego_goal_range = None
        self.ego_goal_pose = None
        self.pTree = None

    def plan_global_route(self, observation):
        """全局路径寻优,route.

        Args:
            observation (_type_): 环境信息

        Returns:
            _type_:
        """
        self.ego_start_state = GlobalRoutePlanner.get_ego_start_state(observation)
        self.ego_goal_range = observation["test_setting"]["goal"]
        # self.ego_goal_pose = observation["test_setting"]["goal_pose"]

        method = "CubicSplineSmooth"
        # method = "ConcatRefPath"
        self.get_path_search_tree_by_BFS(observation)
        if method == "ConcatRefPath":
            self.refline = self.get_ego_reference_path_to_goal()
        elif method == "CubicSplineSmooth":
            self.cubic_spline, self.refline_smooth = self.get_smooth_ego_reference_path_to_goal()
            # self.refline_smooth = self.calc_curvature_speed_limit()
        else:
            pass  # error

    def get_path_search_tree_by_BFS(self, observation):
        ################# 开始位置所在几何,匹配路径 #################
        start_polygon_token = observation["hdmaps_info"]["tgsc_map"].get_polygon_token_using_node(self.ego_start_state.x, self.ego_start_state.y)
        # start_polygon_id = int(start_polygon_token.split("-")[1])
        start_path_token, start_path_id = self.get_best_matching_path_from_polygon(
            (self.ego_start_state.x, self.ego_start_state.y, self.ego_start_state.yaw), start_polygon_token, observation
        )

        ################# 目标pose所在几何,匹配路径  #################
        goal_center_x = sum(self.ego_goal_range["x"]) / len(self.ego_goal_range["x"])
        goal_center_y = sum(self.ego_goal_range["y"]) / len(self.ego_goal_range["y"])
        goal_polygon_token = observation["hdmaps_info"]["tgsc_map"].get_polygon_token_using_node(goal_center_x, goal_center_y)
        # start_polygon_id = int(start_polygon_token.split("-")[1])
        goal_path_token, goal_path_id = None, None

        ################# 广度优先搜索,搜索所有路径组成的路线 #################
        self.pTree = PathSearchTree(50)
        # root node
        self.pTree.add_node_to_search_tree(parent_node=None, child_node=observation["hdmaps_info"]["tgsc_map"].reference_path[start_path_id], layer=0)
        layer_num = 0
        while not self.pTree.reached_goal and layer_num <= 5:  # 假设最大搜索层级为5
            current_layer_nodes = self.pTree.get_nodes_of_layer(layer=layer_num)
            if not current_layer_nodes:  # 如果当前层没有节点，结束搜索
                print("##log## 如果当前层没有节点，结束搜索")
                break

            for node in current_layer_nodes:
                outgoing_tokens = node["outgoing_tokens"]
                for path_token in outgoing_tokens:
                    path_id = int(path_token.split("-")[1])
                    child_node = observation["hdmaps_info"]["tgsc_map"].reference_path[path_id]
                    self.pTree.add_node_to_search_tree(parent_node=node, child_node=child_node, layer=layer_num + 1)

                    check_goal_method = "usingGoalRange"
                    # check_goal_method = "usingGoalPose"
                    if check_goal_method == "usingGoalRange":  # 方法1：检查每个新增的节点路径是否达到目标
                        if self.has_waypoint_inside_goal_area(child_node["waypoints"], self.ego_goal_range["x"], self.ego_goal_range["y"]):
                            self.pTree.reached_goal = True
                            self.pTree.goal_node_path = child_node
                            break  # 找到目标后立即退出循环
                    elif check_goal_method == "usingGoalPose":  # 方法2：检查每个新增的节点路径 是否为 goal_path_token
                        if child_node["token"] == goal_path_token:
                            self.pTree.reached_goal = True
                            self.pTree.goal_node_path = child_node
                            break  # 找到目标后立即退出循环

                if self.pTree.reached_goal:
                    break
            layer_num += 1  # 移至下一层继续搜索

        if self.pTree.reached_goal:  # 完成路径搜索
            print("##log## 路径搜索成功,到达目标.")
        else:
            print("##log## 路径搜索失败,无法到达目标.")

        return self.pTree.reached_goal

    def get_ego_reference_path_to_goal(self):
        """获取HD Map中ego车到达目标区域的参考路径.
        Breadth First Search + 直接拼接方法

        注1:该参考路径仅表示了对于道路通行规则的指示
            数据处理来源:1)在地图中手动标记dubinspose;2)生成dubins curve;3)离散拼接.
        注2:显然该参考路径存在以下缺陷--
            1)在实际道路中使用该参考路径跟车行驶时不符合曲率约束;2)onsite_mine仿真中存在与边界发生碰撞的风险.
        注3:onsite_mine自动驾驶算法设计鼓励参与者设计符合道路场景及被控车辆特性的实时轨迹规划算法.

        输入:observation——环境信息;
        输出:ego车到达目标区域的参考路径.
        """
        if self.pTree.reached_goal:  # 完成搜索后
            # 回溯搜索树结果，找到从目标到起点的多段路径
            path_segments = self.pTree.backtrack_from_leaf_node()
            # 返回拼接后的路径
            self.refline = self.pTree.concatenate_paths(path_segments)
            return self.refline  # 这是从起始点到目标点的完整路径
        else:
            print("##log## 无法到达目标.")
            return None

    def get_smooth_ego_reference_path_to_goal(self):
        if self.pTree.reached_goal:  # 完成搜索后
            # 回溯搜索树结果，找到从目标到起点的多段路径
            path_segments = self.pTree.backtrack_from_leaf_node()
            # 返回拼接后的路径
            full_path_waypoints = self.pTree.concatenate_paths_delete_middle_segment_points(path_segments)
            self.refline = np.array(full_path_waypoints)
            self.cubic_spline, self.refline_smooth = self.generate_frenet_frame(centerline_pts=self.refline)

            return self.cubic_spline, self.refline_smooth  # 这是从起始点到目标点的完整路径
        else:
            print("##log## 无法到达目标.")
            return None

    def generate_frenet_frame(self, centerline_pts: np.ndarray, sampling_interval: int = 3, resampling_distance: float = 0.1):
        """依据参考路径中心线(稀疏的)点序列进行参数化曲线建模（三次样条曲线）,然后重新离散化采样参考路径中心线各值.

        Args:
            centerline_pts (np.ndarray): 全局规划的参考路径 中心线, 稀疏的点序列[x, y, yaw , width]*i.
            sampling_interval (int): 采样间隔，默认每3个点采样一次，保留首尾。
            resampling_distance (float): 重新采样距离，默认0.1m。

        Returns:
            cubic_spline (CubicSpline2D): cubic spline 三次样条曲线 组合的参考路径多段线.
            (ref_xy, ref_yaw, ref_rk(np.ndarray): 全局规划的参考路径 中心线,0.1m 等间隔采样[x, y, yaw , curve]*N.
        """
        # 保留第一个和最后一个点，对中间的点进行降采样
        pts_down_sampling = np.vstack([centerline_pts[0], centerline_pts[1:-1:sampling_interval], centerline_pts[-1]])

        self.cubic_spline = CubicSpline2D(pts_down_sampling[:, 0], pts_down_sampling[:, 1])
        # 按照resampling_distance m间隔对参考路径重新进行离散采样
        s = np.arange(0, self.cubic_spline.s_list[-1], resampling_distance)
        ref_xy = [self.cubic_spline.calc_position(i_s) for i_s in s]
        # 返回从 x 轴到点 (x, y) 的角度,角度范围[-PI,PI),负值表示顺时针方向的角度,正值表示逆时针方向的角度.
        ref_yaw = [self.cubic_spline.calc_yaw(i_s) for i_s in s]
        ref_rk = [self.cubic_spline.calc_curvature(i_s) for i_s in s]
        self.refline_smooth = np.column_stack((ref_xy, ref_yaw, ref_rk))  # todo 海拔高度\坡度应该作插值
        return self.cubic_spline, self.refline_smooth

    def calc_curvature_speed_limit(self):
        """计算参考路径曲率限速
        #todo 计算参考路径曲率限速
        """
        pass

    def get_best_matching_path_from_polygon(self, veh_pose: Tuple[float, float, float], polygon_token: str, observation) -> str:
        """根据veh_pose(x,y,yaw)车辆定位位姿,从polygon_token所属的 link_referencepath_tokens 中匹配最佳参考路径.

        方法:
        1) 匹配2条path最近点;
        2) 获取最佳path;

        Args:
            veh_pose (Tuple[float,float,float]):车辆的位姿.
            polygon_token (str):指定的polygon token.
            observation:包含地图信息

        Returns:
            str:最佳匹配的 path_token,id_path.
        """
        semantic_map = observation["hdmaps_info"]["tgsc_map"]
        if not polygon_token.startswith("polygon-"):
            raise ValueError(f"Invalid polygon_token:{polygon_token}")

        id_polygon = int(polygon_token.split("-")[1])
        if id_polygon > len(semantic_map.polygon):
            raise IndexError(f"Polygon ID {id_polygon} out of bounds.请检查.")
        if semantic_map.polygon[id_polygon]["type"] == "intersection":
            raise IndexError(f"##log## Polygon ID = {id_polygon},目前未处理自车初始位置在交叉口区域的寻路逻辑.")

        link_referencepath_tokens = semantic_map.polygon[id_polygon]["link_referencepath_tokens"]

        candidate_paths = PriorityQueue()
        for _, path_token in enumerate(link_referencepath_tokens):
            id_path = int(path_token.split("-")[1])
            if semantic_map.reference_path[id_path]["type"] == "base_path":
                # 匹配最近点
                waypoint = GlobalRoutePlanner.find_nearest_waypoint(
                    waypoints=np.array(semantic_map.reference_path[id_path]["waypoints"]), veh_pose=veh_pose, downsampling_rate=5
                )
                yaw_diff = GlobalRoutePlanner.calc_yaw_diff_two_waypoints(waypoint1=(waypoint[0], waypoint[1], waypoint[2]), waypoint2=veh_pose)
                path_info = {"path_token": path_token, "id_path": id_path, "waypoint": waypoint, "yaw_diff": abs(yaw_diff)}
                candidate_paths.put((path_info["yaw_diff"], path_info))  # yaw_diff由小到大排序

        if candidate_paths.empty():
            raise ValueError(f"##log## Polygon ID = {id_polygon},所属路径均为connector_path,有问题.")
        # 得到同向最佳path的 token,id
        best_path_info = candidate_paths.get()  # 自动返回优先级最高的元素（优先级数值最小的元素）并从队列中移除它。

        return best_path_info[1]["path_token"], best_path_info[1]["id_path"]

    @staticmethod
    def find_nearest_waypoint(waypoints: np.array, downsampling_rate: int = 5, veh_pose: Tuple[float, float, float] = None):
        waypoints_downsampling = np.array(waypoints[::downsampling_rate])  # downsampling_rate,每5个路径点抽取一个点
        distances = np.sqrt((waypoints_downsampling[:, 0] - veh_pose[0]) ** 2 + (waypoints_downsampling[:, 1] - veh_pose[1]) ** 2)
        id_nearest = np.argmin(distances)
        return waypoints_downsampling[id_nearest]

    @staticmethod
    def calc_yaw_diff_two_waypoints(waypoint1: Tuple[float, float, float], waypoint2: Tuple[float, float, float]):
        """计算两个路径点之间的夹角,结果在[-pi,pi]范围内,"""
        angle1 = waypoint1[2]
        angle2 = waypoint2[2]
        yaw_diff = (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi
        return yaw_diff

    @staticmethod
    def get_ego_start_state(observation) -> State:
        return State(
            x=observation["vehicle_info"]["ego"]["x"],
            y=observation["vehicle_info"]["ego"]["y"],
            yaw=observation["vehicle_info"]["ego"]["yaw_rad"],
            v=observation["vehicle_info"]["ego"]["v_mps"],
            a=observation["vehicle_info"]["ego"]["acc_mpss"],
            yaw_rate=observation["vehicle_info"]["ego"]["yawrate_radps"],
        )

    @staticmethod
    def has_waypoint_inside_goal_area(
        ref_path_waypoints: np.array = None,
        goal_area_x: List = None,
        goal_area_y: List = None,
    ) -> bool:
        """计算参考路径的waypoints 是否 有点waypoint在目标区域内部.

        Args:
            ref_path_waypoints (np.array, optional): 参考路径的waypoints. Defaults to None.
            goal_area_x (List, optional): 目标区域x坐标列表. Defaults to None.
            goal_area_y (List, optional): 目标区域y坐标列表. Defaults to None.

        Returns:
            bool: 参考路径的waypoints 是否 有点waypoint在目标区域内部
        """
        if ref_path_waypoints is None or goal_area_x is None or goal_area_y is None:
            return False

        # Create Polygon object representing the goal area
        goal_area_coords = list(zip(goal_area_x, goal_area_y))
        goal_area_polygon = Polygon(goal_area_coords)

        # Check each waypoint
        for waypoint in ref_path_waypoints:
            x, y = waypoint[0], waypoint[1]
            if goal_area_polygon.contains(Point(x, y)):
                return True
        return False
