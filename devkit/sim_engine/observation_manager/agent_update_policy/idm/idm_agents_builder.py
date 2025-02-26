import logging
import os
from typing import List, Optional, Tuple

from matplotlib import pyplot as plt
from matplotlib import patches
from descartes import PolygonPatch
from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm

from devkit.common.actor_state.agent import Agent
from devkit.common.actor_state.oriented_box import OrientedBox
from devkit.common.actor_state.state_representation import StateSE2, StateVector2D
from devkit.common.actor_state.tracked_objects_types import TrackedObjectType
from devkit.sim_engine.map_manager.abstract_map import AbstractMap
from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.sim_engine.observation_manager.agent_update_policy.idm.idm_agent import IDMAgent, IDMInitialState
from devkit.sim_engine.observation_manager.agent_update_policy.idm.idm_agent_manager import UniqueIDMAgents
from devkit.sim_engine.observation_manager.agent_update_policy.idm.idm_policy import IDMPolicy
from devkit.sim_engine.observation_manager.agent_update_policy.idm.utils import create_path_from_se2

from devkit.sim_engine.path.interpolated_path import InterpolatedPath
from devkit.sim_engine.map_manager.occupancy_map.abstract_occupancy_map import OccupancyMap
from devkit.sim_engine.map_manager.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMap
from devkit.sim_engine.map_manager.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMapFactory
from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphEdgeRefPathMapObject
from devkit.sim_engine.planning.planner.route_planner.global_route_path_planner import GlobalRoutePathPlanner
from devkit.sim_engine.planning.planner.route_planner.global_route_path_planner import create_route_path_planner
from devkit.sim_engine.planning.planner.route_planner.utils.minesim_graph import create_minesim_graph
from devkit.sim_engine.observation_manager.agent_update_policy.idm.utils import consolidate_agent_trajectory_to_path

logger = logging.getLogger(__name__)


def _debug_plot_agent_discrete_path_waypoints(
    agent_track_token: str,
    true_path_state_se2: List[StateSE2],
    stitched_route_waypoints: Optional[np.ndarray],
    agent_path: InterpolatedPath,
    box_on_agent_path: OrientedBox,
    agent: Agent,
    scenario_name: str,
):
    """
    用于临时 debug 绘图，显示 agent 的路径与参考路径。
    保存路径：当前文件目录的 `cache_figures` 文件夹，文件命名包含时间戳。

    Debug function to visualize agent's path and related information.
    Saves plots in the `cache_figures` folder with timestamped filenames.

    :param agent_track_token: Unique identifier of the agent.
    :param true_path_state_se2: Original trajectory points of the agent.
    :param stitched_route_waypoints: Path points generated after stitching.
    :param agent_path: Interpolated path used by the agent.
    :param box_on_agent_path: Bounding box of the agent on the path.
    """
    # Ensure cache folder exists
    output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "cache_figures")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 10))
    plt.title(f"Debug Visualization for Agent {agent_track_token}")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")

    # 1. 绘制真实路径
    if true_path_state_se2:
        true_path_x = [state.x for state in true_path_state_se2]
        true_path_y = [state.y for state in true_path_state_se2]
        plt.plot(true_path_x, true_path_y, "b-", label="True Path (Agent Trajectory)", marker="o", markersize=3)

    # 2. 绘制拼接路径
    if stitched_route_waypoints is not None and len(stitched_route_waypoints) > 0:
        plt.plot(stitched_route_waypoints[:, 0], stitched_route_waypoints[:, 1], "g--", label="Stitched Route Path", marker=".", markersize=3)

    # 3. 绘制插值路径
    sampled_path = agent_path.get_sampled_path()
    interpolated_points = np.array([[state.x, state.y] for state in sampled_path])
    plt.plot(interpolated_points[:, 0], interpolated_points[:, 1], "r-", label="Interpolated Path")

    # 4. 绘制代理的占用框（Bounding Box） box_on_agent_path
    corners = np.array([[corner.x, corner.y] for corner in box_on_agent_path.all_corners()])

    # 闭合多边形
    plt.plot(np.append(corners[:, 0], corners[0, 0]), np.append(corners[:, 1], corners[0, 1]), "purple", label="Agent Occupancy Box", linewidth=2)
    # plt.fill(corners[:, 0], corners[:, 1], color="purple", alpha=0.3)

    # 5. 绘制代理的框和箭头
    _plot_single_vehicle(
        agent=agent,
        key=agent_track_token,
        x=agent.center.x,
        y=agent.center.y,
        yaw=agent.center.heading,
        veh_length=agent.box.length,
        veh_width=agent.box.width,
        plot_yaw_triangle=True,
        plot_box_edge=False,
        plot_vehicle_id=True,
    )

    plt.axis("equal")
    plt.legend()
    plt.grid()

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = os.path.join(output_dir, f"{scenario_name}_agent_path_debug_{agent_track_token}_{timestamp}.svg")

    plt.savefig(figure_filename, dpi=600)
    plt.close()
    logger.info(f"#log# Debug plot for agent {agent_track_token} saved to: {figure_filename}")


def _plot_single_vehicle(
    agent: Agent,
    key: str,
    x: float,
    y: float,
    yaw: float,
    veh_length: float,
    veh_width: float,
    plot_yaw_triangle: bool = False,
    plot_box_edge: bool = False,
    plot_vehicle_id: bool = False,
):
    """
    绘制车辆的占用框和航向箭头。

    :param agent: Agent 对象。
    :param key: 车辆标识符。
    :param x: 中心点 x 坐标。
    :param y: 中心点 y 坐标。
    :param yaw: 航向角（弧度）。
    :param veh_length: 车辆长度。
    :param veh_width: 车辆宽度。
    :param plot_yaw_triangle: 是否绘制航向箭头。
    :param plot_box_edge: 是否绘制框边界。
    :param plot_vehicle_id: 是否在框上标注车辆 ID。
    """
    # 1. 计算车辆的四个角点
    half_length = veh_length / 2
    half_width = veh_width / 2

    x_A3 = x - half_length * np.cos(yaw) + half_width * np.sin(yaw)
    y_A3 = y - half_length * np.sin(yaw) - half_width * np.cos(yaw)
    width_x = veh_length
    height_y = veh_width

    rect = patches.Rectangle(
        xy=(x_A3, y_A3),
        width=width_x,
        height=height_y,
        angle=yaw / np.pi * 180,
        alpha=0.5,
        facecolor="#3398da",
        fill=True,
        edgecolor="black" if plot_box_edge else None,
        linewidth=1.0 if plot_box_edge else 0.0,
        zorder=22,
    )
    plt.gca().add_patch(rect)

    # 2. 绘制航向箭头（等边三角形）
    if plot_yaw_triangle:
        triangle_length = veh_length * 0.5
        top_point = (x + triangle_length * np.cos(yaw), y + triangle_length * np.sin(yaw))
        left_point = (x - triangle_length * np.sin(yaw) / 2, y + triangle_length * np.cos(yaw) / 2)
        right_point = (x + triangle_length * np.sin(yaw) / 2, y - triangle_length * np.cos(yaw) / 2)
        triangle_points = [top_point, left_point, right_point]
        plt.gca().add_patch(patches.Polygon(triangle_points, closed=True, color="#154361", zorder=22))

    # 3. 绘制车辆标识
    if plot_vehicle_id:
        plt.annotate(key, (x + 2.0, y + 2.0), fontsize=10, zorder=25)


def _debug_plot_agent_discrete_path_waypoints_with_speed_limit(
    agent_track_token: str,
    true_path_state_se2: List[StateSE2],
    stitched_route_waypoints: Optional[np.ndarray],
    agent_path: InterpolatedPath,
    box_on_agent_path: OrientedBox,
    agent: Agent,
    scenario_name: str,
):
    """新增限速可视化功能的调试绘图"""
    # Ensure cache folder exists
    output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "cache_figures")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 10))
    plt.title(f"Debug Visualization for Agent {agent_track_token}")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")

    # # 1. 绘制真实路径
    # if true_path_state_se2:
    #     true_path_x = [state.x for state in true_path_state_se2]
    #     true_path_y = [state.y for state in true_path_state_se2]
    #     plt.plot(true_path_x, true_path_y, "b-", label="True Path (Agent Trajectory)", marker="o", markersize=3)

    # # 2. 绘制拼接路径
    # if stitched_route_waypoints is not None and len(stitched_route_waypoints) > 0:
    #     plt.plot(stitched_route_waypoints[:, 0], stitched_route_waypoints[:, 1], "g--", label="Stitched Route Path", marker=".", markersize=3)

    # 3. 绘制插值路径
    # ================ 新增限速可视化逻辑 ================
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable

    # 3.1 绘制带限速颜色映射的插值路径
    sampled_path = agent_path.get_sampled_path()
    interpolated_points = np.array([[state.x, state.y] for state in sampled_path])

    try:
        # 获取车辆参数
        vehicle_width = agent.box.width
        from devkit.common.actor_state.vehicle_parameters import get_vehicle_waypoint_speed_limit_data

        max_lateral_accel, max_speed_limit, min_speed_limit = get_vehicle_waypoint_speed_limit_data(vehicle_width)

        # 计算路径限速
        if agent_path._speed_interp is None:
            agent_path.compute_speed_limits(max_lateral_accel=max_lateral_accel, max_speed=max_speed_limit, min_speed=min_speed_limit)

        # 获取采样点的限速值
        speeds = [agent_path.get_speed_limit_at_progress(p.progress) for p in sampled_path]
        norm = plt.Normalize(min(speeds), max(speeds))

        # 创建颜色映射线段
        points = interpolated_points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="jet", norm=norm, alpha=0.8)
        lc.set_array(np.array(speeds))
        plt.gca().add_collection(lc)

        # 添加颜色条
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap="jet"), ax=plt.gca())
        cbar.set_label("Speed Limit (m/s)", rotation=270, labelpad=15)

    except (AttributeError, RuntimeError) as e:
        # 限速数据不可用时回退到普通绘制
        plt.plot(interpolated_points[:, 0], interpolated_points[:, 1], "r-", label="Interpolated Path (No Speed Data)")

    # ================ 原有其他绘图逻辑保持不变 ================
    # 4. 绘制代理的占用框（Bounding Box） box_on_agent_path
    corners = np.array([[corner.x, corner.y] for corner in box_on_agent_path.all_corners()])

    # 闭合多边形
    plt.plot(np.append(corners[:, 0], corners[0, 0]), np.append(corners[:, 1], corners[0, 1]), "purple", label="Agent Occupancy Box", linewidth=2)
    # plt.fill(corners[:, 0], corners[:, 1], color="purple", alpha=0.3)

    # 5. 绘制代理的框和箭头
    _plot_single_vehicle(
        agent=agent,
        key=agent_track_token,
        x=agent.center.x,
        y=agent.center.y,
        yaw=agent.center.heading,
        veh_length=agent.box.length,
        veh_width=agent.box.width,
        plot_yaw_triangle=True,
        plot_box_edge=False,
        plot_vehicle_id=True,
    )

    plt.axis("equal")
    plt.legend()
    plt.grid()

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = os.path.join(output_dir, f"{scenario_name}_agent_path_debug_add_speed_limit_{agent_track_token}_{timestamp}.svg")

    plt.savefig(figure_filename, dpi=600)
    plt.close()
    logger.info(f"#log# Debug plot for agent {agent_track_token} saved to: {figure_filename}")


def _debug_plot_agent_discrete_path_waypoints_with_speed_limit_and_curvatures(
    agent_track_token: str,
    true_path_state_se2: List[StateSE2],
    stitched_route_waypoints: Optional[np.ndarray],
    agent_path: InterpolatedPath,
    box_on_agent_path: OrientedBox,
    agent: Agent,
    scenario_name: str,
):
    """集成路径限速与曲率对比的调试绘图"""
    # 初始化输出目录
    output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "cache_figures")
    os.makedirs(output_dir, exist_ok=True)

    # 创建带子图的画布
    plt.figure(figsize=(16, 12))  # 增大画布尺寸
    plt.suptitle(f"Agent {agent_track_token} Path Analysis", y=0.98, fontsize=14)

    # ==================== 子图1：路径可视化 ====================
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title("Spatial Path Visualization")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")

    # 绘制基础路径（保持原有逻辑）
    # if true_path_state_se2:
    #     true_path_x = [state.x for state in true_path_state_se2]
    #     true_path_y = [state.y for state in true_path_state_se2]
    #     ax1.plot(true_path_x, true_path_y, "b-", label="Ground Truth Path", marker="o", markersize=3)

    # if stitched_route_waypoints is not None and len(stitched_route_waypoints) > 0:
    #     # ax1.plot(stitched_route_waypoints[:, 0], stitched_route_waypoints[:, 1], "g--", label="Stitched Path", marker=".", markersize=3)
    #     # 绘制拼接好的路径连线（绿色虚线）
    #     ax1.plot(stitched_route_waypoints[:, 0], stitched_route_waypoints[:, 1], color="green", linestyle="--", linewidth=1.5, label="Stitched Path")
    #     # 可选：叠加散点图突出路径点（蓝色小圆点）
    #     ax1.scatter(stitched_route_waypoints[:, 0], stitched_route_waypoints[:, 1], color="blue", marker=".", s=10, zorder=3, label="Waypoints")

    # ==================== 新增限速热力图 ====================
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable

    sampled_path = agent_path.get_sampled_path()
    interpolated_points = np.array([[state.x, state.y] for state in sampled_path])
    progress_list = [p.progress for p in sampled_path]

    try:
        # 获取车辆参数并计算限速
        vehicle_width = agent.box.width
        from devkit.common.actor_state.vehicle_parameters import get_vehicle_waypoint_speed_limit_data

        max_lateral_accel, max_speed, min_speed = get_vehicle_waypoint_speed_limit_data(vehicle_width)
        if not hasattr(agent_path, "_speed_interp") or agent_path._speed_interp is None:
            agent_path.compute_speed_limits(max_lateral_accel, max_speed, min_speed)

        # 获取曲率数据（假设已存储在agent_path）
        curvatures = agent_path.get_curvature_list(use_smoothed=True)  # 需要确保InterpolatedPath有此方法
        speeds = [agent_path.get_speed_limit_at_progress(p) for p in progress_list]

        # 创建颜色映射路径
        points = interpolated_points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(min(speeds), max(speeds))
        lc = LineCollection(segments, cmap="viridis", norm=norm, linewidth=2, alpha=0.8)
        lc.set_array(np.array(speeds))
        ax1.add_collection(lc)

        # 添加颜色条
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax1)
        cbar.set_label("Speed Limit (m/s)", rotation=270, labelpad=20)

    except Exception as e:
        ax1.plot(interpolated_points[:, 0], interpolated_points[:, 1], "r-", label="Interpolated Path (No Speed Data)")

    # 绘制车辆框体
    corners = np.array([[corner.x, corner.y] for corner in box_on_agent_path.all_corners()])
    ax1.plot(np.append(corners[:, 0], corners[0, 0]), np.append(corners[:, 1], corners[0, 1]), "purple", label="Agent Bounding Box", linewidth=1.5)
    _plot_single_vehicle(agent, agent_track_token, agent.center.x, agent.center.y, agent.center.heading, agent.box.length, agent.box.width)

    ax1.axis("equal")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # ==================== 子图2：曲率与限速对比 ====================
    # - 原始曲率：红色虚线 (`color='tab:red', linestyle=':'`)
    # - 平滑曲率：蓝色实线 (`color='tab:blue', linestyle='-'`)
    # - 限速曲线：绿色实线 (`color='tab:green', linestyle='-'`)
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_title("Curvature Analysis & Speed Profile")
    ax2.set_xlabel("Path Progress (m)")
    ax2.set_ylabel("Curvature (1/m)", color="tab:red")
    ax2.grid(True, alpha=0.3)

    # 获取曲率数据
    progress_list = [p.progress for p in agent_path.get_sampled_path()]
    # 绘制原始曲率（红色虚线）
    raw_curvatures = agent_path.get_curvature_list(use_smoothed=False)
    line_raw = ax2.plot(progress_list, raw_curvatures, color="tab:red", linestyle=":", label="Raw Curvature", alpha=0.6, linewidth=1.0)

    # 绘制平滑曲率（蓝色实线）
    smoothed_curvatures = agent_path.get_curvature_list(use_smoothed=True)
    line_smooth = ax2.plot(progress_list, smoothed_curvatures, color="tab:blue", linestyle="-", label="Smoothed Curvature", linewidth=1.8)

    # 绘制限速曲线（绿色实线）
    # 创建一个新的副坐标轴 `ax2b` 来绘制限速曲线
    ax2b = ax2.twinx()  # 创建共享x轴的另一个坐标轴
    speeds = [agent_path.get_speed_limit_at_progress(p) for p in progress_list]
    line_speed = ax2b.plot(progress_list, speeds, color="tab:green", label="Speed Limit", linestyle="-", linewidth=1.8)

    # 合并图例
    lines = line_raw + line_smooth + line_speed
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left", ncol=3)

    # 设置坐标轴范围优化
    max_curv = max(max(raw_curvatures), max(smoothed_curvatures)) * 1.2
    ax2.set_ylim(0, max_curv)
    ax2b.set_ylim(min(speeds) * 0.8, max(speeds) * 1.1)

    # 自动调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 保留主标题空间

    import datetime

    # 保存图像
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = os.path.join(output_dir, f"{scenario_name}_agent_path_add_speed_limit_curvatures{agent_track_token}_{timestamp}.svg")
    plt.savefig(figure_filename, dpi=600, bbox_inches="tight")
    plt.close()

    logger.info(f"Agent analysis report saved: {figure_filename}")


def build_idm_agents_on_map_rails(
    target_velocity: float,
    min_gap_to_lead_agent: float,
    headway_time: float,
    accel_max: float,
    decel_max: float,
    minimum_path_length: float,
    scenario: AbstractScenario,
    open_loop_detections_types: List[TrackedObjectType],
) -> Tuple[UniqueIDMAgents, OccupancyMap]:
    """
    Build unique agents from a scenario. InterpolatedPaths are created for each agent according to their driven path

    :param target_velocity: Desired velocity in free traffic [m/s]
    :param min_gap_to_lead_agent: Minimum relative distance to lead vehicle [m]
    :param headway_time: Desired time headway. The minimum possible time to the vehicle in front [s]
    :param accel_max: maximum acceleration [m/s^2]
    :param decel_max: maximum deceleration (positive value) [m/s^2]
    :param minimum_path_length: [m] The minimum path length
    :param scenario: scenario
    :param open_loop_detections_types: The open-loop detection types to include.
    :return: a dictionary of IDM agent uniquely identified by a track_token
    """
    unique_agents: UniqueIDMAgents = {}

    # step 1: 提取所有的检测结果
    detections = scenario.initial_tracked_objects
    map_api = scenario.map_api
    ego_agent = scenario.initial_ego_state.agent

    # step 2: 构建初始的 init_agent_occupancy
    # 将所有的 open_loop_detections （一般为静态障碍物） + EGO ：构建为  STRTreeOccupancyMap
    open_loop_detections = detections.tracked_objects.get_tracked_objects_of_types(open_loop_detections_types)
    # An occupancy map used only for collision checking
    init_agent_occupancy = STRTreeOccupancyMapFactory.get_from_boxes(open_loop_detections)
    init_agent_occupancy.insert(geometry_id=ego_agent.token, geometry=ego_agent.box.geometry)

    # step 3: 构造一个 全局地图，用于每个 agent 搜索路径；# !这个过程相似与： MineSim/devkit/sim_engine/planning/planner/abstract_idm_planner.py
    route_path_planner: GlobalRoutePathPlanner = create_route_path_planner(
        map_name=map_api.map_name,
        semantic_map=map_api.semantic_map,
        minesim_graph=create_minesim_graph(map_name=map_api.map_name, semantic_map=map_api.semantic_map),
        is_initial_path_stitched=True,
    )

    # step 4: 依次构建每个 smart agent 的 occupancy map ----- smart_dynamic_agents  循环处理每个跟踪到的车辆代理（`agent`）；
    # Initialize occupancy map
    occupancy_map = STRTreeOccupancyMap({})
    desc = "Converting detections to smart agents"

    agent: Agent
    # -4.1 **筛选条件**：仅处理类型为车辆的对象。====================
    smart_dynamic_agents = detections.tracked_objects.get_dynamic_smart_agents()
    for agent in tqdm(smart_dynamic_agents, desc=desc, leave=False):
        # filter for only vehicles
        if agent.track_token not in unique_agents:
            # ==================== 4.2: 为 agent 匹配合适的参考路径 ====================
            # 4.2.1: 将agent的未来轨迹整合为一条 path waypoints;
            smart_dynamic_agents[0]._metadata.track_id_minesim
            for dynamic_obstacle_vehicle in scenario.minesim_tracked_metadata.dynamic_obstacle_vehicles:
                if agent.metadata.track_id_minesim == dynamic_obstacle_vehicle.vehicle_id:
                    true_path_state_se2 = consolidate_agent_trajectory_to_path(dynamic_obstacle_vehicle=dynamic_obstacle_vehicle)
            pass

            # 4.2.2: 根据 true_path_state_se2 终点查找一条路径作为未来路径拼接
            # IF agent 从 ROAD polygon 开始， 从 ROAD polygon 结束：查找 agent 起点 pose, 终点 pose;
            route_path_waypoints = route_path_planner.get_agent_feasible_global_route_path(start_pose_statese2=true_path_state_se2[-1])
            # 4.2.3 拼接： true_path_state_se2 最终点 + 未来路径 and true_path_state_se2
            stitched_route_waypoints = route_path_planner.stitch_initial_route_waypoints(
                vehicle_token=agent.track_token,
                start_pose_statese2=true_path_state_se2[-1],
                route_path_waypoints=route_path_waypoints,
                sampling_interval=6,
                ref_path_waypoints_interval=route_path_planner.semantic_map.reference_path[2]["waypoint_sampling_interval_meter"],
            )
            # 4.2.4 拼接： true_path_state_se2 + stitched_route_waypoints
            agent_route_path_waypoints: List[StateSE2] = true_path_state_se2[:-2]

            # Check if stitched_route_waypoints is a valid, non-empty array
            if stitched_route_waypoints is not None and len(stitched_route_waypoints) > 0:
                # 遍历 stitched_route_waypoints 并添加到 agent_route_path_waypoints 中
                for waypoint in stitched_route_waypoints:
                    agent_route_path_waypoints.append(StateSE2(x=waypoint[0], y=waypoint[1], heading=waypoint[2]))
                logging.info(f"#log# get vehicle_token={agent.track_token} 的 参考路径点，用于交互式仿真。")
            else:
                logging.error(f"#log# stitched_route_waypoints is empty or invalid, error!!!")

            # route, progress = get_starting_segment(agent, map_api)
            agent_path = create_path_from_se2(states=agent_route_path_waypoints)
            # agent_path_linestring = path_to_linestring(path=agent_route_path_waypoints)

            # Ignore agents that a route path cannot be built for
            if agent_path is None:
                continue

            # 4.3 路径占用空间映射：匹配一条合适的 参考路径 的 开始 waypoint ，构造 path occupancy_map ；后续用于计算与其它agent的 碰撞关系
            # Snap agent to baseline path
            state_on_agent_path, init_progress = agent_path.get_nearest_pose_from_position(agent.center.point)
            box_on_agent_path = OrientedBox.from_new_pose(
                agent.box, StateSE2(state_on_agent_path.x, state_on_agent_path.y, state_on_agent_path.heading)
            )

            # 4.4 碰撞检测：
            # Check for collision 检测代理是否与其他对象发生碰撞，如果 未发生碰撞则跳过该代理。
            if not init_agent_occupancy.intersects(box_on_agent_path.geometry).is_empty():
                continue

            # 4.5 更新占用地图：将代理的空间占用信息添加到 `init_agent_occupancy` 和 `occupancy_map` 中。
            # Add to init_agent_occupancy for collision checking
            init_agent_occupancy.insert(geometry_id=agent.track_token, geometry=box_on_agent_path.geometry)
            # Add to occupancy_map to pass on to IDMAgentManger
            occupancy_map.insert(geometry_id=agent.track_token, geometry=box_on_agent_path.geometry)

            # 4.6 **处理代理速度**：
            # - 如果代理的速度数据为空（例如 `NaN`），则使用 ego 车辆的速度来替代。
            # - 否则，基于代理的速度数据设置代理的速度。
            # Project velocity into local frame
            if np.isnan(agent.velocity.array).any():
                ego_state = scenario.get_ego_state_at_iteration(0)
                logger.debug(f"#log# Agents has nan velocity. Setting velocity to ego's velocity of " f"{ego_state.dynamic_car_state.speed}")
                velocity = StateVector2D(ego_state.dynamic_car_state.speed, 0.0)
            else:
                velocity = StateVector2D(np.hypot(agent.velocity.x, agent.velocity.y), 0)

            # 5. **创建 IDM 代理**：
            # - 对每个代理创建 `IDMAgent` 对象，初始化其状态、路径、策略等信息。具体的策略使用 `IDMPolicy` 来配置，涉及的参数有目标速度、最小跟车距离、头距时间等。
            initial_state = IDMInitialState(
                metadata=agent.metadata,
                tracked_object_type=agent.tracked_object_type,
                box=box_on_agent_path,
                velocity=velocity,
                path_progress=init_progress,
                predictions=agent.predictions,
            )
            # target_velocity = route.speed_limit_mps or target_velocity
            unique_agents[agent.track_token] = IDMAgent(
                start_iteration=0,
                initial_state=initial_state,
                path=agent_path,
                policy=IDMPolicy(
                    target_velocity=target_velocity,
                    min_gap_to_lead_agent=min_gap_to_lead_agent,
                    headway_time=headway_time,
                    accel_max=accel_max,
                    decel_max=decel_max,
                ),
            )
            # ! DEBUG plot
            if False:
                _debug_plot_agent_discrete_path_waypoints(
                    agent_track_token=agent.track_token,  # str
                    true_path_state_se2=true_path_state_se2,  # List[StateSe2]
                    stitched_route_waypoints=stitched_route_waypoints,  # np.array
                    agent_path=agent_path,  # InterpolatedPath
                    box_on_agent_path=box_on_agent_path,
                    agent=agent,
                    scenario_name=scenario.scenario_name,
                )

            if False:
                _debug_plot_agent_discrete_path_waypoints_with_speed_limit(
                    agent_track_token=agent.track_token,  # str
                    true_path_state_se2=true_path_state_se2,  # List[StateSe2]
                    stitched_route_waypoints=stitched_route_waypoints,  # np.array
                    agent_path=agent_path,  # InterpolatedPath
                    box_on_agent_path=box_on_agent_path,
                    agent=agent,
                )
            if True:
                _debug_plot_agent_discrete_path_waypoints_with_speed_limit_and_curvatures(
                    agent_track_token=agent.track_token,  # str
                    true_path_state_se2=true_path_state_se2,  # List[StateSe2]
                    stitched_route_waypoints=stitched_route_waypoints,  # np.array
                    agent_path=agent_path,  # InterpolatedPath
                    box_on_agent_path=box_on_agent_path,
                    agent=agent,
                    scenario_name=scenario.scenario_name,
                )

    #  - 返回生成的 `unique_agents` 字典和 `occupancy_map` 占用地图。
    return unique_agents, occupancy_map
